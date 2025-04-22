from argparse import ArgumentParser
from csv import DictReader, DictWriter, writer
from scripts.filters import all_filters, filter_names
from numpy import array, mean, median, std
from pathlib import Path
from PIL import Image
from torch import cuda, backends, set_default_device
from typing import Optional
from unisal.unisal.train import Trainer
from utilities import common_images, list_relative_paths, to_density, image_difference, kl_div, information_gain

# Description of command line arguments
parser = ArgumentParser(description='Compute predicted saliency metrics when \
applying post processing image filters to a given set of images.')
parser.add_argument('--data-directory', type=str, default='heuristics_data',
    help='The directory containing the image datasets to study.')
parser.add_argument('--filter-subdivisions', type=int, default=10,
    help='Specify how many subdivisions of filter strength (between minimum \
    and maximum strength as defined in filter code) with which to apply \
    filters.')
parser.add_argument('--unisal-path', type=str,
    default='unisal/training_runs/pretrained_unisal',
    help='The path to the pre-trained UNISAL model.')
parser.add_argument('--verbose', action='store_true', default=False,
    help='Verbose logging.')
parser.add_argument('--no-apply-filters', action='store_true', default=False,
    help='Do not perform the `apply_filters` step. Will prevent \
    `compute_saliency`, `compute_metrics`, and `aggregate_metrics` steps if \
    `apply_filters` has not been run before and stored results in the data \
    directory.')
parser.add_argument('--no-compute-saliency', action='store_true', default=False,
    help='Do not perform the `compute_saliency` step. Will prevent \
    `computer_metrics` and `aggregate_metrics` steps if `compute_saliency` has \
    not been run before and stored results in the data directory.')
parser.add_argument('--no-compute-metrics', action='store_true', default=False,
    help='Do not perform the `compute_metrics` step. Will prevent \
    `aggregate_metrics` step if `compute_metrics` has not been run before and \
    stored results in the data directory.')
parser.add_argument('--no-aggregate-metrics', action='store_true', default=False,
    help='Do not perform the `aggregate_metrics` step. Will store results in \
    the data directory.')
parser.add_argument('--omit-filters', nargs='*', default=[],
    help='Specify a list of filters to omit from the analysis. Filters \
    include: ' + ', '.join(filter_names))
args = parser.parse_args()

# Set device to CUDA or MPS if available, otherwise use CPU
device = 'cpu'
if cuda.is_available():
    device = 'cuda'
elif hasattr(backends, 'mps') and backends.mps.is_available():
    device = 'mps'
set_default_device(device)

def apply_filters(data_directory: str, filters: list[callable], strength_subdivisions: int = 10, verbose: bool = False):
    """
    Apply a list of filters to all images in a given directory.
    """
    for dataset in Path(data_directory).resolve().iterdir():
        if not dataset.is_dir():
            continue
        base_images = dataset / 'base' / 'images'
        for _, _, filenames in base_images.walk():
            for filename in filenames:
                image_path = base_images / filename
                if image_path.suffix in ['.jpg', '.jpeg', '.png']:
                    image = Image.open(image_path)
                    image_data = array(image)
                    for filter in filters:
                        for i in range(1, strength_subdivisions + 1, 1):
                            new_filter_path = dataset / f"{filter.__name__}_{i}"
                            new_image_path = new_filter_path / 'images' / filename
                            if not new_filter_path.exists():
                                (new_filter_path / 'images').mkdir(parents=True)
                            elif new_image_path.exists():
                                continue
                            if verbose:
                                print(f"Creating {new_image_path} with {filter.__name__} filter")
                            Image.fromarray(filter(image_data, i)).save(new_image_path)

def generate_saliency_maps(data_directory: str, unisal_path: str, verbose: bool = False):
    """
    Generate saliency maps for all images using pre-trained gaze density
    prediction models.
    """
    unisal = Trainer.init_from_cfg_dir(Path(unisal_path))
    unisal.model.to(device)
    data_path = Path(data_directory).resolve()
    for dataset in data_path.iterdir():
        for root, directories, _ in dataset.walk():
            if 'images' in directories:
                image_set = set(list_relative_paths(root / 'images'))
                saliency_set = set(list_relative_paths(root / 'saliency'))
                if len(image_set.difference(saliency_set)) == 0:
                    continue
                source = dataset.relative_to(data_path).name
                if verbose:
                    print(f"Generating saliency maps for {root}")
                unisal.generate_predictions_from_path(Path(root), is_video=False, source=source)

# Reusable list of considered metrics
metrics = [
    'image_difference',
    'saliency_divergence',
    'saliency_divergence_log',
    'divergence_per_difference',
    'divergence_per_difference_log',
    'base_information_gain',
    'base_information_gain_log',
    'filtered_information_gain',
    'filtered_information_gain_log',
    'information_asymmetry',
    'information_asymmetry_log',
    'information_asymmetry_per_difference',
    'information_asymmetry_per_difference_log'
]

def compute_metrics(data_directory: str, verbose: bool = False):
    """
    Compute the normalized divergence and information gain between saliency maps
    produced from filtered images and the base saliency map.
    """
    for dataset in Path(data_directory).resolve().iterdir():
        if not dataset.is_dir() or (dataset / 'metrics.csv').exists():
            continue
        base_path = dataset / 'base'
        with open(dataset / 'metrics.csv', 'w') as metrics_file:
            metrics_writer = writer(metrics_file)
            metrics_writer.writerow(['filter_name', 'image_name'] + metrics)
            for filter_path in dataset.iterdir():
                if not filter_path.is_dir() or filter_path == base_path or filter_path.name.startswith('barrel_distortion'):
                    continue
                paths = [base_path / 'images', filter_path / 'images', base_path / 'saliency', filter_path / 'saliency']
                common_image_paths = common_images(paths)
                for image_path in common_image_paths:
                    base_image = array(Image.open(paths[0] / image_path))
                    filtered_image = array(Image.open(paths[1] / image_path))
                    base_saliency = array(Image.open(paths[2] / image_path))
                    filtered_saliency = array(Image.open(paths[3] / image_path))
                    base_density = to_density(base_saliency)
                    base_density_log = to_density(base_saliency, log=True)
                    filtered_density = to_density(filtered_saliency)
                    filtered_density_log = to_density(filtered_saliency, log=True)
                    if base_image.shape != filtered_image.shape:
                        print(f"Image 1 shape: {base_image.shape}, Image 2 shape: {filtered_image.shape}")
                        print(f"Image 1 path: {paths[0] / image_path}, Image 2 path: {paths[1] / image_path}")
                    image_difference_value = image_difference(base_image, filtered_image)
                    saliency_divergence = kl_div(base_density, filtered_density)
                    saliency_divergence_log = kl_div(base_density_log, filtered_density_log, log=True)
                    base_information_gain = information_gain(base_density)
                    base_information_gain_log = information_gain(base_density_log, log=True)
                    filtered_information_gain = information_gain(filtered_density)
                    filtered_information_gain_log = information_gain(filtered_density_log, log=True)
                    information_asymmetry = filtered_information_gain - base_information_gain
                    information_asymmetry_log = filtered_information_gain_log - base_information_gain_log
                    metrics_writer.writerow([
                        filter_path.name,
                        image_path.name,
                        image_difference_value,
                        saliency_divergence,
                        saliency_divergence_log,
                        saliency_divergence / image_difference_value,
                        saliency_divergence_log / image_difference_value,
                        base_information_gain,
                        base_information_gain_log,
                        filtered_information_gain,
                        filtered_information_gain_log,
                        information_asymmetry,
                        information_asymmetry_log,
                        information_asymmetry / image_difference_value,
                        information_asymmetry_log / image_difference_value
                    ])
                if verbose:
                    print(f"Computing metrics for {filter_path}")

def aggregate_metrics(data_directory: str, verbose: bool = False):
    """
    Aggregate metrics both by dataset and filter.
    """
    data_path = Path(data_directory).resolve()
    with open(data_path / 'aggregate.csv', 'w') as aggregate_file:
        aggregates = ['dataset_name', 'filter_name']
        for metric in metrics:
            aggregates.append(f'{metric}_mean')
            aggregates.append(f'{metric}_median')
            aggregates.append(f'{metric}_std')
        aggregate_writer = DictWriter(aggregate_file, fieldnames=aggregates)
        aggregate_writer.writeheader()
        general_data = {}
        def write_rows(filter_data: dict, general_data: Optional[dict] = None, dataset_name: str = 'general'):
            for filter_name, filter_rows in filter_data.items():
                if general_data is not None and filter_name not in general_data:
                    general_data[filter_name] = {}
                aggregate_row = { 'dataset_name': dataset_name, 'filter_name': filter_name }
                for key in filter_rows.keys():
                    aggregate_row[f'{key}_mean'] = mean(array(filter_rows[key]))
                    aggregate_row[f'{key}_median'] = median(array(filter_rows[key]))
                    aggregate_row[f'{key}_std'] = std(array(filter_rows[key]))
                    if general_data is not None:
                        if key not in general_data[filter_name]:
                            general_data[filter_name][key] = []
                        general_data[filter_name][key].extend(filter_rows[key])
                aggregate_writer.writerow(aggregate_row)
        for dataset in data_path.iterdir():
            if not (dataset.is_dir() and (dataset / 'metrics.csv').exists()):
                continue
            if verbose:
                print(f'Reading metrics file for {dataset}')
            filter_data = {}
            filter_data['all_filters'] = {}
            # Read metrics for each dataset into dict
            with open(dataset / 'metrics.csv', 'r') as metrics_file:
                metrics_reader = DictReader(metrics_file)
                for row in metrics_reader:
                    filter_name = row['filter_name']
                    if filter_name not in filter_data:
                        filter_data[filter_name] = {}
                        for key in row.keys():
                            if key not in metrics:
                                continue
                            filter_data[filter_name][key] = []
                            if key not in filter_data['all_filters']:
                                filter_data['all_filters'][key] = []
                    for key in row.keys():
                        if key not in metrics:
                            continue
                        filter_data[filter_name][key].append(float(row[key]))
                        filter_data['all_filters'][key].append(float(row[key]))
            if verbose:
                print(f'Writing rows for {dataset}')
            write_rows(filter_data, general_data, dataset.name)
        if verbose:
            print('Writing general rows')
        write_rows(general_data)

# Run computation steps
if not args.no_apply_filters:
    filters = list(filter(lambda x: x not in args.omit_filters, all_filters))
    apply_filters(args.data_directory, filters, args.filter_subdivisions, args.verbose)
if not args.no_compute_saliency:
    generate_saliency_maps(args.data_directory, args.unisal_path, args.verbose)
if not args.no_compute_metrics:
    compute_metrics(args.data_directory, args.verbose)
if not args.no_aggregate_metrics:
    aggregate_metrics(args.data_directory, args.verbose)