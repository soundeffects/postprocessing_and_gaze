from argparse import ArgumentParser
from csv import DictReader, DictWriter, writer
from filters import all_filters, filter_names
from numpy import ndarray, exp, sum, load, array, mean, median, std
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp, rel_entr
from torch import cuda, backends, set_default_device
from typing import Optional
from unisal.unisal.train import Trainer

# Description of command line arguments
parser = ArgumentParser(description='Compute predicted saliency metrics when \
applying post processing image filters to a given set of images.')
parser.add_argument('--data-directory', type=str, default='data',
    help='The directory containing the image datasets to study.')
parser.add_argument('--filter-subdivisions', type=int, default=10,
    help='Specify how many subdivisions of filter strength (between minimum \
    and maximum strength as defined in filter code) with which to apply \
    filters.')
parser.add_argument('--unisal-path', type=str, default='unisal/training_runs/pretrained_unisal',
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

# The error tolerance for the sum of a probability distribution, which should be
# 1.0
PDF_EPSILON = 1e-4

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
    'information_gain_difference',
    'information_gain_difference_log'
]

def list_relative_paths(path: Path) -> list[Path]:
    """
    List all files in a given directory relative to the given path.
    """
    return [ filename.relative_to(path) for filename in path.glob('*') ]

def is_pdf(function: ndarray) -> bool:
    """
    Check if a function is a probability distribution.
    """
    return abs(function.sum() - 1) < PDF_EPSILON and function.min() >= 0 and function.max() <= 1

def to_density(saliency_map: ndarray, log: bool = False) -> ndarray:
    """
    Convert a saliency map to a density distribution. If the 'log' parameter is
    set to 'True', the saliency map will be converted to a log density
    distribution.
    """
    density = saliency_map - logsumexp(saliency_map) if log \
        else saliency_map / sum(saliency_map)
    assert is_pdf(exp(density) if log else density)
    return density

def nonzero_pdf(pdf: ndarray) -> float:
    """
    Ensure that no sample of the PDF is zero (for log calculations) while
    keeping the PDF sum at 1.0 within the PDF error tolerance.
    """
    # The remaining tolerance per sample is the remaining tolerance of the
    # whole PDF, divided by the number of samples, halved to account for
    # floating point error.
    epsilon = (PDF_EPSILON + 1.0 - sum(pdf)) / pdf.size / 2.0
    density = pdf + epsilon
    assert is_pdf(density)
    return density

def kl_div(p: ndarray, q: ndarray, log: bool = False) -> float:
    """
    Compute the KL divergence between two density distributions, normalized
    by the size of the distributions. If the 'log' parameter is set to 'True',
    the distributions are assumed to be log density distributions.
    """
    assert p.shape == q.shape
    p_div = exp(p) if log else nonzero_pdf(p)
    q_div = exp(q) if log else nonzero_pdf(q)
    assert is_pdf(p_div)
    assert is_pdf(q_div)
    return rel_entr(p_div, q_div).sum() / p_div.size

def information_gain(p: ndarray, log: bool = False) -> float:
    """
    Compute the information gain of a gaze density distribution over an image-
    independent distribution called the center bias. If the 'log' parameter is
    set to 'True', the distributions are assumed to be log density
    distributions.
    """
    center_bias = load('src/centerbias_mit1003.npy')
    scaling_shape = (p.shape[0] / center_bias.shape[0], p.shape[1] / center_bias.shape[1])
    center_bias = to_density(zoom(center_bias, scaling_shape, order=0, mode='nearest'), log)
    return kl_div(p, center_bias, log)

def image_difference(image_1: ndarray, image_2: ndarray) -> float:
    """
    Compute the summed pixel-wise squared difference between two images,
    normalized by the number of pixels.
    """
    assert image_1.shape == image_2.shape
    return ((image_1 - image_2) ** 2).sum() / image_1.size

def iterate_filter(image: ndarray, filter: callable, iterations: int) -> ndarray:
    """
    Apply a filter to an image a given number of times.
    """
    data = image
    for _ in range(iterations):
        data = filter(data)
    return data

def apply_filters(data_directory: str, filters: list[callable], filter_subdivisions: int, verbose: bool = False):
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
                        missed_iterations = 1
                        for i in range(filter_subdivisions):
                            new_image_path = dataset / f"{filter.__name__}_{i}" / 'images' / filename
                            if new_image_path.exists():
                                missed_iterations += 1
                                continue
                            if verbose:
                                print(f"Creating {new_image_path} with {filter.__name__} filter")
                            image_data = iterate_filter(image_data, filter, missed_iterations)
                            missed_iterations = 1
                            Image.fromarray(image_data).save(new_image_path)

def generate_saliency_maps(data_directory: str, unisal_path: str, verbose: bool = False):
    """
    Generate saliency maps for all images using pre-trained gaze density
    prediction models.
    """
    unisal = Trainer.init_from_cfg_dir(Path(unisal_path))
    unisal.model.to(device)
    for dataset in Path(data_directory).resolve().iterdir():
        for root, directories, _ in dataset.walk():
            if 'images' in directories:
                image_set = set(list_relative_paths(root / 'images'))
                saliency_set = set(list_relative_paths(root / 'saliency'))
                if len(image_set.difference(saliency_set)) == 0:
                    continue
                if verbose:
                    print(f"Generating saliency maps for {root}")
                unisal.generate_predictions_from_path(Path(root), is_video=False, source='SALICON')

def compute_metrics(data_directory: str, verbose: bool = False):
    """
    Compute the normalized divergence and information gain between saliency maps
    produced from filtered images and the base saliency map.
    """
    for dataset in Path(data_directory).resolve().iterdir():
        if not dataset.is_dir():
            continue
        base_path = dataset / 'base'
        with open(dataset / 'metrics.csv', 'w') as metrics_file:
            metrics_writer = writer(metrics_file)
            metrics_writer.writerow(['filter_name', 'image_name'] + metrics)
            for filter_path in dataset.iterdir():
                if not filter_path.is_dir() or filter_path == base_path:
                    continue
                # Find intersection of image paths from all four sources being
                # compared
                common_image_paths =\
                set(list_relative_paths(base_path / 'images')) &\
                set(list_relative_paths(filter_path / 'images')) &\
                set(list_relative_paths(base_path / 'saliency')) &\
                set(list_relative_paths(filter_path / 'saliency'))
                for image_path in common_image_paths:
                    base_image = array(Image.open(base_path / 'images' / image_path))
                    filtered_image = array(Image.open(filter_path / 'images' / image_path))
                    base_saliency = array(Image.open(base_path / 'saliency' / image_path))
                    filtered_saliency = array(Image.open(filter_path / 'saliency' / image_path))
                    base_density = to_density(base_saliency)
                    base_density_log = to_density(base_saliency, log=True)
                    filtered_density = to_density(filtered_saliency)
                    filtered_density_log = to_density(filtered_saliency, log=True)
                    image_difference_value = image_difference(base_image, filtered_image)
                    saliency_divergence = kl_div(base_density, filtered_density)
                    saliency_divergence_log = kl_div(base_density_log, filtered_density_log, log=True)
                    base_information_gain = information_gain(base_density)
                    base_information_gain_log = information_gain(base_density_log, log=True)
                    filtered_information_gain = information_gain(filtered_density)
                    filtered_information_gain_log = information_gain(filtered_density_log, log=True)
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
                        filtered_information_gain - base_information_gain,
                        filtered_information_gain_log - base_information_gain_log
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
            if not dataset.is_dir() and (dataset / 'metrics.csv').exists():
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