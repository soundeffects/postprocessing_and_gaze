from argparse import ArgumentParser
from csv import writer
from pathlib import Path
from PIL import Image
from numpy import array, mean
from torch import cuda, backends, set_default_device
from unisal.unisal.train import Trainer
from utilities import common_images, list_relative_paths, to_density, kl_div, information_asymmetry

# Description of command line arguments
parser = ArgumentParser(description='Compute predicted saliency metrics for \
a set of images and their transformations, and check their similiarity to \
measured gaze distributions.')
parser.add_argument('--data-directory', type=str, default='../performance_data',
    help='The directory containing the image datasets to study.')
parser.add_argument('--unisal-path', type=str,
    default='unisal/training_runs/pretrained_unisal',
    help='The path to the pre-trained UNISAL model.')
parser.add_argument('--verbose', action='store_true', default=False,
    help='Verbose logging.')
parser.add_argument('--no-compute-saliency', action='store_true', default=False,
    help='Do not perform the `compute_saliency` step. Will prevent \
    `compute_metrics` step if `compute_saliency` has not been run before and \
    stored results in the data directory.')
parser.add_argument('--no-compute-metrics', action='store_true', default=False,
    help='Do not perform the `compute_metrics` step.')
args = parser.parse_args()

# Set device to CUDA or MPS if available, otherwise use CPU
device = 'cpu'
if cuda.is_available():
    device = 'cuda'
elif hasattr(backends, 'mps') and backends.mps.is_available():
    device = 'mps'
set_default_device(device)

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
                if verbose:
                    print(f"Generating saliency maps for {root}")
                unisal.generate_predictions_from_path(Path(root), is_video=False, source='MIT300')

metrics = [
    'divergence',
    'divergence_log',
    'reference_gap',
    'reference_gap_log',
    'information_asymmetry',
    'information_asymmetry_log',
    'information_asymmetry_gap',
    'information_asymmetry_gap_log'
]

def compute_metrics(data_directory: str, verbose: bool = False):
    """
    Compute metrics for all images in the data directory.
    """
    data_path = Path(data_directory).resolve()
    if (data_path / 'metrics.csv').exists():
        return
    with open(data_path / 'metrics.csv', 'w') as metrics_file:
        metrics_writer = writer(metrics_file)
        metrics_writer.writerow(['transformation'] + metrics)
        metric_cache = {}
        for root, directories, _ in data_path.walk():
            if 'saliency' in directories and 'truth' in directories:
                paths = [root / 'saliency', root / 'truth']
                common_set = common_images(paths)
                for image in common_set:
                    prediction = array(Image.open(paths[0] / image))
                    truth = array(Image.open(paths[1] / image))
                    predicted_density = to_density(prediction)
                    true_density = to_density(truth)
                    predicted_density_log = to_density(prediction, log=True)
                    true_density_log = to_density(truth, log=True)
                    if not root.name in metric_cache:
                        metric_cache[root.name] = {}
                    if not 'divergences' in metric_cache[root.name]:
                        metric_cache[root.name]['divergences'] = []
                    metric_cache[root.name]['divergences'].append(kl_div(predicted_density, true_density, normalize=False))
                    if not 'divergences_log' in metric_cache[root.name]:
                        metric_cache[root.name]['divergences_log'] = []
                    metric_cache[root.name]['divergences_log'].append(kl_div(predicted_density_log, true_density_log, log=True, normalize=False))
                    if not 'information_asymmetries' in metric_cache[root.name]:
                        metric_cache[root.name]['information_asymmetries'] = []
                    metric_cache[root.name]['information_asymmetries'].append(information_asymmetry(predicted_density, true_density, normalize=False))
                    if not 'information_asymmetries_log' in metric_cache[root.name]:
                        metric_cache[root.name]['information_asymmetries_log'] = []
                    metric_cache[root.name]['information_asymmetries_log'].append(information_asymmetry(predicted_density_log, true_density_log, log=True, normalize=False))
                metric_cache[root.name]['divergence'] = mean(metric_cache[root.name]['divergences'])
                metric_cache[root.name]['divergence_log'] = mean(metric_cache[root.name]['divergences_log'])
                metric_cache[root.name]['information_asymmetry'] = mean(metric_cache[root.name]['information_asymmetries'])
                metric_cache[root.name]['information_asymmetry_log'] = mean(metric_cache[root.name]['information_asymmetries_log'])
        for transformation, transformation_data in metric_cache.items():
            transformation_data['reference_gap'] = transformation_data['divergence'] - metric_cache['Reference']['divergence']
            transformation_data['reference_gap_log'] = transformation_data['divergence_log'] - metric_cache['Reference']['divergence_log']
            transformation_data['information_asymmetry_gap'] = transformation_data['information_asymmetry'] - metric_cache['Reference']['information_asymmetry']
            transformation_data['information_asymmetry_gap_log'] = transformation_data['information_asymmetry_log'] - metric_cache['Reference']['information_asymmetry_log']
            metrics_writer.writerow([transformation] + [transformation_data[metric] for metric in metrics])

# Run all computations
if not args.no_compute_saliency:
    generate_saliency_maps(args.data_directory, args.unisal_path, args.verbose)
if not args.no_compute_metrics:
    compute_metrics(args.data_directory, args.verbose)