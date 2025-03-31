from argparse import ArgumentParser
from pathlib import Path
from torch import cuda, backends, set_default_device
from unisal.unisal.train import Trainer

# Description of command line arguments
parser = ArgumentParser(description='Compute predicted saliency metrics for \
a set of images and their transformations, and check their similiarity to \
measured gaze distributions.')
parser.add_argument('--data-directory', type=str, default='performance_data',
    help='The directory containing the image datasets to study.')
parser.add_argument('--unisal-path', type=str,
    default='unisal/training_runs/pretrained_unisal',
    help='The path to the pre-trained UNISAL model.')
parser.add_argument('--verbose', action='store_true', default=False,
    help='Verbose logging.')
parser.add_argument('--no-compute-saliency', action='store_true', default=False,
    help='Do not perform the `compute_saliency` step. Will prevent \
    `compute_metrics` and `aggregate_metrics` steps if `compute_saliency` has \
    not been run before and stored results in the data directory.')
parser.add_argument('--no-compute-metrics', action='store_true', default=False,
    help='Do not perform the `compute_metrics` step. Will prevent \
    `aggregate_metrics` step if `compute_metrics` has not been run before and \
    stored results in the data directory.')
parser.add_argument('--no-aggregate-metrics', action='store_true', default=False,
    help='Do not perform the `aggregate_metrics` step. Will store results in \
    the data directory.')
args = parser.parse_args()

# Set device to CUDA or MPS if available, otherwise use CPU
device = 'cpu'
if cuda.is_available():
    device = 'cuda'
elif hasattr(backends, 'mps') and backends.mps.is_available():
    device = 'mps'
set_default_device(device)

def list_relative_paths(path: Path) -> list[Path]:
    """
    List all files in a given directory relative to the given path.
    """
    return [ filename.relative_to(path) for filename in path.glob('*') ]

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

def compute_metrics(data_directory: str, verbose: bool = False):
    """
    Compute metrics for all images in the data directory.
    """
    pass

def aggregate_metrics(data_directory: str, verbose: bool = False):
    """
    Aggregate metrics for all images in the data directory. """
    pass

# Run all computations
if not args.no_compute_saliency:
    generate_saliency_maps(args.data_directory, args.unisal_path, args.verbose)
if not args.no_compute_metrics:
    compute_metrics(args.data_directory, args.verbose)
if not args.no_aggregate_metrics:
    aggregate_metrics(args.data_directory, args.verbose)