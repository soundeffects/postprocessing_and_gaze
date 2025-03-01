import numpy
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from scipy.special import logsumexp, rel_entr
from pathlib import Path
from unisal.unisal.train import Trainer

def list_relative_paths(path: Path) -> list[Path]:
    """
    List all files in a given directory relative to the given path.
    """
    return [ filename.relative_to(path) for filename in path.glob('*') ]

def normalize_image(image: numpy.ndarray, rescale: bool = False) -> numpy.ndarray:
    """
    Normalize the intensity values of an image to be within the range of 
    [0, 255]. The 'rescale' parameter will move the relative scale such that the
    minimum intensity value of the image will be zero.
    """
    min_value = 0
    if rescale or image.min() < 0:
        min_value = image.min()
    return ((image - min_value) / (image.max() - min_value) * 255).astype(numpy.uint8)

def clip_image(image: numpy.ndarray, min_value: float = 0, max_value: float = 255) -> numpy.ndarray:
    """
    Clip the intensity values of an image to be within the range
    [min_value, max_value].
    """
    return numpy.clip(image, min_value, max_value).astype(numpy.uint8)

def is_pdf(function: numpy.ndarray) -> bool:
    """
    Check if a function is a probability distribution.
    """
    epsilon = 1e-4
    return abs(function.sum() - 1) < epsilon and function.min() >= 0 and function.max() <= 1

def to_log_density(saliency_map: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a saliency map to a log density distribution.
    """
    log_density = saliency_map - logsumexp(saliency_map)
    assert is_pdf(numpy.exp(log_density))
    return log_density

def kl_div(p: numpy.ndarray, q: numpy.ndarray) -> float:
    """
    Compute the KL divergence between two log density distributions, normalized
    by the size of the distributions.
    """
    assert p.shape == q.shape

    p_exp = numpy.exp(p)
    q_exp = numpy.exp(q)
    assert is_pdf(p_exp)
    assert is_pdf(q_exp)

    return rel_entr(p_exp, q_exp).sum() / p_exp.size

def information_gain(p: numpy.ndarray) -> float:
    """
    Compute the information gain of a gaze density distribution over an image-
    independent distribution called the center bias.

    Reference: https://doi.org/10.1073/pnas.1510393112
    """
    center_bias = numpy.load('src/centerbias_mit1003.npy')
    scaling_shape = (p.shape[0] / center_bias.shape[0], p.shape[1] / center_bias.shape[1])
    center_bias = to_log_density(zoom(center_bias, scaling_shape, order=0, mode='nearest'))
    return kl_div(p, center_bias)

def image_difference(image_1: numpy.ndarray, image_2: numpy.ndarray) -> float:
    """
    Compute the summed pixel-wise squared difference between two images,
    normalized by the number of pixels.
    """
    assert image_1.shape == image_2.shape
    return ((image_1 - image_2) ** 2).sum() / image_1.size

def gaussian_noise(image: numpy.ndarray, sigma: float = 16) -> numpy.ndarray:
    """
    Add Gaussian noise to an image.
    """
    signal = normalize_image(image)
    noise = numpy.random.normal(0, sigma, image.shape)
    return clip_image(signal + noise)

def gaussian_blur(image: numpy.ndarray, sigma: float = 1) -> numpy.ndarray:
    """
    Apply a Gaussian blur to an image.
    """
    signal = normalize_image(image)
    return clip_image(gaussian_filter(signal, sigma))

def iterate_filter(image: numpy.ndarray, filter: callable, iterations: int) -> numpy.ndarray:
    """
    Apply a filter to an image a given number of times.
    """
    data = image
    for _ in range(iterations):
        data = filter(data)
    return data

def apply_filters(data_directory: str, filters: list[callable], iterations: int, verbose: bool = False):
    """
    Apply a list of filters to all images in a given directory.
    """
    for dataset in Path(data_directory).resolve().iterdir():
        base_images = dataset / 'base' / 'images'
        for _, _, filenames in base_images.walk():
            for filename in filenames:
                image_path = base_images / filename
                if image_path.suffix in ['.jpg', '.jpeg', '.png']:
                    image = Image.open(image_path)
                    image_data = numpy.array(image)
                    for filter in filters:
                        missed_iterations = 1
                        for i in range(iterations):
                            new_image_path = dataset / f"{filter.__name__}_{i}" / 'images' / filename
                            if new_image_path.exists():
                                missed_iterations += 1
                                continue
                            if verbose:
                                print(f"Creating {new_image_path}")
                            image_data = iterate_filter(image_data, filter, missed_iterations)
                            missed_iterations = 1
                            Image.fromarray(image_data).save(new_image_path)

def generate_saliency_maps(data_directory: str, unisal_path: str, verbose: bool = False):
    """
    Generate saliency maps for all images using pre-trained gaze density
    prediction models.
    """
    unisal = Trainer.init_from_cfg_dir(Path(unisal_path))

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
        base_path = dataset / 'base'
        for filter_path in dataset.iterdir():
            if filter_path == base_path:
                continue
            image_set = set(list_relative_paths(filter_path / 'images'))
            saliency_set = set(list_relative_paths(filter_path / 'saliency'))
            for image_path in image_set.intersection(saliency_set):
                image = Image.open(filter_path / 'images' / image_path)
                saliency = Image.open(filter_path / 'saliency' / image_path)
                image_data = numpy.array(image)
                saliency_data = numpy.array(saliency)
                print(image_data.shape)
                print(saliency_data.shape)
            if verbose:
                print(f"Computing metrics for {filter_path}")


generate_saliency_maps('data', 'unisal/training_runs/pretrained_unisal')

"""
saliency = Image.open('data/MIT300/base/saliency/i1.jpg')
saliency.show()
saliency_data = numpy.array(saliency)
saliency_density = saliency_data / numpy.sum(saliency_data)
Image.fromarray(normalize_image(saliency_density)).show()
print(is_pdf(numpy.exp(to_log_density(saliency_data))))
"""

# First, we apply image filters to all images in the dataset

# Then, we apply to model to base and filtered images to get saliency maps

# Then, we convert the saliency maps to log densities, and we compare to the base log density

