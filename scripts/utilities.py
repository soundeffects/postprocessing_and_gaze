from numpy import ndarray, exp, sum, load, float32
from pathlib import Path
from torch import cuda, backends, set_default_device, Tensor, tensor as torch_tensor
from scipy.ndimage import zoom
from scipy.special import logsumexp, rel_entr

# The error tolerance for the sum of a probability distribution, which should be
# 1.0
PDF_EPSILON = 1e-4

# Storing a global PyTorch device
DEVICE = None

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

def kl_div(p: ndarray, q: ndarray, log: bool = False, normalize: bool = True) -> float:
    """
    Compute the KL divergence between two density distributions, normalized
    by the size of the distributions. If the 'log' parameter is set to 'True',
    the distributions are assumed to be log density distributions. If the
    'normalize' parameter is set to 'True', the KL divergence will be normalized
    by the size of the distributions.
    """
    assert p.shape == q.shape
    p_div = exp(p) if log else nonzero_pdf(p)
    q_div = exp(q) if log else nonzero_pdf(q)
    assert is_pdf(p_div)
    assert is_pdf(q_div)
    divergence = rel_entr(p_div, q_div).sum()
    return divergence / p_div.size if normalize else divergence

def load_centerbias(shape: tuple[int, int], log: bool = False):
    """
    Load the center bias from the MIT 1003 dataset and scale it to the given
    shape. If the 'log' parameter is set to 'True', the center bias will be
    converted to a log density distribution.
    """
    center_bias = load('scripts/centerbias_mit1003.npy')
    scaling_shape = (shape[0] / center_bias.shape[0], shape[1] / center_bias.shape[1])
    return to_density(zoom(center_bias, scaling_shape, order=0, mode='nearest'), log)

def information_gain(p: ndarray, log: bool = False, normalize: bool = True) -> float:
    """
    Compute the information gain of a gaze density distribution over an image-
    independent distribution called the center bias. If the 'log' parameter is
    set to 'True', the distributions are assumed to be log density
    distributions. If the 'normalize' parameter is set to 'True', the
    information gain will be normalized by the size of the distributions.
    """
    center_bias = load_centerbias(p.shape, log)
    return kl_div(p, center_bias, log, normalize)

def image_difference(image_1: ndarray, image_2: ndarray, normalize: bool = True) -> float:
    """
    Compute the summed pixel-wise squared difference between two images,
    normalized by the number of pixels.
    """
    assert image_1.shape == image_2.shape
    difference = ((image_1 - image_2) ** 2).sum()
    return difference / image_1.size if normalize else difference

def common_images(paths: list[Path]) -> set[Path]:
    """
    Find the intersection of image paths from all given paths.
    """
    common_set = set(list_relative_paths(paths[0]))
    for path in paths[1:]:
        common_set &= set(list_relative_paths(path))
    return common_set

def information_asymmetry(p: ndarray, q: ndarray, log: bool = False, normalize: bool = True) -> float:
    """
    Compute the information asymmetry between two density distributions. If the
    'log' parameter is set to 'True', the distributions are assumed to be log
    density distributions. If the 'normalize' parameter is set to 'True', the
    information asymmetry will be normalized by the size of the distributions.
    """
    return information_gain(p, log, normalize) - information_gain(q, log, normalize)

def pytorch_device() -> str:
    """
    Set the PyTorch device to use GPU/NPU acceleration if available.
    """
    global DEVICE
    if DEVICE:
        return DEVICE
    DEVICE = 'cpu'
    if cuda.is_available():
        DEVICE = 'cuda'
    elif hasattr(backends, 'mps') and backends.mps.is_available():
        DEVICE = 'mps'
    set_default_device(DEVICE)
    print(f'Using {DEVICE} as pytorch device')
    return DEVICE

def tensor(data: ndarray) -> Tensor:
    """
    Create a PyTorch tensor, and automatically convert the floating point
    precision to float32.
    """
    pytorch_device()
    return torch_tensor(data.astype(float32))