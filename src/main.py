import numpy
from PIL import Image

def saliency_to_log_density(saliency_map: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a saliency map to a log density distribution.
    """
    # Normalize saliency map to be within the range [1, 256]--we avoid 0 to avoid log(0)
    normalized_saliency =  (255 * (saliency_map / saliency_map.max())).astype(int) + 1

    # Convert to PDF, then to log density
    log_density = numpy.log(normalized_saliency / normalized_saliency.sum())

    # Probability distribution should sum to 1
    assert numpy.exp(log_density).sum() > 0.9999

    return log_density

def kl_divergence(p: numpy.ndarray, q: numpy.ndarray) -> float:
    """
    Compute the KL divergence between two probability distributions.
    """
    return numpy.sum(p * numpy.log(p / q))

def auc_score(p: numpy.ndarray, q: numpy.ndarray) -> float:
    """
    Compute the AUC score between two probability distributions.
    """
    return numpy.trapezoid(p, q)

def information_gain(p: numpy.ndarray, q: numpy.ndarray) -> float:
    """
    Compute the information gain between two probability distributions.
    """
    return 

# First, we apply image filters to all images in the dataset

# Then, we apply to model to base and filtered images to get saliency maps

# Then, we convert the saliency maps to log densities, and we compare to the base log density

