from argparse import ArgumentParser
from cv2 import COLOR_HSV2RGB, COLOR_RGB2GRAY, COLOR_RGB2HSV, cvtColor, COLOR_RGB2LAB, COLOR_LAB2RGB, split, merge, createCLAHE
from scripts.kromo import add_chromatic, add_jitter, blend_images
from matplotlib import pyplot
from numpy import mod, ndarray, uint8, clip, array, ndindex, sqrt, zeros, sum, roll, repeat, newaxis
from numpy.random import normal, default_rng
from pathlib import Path
from perlin_numpy import generate_fractal_noise_2d
from PIL import Image, ImageOps, ImageEnhance
from pykuwahara import kuwahara
from scipy.ndimage import gaussian_filter, zoom, sobel
from sklearn.cluster import KMeans

def normalize_image(image: ndarray, rescale: bool = False) -> ndarray:
    """
    Normalize the intensity values of an image to be within the range of 
    [0, 255]. The 'rescale' parameter will move the relative scale such that the
    minimum intensity value of the image will be zero.
    """
    min_value = 0
    if rescale or image.min() < 0:
        min_value = image.min()
    return ((image - min_value) / (image.max() - min_value) * 255).astype(uint8)

def clip_image(image: ndarray, min_value: int = 0, max_value: int = 255) -> ndarray:
    """
    Clip the intensity values of an image to be within the range
    [min_value, max_value].
    """
    return clip(image, min_value, max_value).astype(uint8)

def next_power_of_two(n: int) -> int:
    """
    Return the next power of two greater than or equal to n.
    """
    return 1 << (n - 1).bit_length()

def bounded_strength(strength: float, min_value: float = 1, max_value: float = 10) -> bool:
    """
    Asserts the strength of a filter to be within the range [min_value,
    max_value].
    """
    return clip(strength, min_value, max_value)

def grayscale(image: ndarray) -> ndarray:
    """
    Convert an image to grayscale.
    """
    return cvtColor(image, COLOR_RGB2GRAY)

def gaussian_noise(image: ndarray, strength: float = 1) -> ndarray:
    """
    Add Gaussian noise (high-frequency noise) to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    noise = normal(0, 16 * strength, image.shape)
    return clip_image(signal + noise)

def perlin_noise(image: ndarray, strength: float = 1) -> ndarray:
    """
    Add Perlin noise (smooth, low-frequency noise) to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    noise_size = (next_power_of_two(image.shape[0]), next_power_of_two(image.shape[1]))
    seed_generator = default_rng()
    generator_r = default_rng(seed=seed_generator.integers(0, 1 << 8))
    noise_r = generate_fractal_noise_2d(noise_size, (8, 8), rng=generator_r)
    generator_g = default_rng(seed=seed_generator.integers(0, 1 << 8))
    noise_g = generate_fractal_noise_2d(noise_size, (8, 8), rng=generator_g)
    generator_b = default_rng(seed=seed_generator.integers(0, 1 << 8))
    noise_b = generate_fractal_noise_2d(noise_size, (8, 8), rng=generator_b)
    noise = array([noise_r, noise_g, noise_b]).transpose(1, 2, 0)
    noise = normalize_image(noise) * (strength / 20)
    return clip_image(signal + noise[:image.shape[0], :image.shape[1], :])

def gaussian_blur(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a Gaussian blur (low-pass filter) to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    r = int(strength) * 2
    return clip_image(gaussian_filter(signal, strength, radius=(r, r, 0)))

def gaussian_high_pass(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a Gaussian high-pass filter to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    r = int(10) * 2
    low_pass = gaussian_filter(signal, 10, radius=(r, r, 0))
    return normalize_image(signal - low_pass * (strength / 10))

def horizontal_blur(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a wide, horizontal anisotropic gaussian blur kernel to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    r = int(strength) * 2
    return clip_image(gaussian_filter(signal, strength, radius=(1, r, 0)))

def vertical_blur(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a wide, vertical anisotropic gaussian blur kernel to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    r = int(strength) * 2
    return clip_image(gaussian_filter(signal, strength, radius=(r, 1, 0)))

def focus_blur(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a "focus" blur which blurs the edges of an image radially, and
    leaves the center relatively unchanged.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    r = int(strength) * 4
    blurred = gaussian_filter(signal, strength * 2, radius=(r, r, 0))
    center = (image.shape[0] // 2, image.shape[1] // 2)
    blur_ratios = zeros(image.shape)
    max_distance = sqrt(center[0] ** 2 + center[1] ** 2)
    for index in ndindex(image.shape):
        distance = sqrt((index[0] - center[0]) ** 2 + (index[1] - center[1]) ** 2)
        blur_ratios[index] = distance / max_distance
    return clip_image(signal * (1 - blur_ratios) + blurred * blur_ratios)

def bloom_filter(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a bloom filter, which bleeds bright colors into the surrounding
    area of the image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    bright = (signal > 192).astype(uint8)
    r = int(strength) * 2
    bloom = gaussian_filter(signal * bright, strength, radius=(r, r, 0))
    return clip_image(bloom * (strength / 10) + signal)

def invert(image: ndarray, strength: float = 1) -> ndarray:
    """
    Invert the colors of an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    return clip_image((255 - signal) * (strength / 10) + (signal * (1 - strength / 10)))

def pixelate(image: ndarray, strength: float = 1) -> ndarray:
    """
    Pixelate an image.
    """
    bounded_strength(strength)
    packed_image = Image.fromarray(image)
    downsize = (int(image.shape[1] / (strength * 2)), int(image.shape[0] / (strength * 2)))
    upsize = (image.shape[1], image.shape[0])
    packed_image = packed_image.resize(downsize)
    return clip_image(array(packed_image.resize(upsize, Image.NEAREST)))

def row_shift(image: ndarray, strength: float = 1) -> ndarray:
    """
    Shift each row of an image horizontally by a random amount.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    shifted = zeros(image.shape)
    for i in range(image.shape[0]):
        shift_amount = int(normal(0, strength))
        shifted[i, :, 0] = roll(signal[i, :, 0], shift_amount)
        shifted[i, :, 1] = roll(signal[i, :, 1], shift_amount)
        shifted[i, :, 2] = roll(signal[i, :, 2], shift_amount)
    return clip_image(shifted)

def kuwahara_filter(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply the Kuwahara filter to an image, which blurs continuous regions of the
    image but preserves edges.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    return clip_image(kuwahara(signal, method='gaussian', radius=int(strength)))

class BarrelDeformer:
    """
    Applies a barrel distortion to an image using ImageOps from the PIL library.
    Source: https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
    """
    def __init__(self, k_1: float, k_2: float, w: int, h: int):
        self.k_1 = k_1
        self.k_2 = k_2
        self.w = w
        self.h = h

    def transform(self, x, y):
        # center and scale the grid for radius calculation (distance from center of image)
        x_c, y_c = self.w / 2, self.h / 2 
        x = (x - x_c) / x_c
        y = (y - y_c) / y_c
        radius = sqrt(x**2 + y**2) # distance from the center of image
        m_r = 1 + self.k_1*radius + self.k_2*radius**2 # radial distortion model
        # apply the model 
        x, y = x * m_r, y * m_r
        # reset all the shifting
        x, y = x*x_c + x_c, y*y_c + y_c
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, _):
        gridspace = 20
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]
    
def barrel_distortion(image: ndarray, strength: float = 1) -> ndarray:
    """
    Using the BarrelDeformer class, apply a barrel distortion to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    packed_image = Image.fromarray(signal)
    w, h = packed_image.size
    k_1 = 0.1 * strength
    k_2 = 0.025 * strength
    deformed_image = ImageOps.deform(packed_image, BarrelDeformer(k_1, k_2, w, h))
    zoom_range = (0.1, 0.9)
    current_zoom = zoom_range[0] + (zoom_range[1] - zoom_range[0]) * (strength / 10)
    padding = (int(image.shape[0] * current_zoom / 2), int(image.shape[1] * current_zoom / 2))
    zoom_shape = (1 + current_zoom, 1 + current_zoom, 1)
    zoomed_image = zoom(array(deformed_image), zoom_shape)
    cropped_image = zoomed_image[padding[0]:-padding[0], padding[1]:-padding[1], :]
    return clip_image(cropped_image)

def contrast(image: ndarray, strength: float = 1) -> ndarray:
    """
    Boost the contrast of an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    lab = cvtColor(signal, COLOR_RGB2LAB)
    l, a, b = split(lab)
    clahe = createCLAHE(clipLimit=strength, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return clip_image(cvtColor(merge((cl, a, b)), COLOR_LAB2RGB))

def decontrast(image: ndarray, strength: float = 1) -> ndarray:
    """
    Reduce the contrast of an image.
    """
    bounded_strength(strength)
    packed_image = Image.fromarray(normalize_image(image))
    filter = ImageEnhance.Contrast(packed_image)
    return clip_image(array(filter.enhance(1 / strength)))

def saturate(image: ndarray, strength: float = 1) -> ndarray:
    """
    Saturate the colors of an image.
    """
    bounded_strength(strength)
    packed_image = Image.fromarray(normalize_image(image))
    filter = ImageEnhance.Color(packed_image)
    return clip_image(array(filter.enhance(1 + (strength / 5))))

def desaturate(image: ndarray, strength: float = 1) -> ndarray:
    """
    Desaturate the colors of an image.
    """
    bounded_strength(strength)
    packed_image = Image.fromarray(normalize_image(image))
    filter = ImageEnhance.Color(packed_image)
    return clip_image(array(filter.enhance((10 - strength) / 10)))

def color_quantization(image: ndarray, strength: float = 1) -> ndarray:
    """
    Quantize the colors of an image.
    """
    bounded_strength(strength)
    data = normalize_image(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=(int(100 / strength)), random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    quantized = centers[labels].reshape(image.shape).astype(uint8)
    return clip_image(quantized)

def darken_shadows(image: ndarray, strength: float = 1) -> ndarray:
    """
    Darken the shadows of an image by thresholding all low luminance values and
    multiplying their luminance values by a fractional amount.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    thresholded = (signal[:, :, 0] + signal[:, :, 1] + signal[:, :, 2] < 10 * strength).astype(uint8)
    thresholded = repeat(thresholded[:, :, newaxis], 3, axis=2)
    signal -= (signal * thresholded * (strength / 10)).astype(uint8)
    return signal

def vignette(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a vignette to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    darken_ratios = zeros(image.shape)
    max_distance = sqrt(center[0] ** 2 + center[1] ** 2)
    for index in ndindex(image.shape):
        distance = sqrt((index[0] - center[0]) ** 2 + (index[1] - center[1]) ** 2)
        darken_ratios[index] = (distance / max_distance) ** 4
    return clip_image(signal * (1 - darken_ratios) + signal * ((10 - strength) / 10) * darken_ratios)

def hue_shift(image: ndarray, strength: float = 1) -> ndarray:
    """
    Shift the hue of an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    hsv = cvtColor(signal, COLOR_RGB2HSV)
    h, s, v = split(hsv)
    ch = mod(h + strength * 10, 180).astype(uint8)
    return cvtColor(merge([ch, s, v]), COLOR_HSV2RGB)

def screen_door_effect(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply a screen door effect to an image.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    for i in range(image.shape[0]):
        if i % 2 == 0:
            signal[i, :, 0] = signal[i, :, 0] * ((10 - strength) / 10)
            signal[i, :, 1] = signal[i, :, 1] * ((10 - strength) / 10)
            signal[i, :, 2] = signal[i, :, 2] * ((10 - strength) / 10)
    return clip_image(signal)

def thin_edge_enhancement(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply Difference-of-Gaussians edge detection for high-frequency edges and
    darken the edges.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    gray = grayscale(signal)
    blurred_1 = gaussian_filter(gray, 1, radius=(1, 1))
    blurred_2 = gaussian_filter(gray, 2, radius=(2, 2))
    difference = repeat(normalize_image(blurred_1 - blurred_2)[:, :, newaxis], 3, axis=2)
    mask = clip(1 + ((10 - strength) / 10) - normalize_image(difference) / 255, 0, 1)
    return normalize_image(mask * signal)

def thick_edge_enhancement(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply Difference-of-Gaussians edge detection for low-frequency edges and
    darken the edges.
    """
    bounded_strength(strength)
    signal = normalize_image(image)
    gray = grayscale(signal)
    blurred_1 = gaussian_filter(gray, 8, radius=(8, 8))
    blurred_2 = gaussian_filter(gray, 9, radius=(9, 9))
    difference = repeat(normalize_image(blurred_1 - blurred_2)[:, :, newaxis], 3, axis=2)
    mask = clip(1 + ((10 - strength) / 10) - normalize_image(difference) / 255, 0, 1)
    return normalize_image(mask * signal)

def chromatic_aberration(image: ndarray, strength: float = 1) -> ndarray:
    """
    Apply chromatic aberration to an image, using the `kromo` package.
    """
    bounded_strength(strength)
    im = Image.fromarray(normalize_image(image))
    # Ensure width and height are odd numbers
    new_width = im.size[0]
    new_height = im.size[1]
    if im.size[0] % 2 == 0:
        new_width = im.size[0] - 1
    if im.size[1] % 2 == 0:
        new_height = im.size[1] - 1
    im.resize((new_width, new_height))
    og_im = im.copy()
    im = add_chromatic(im, strength=strength, no_blur=True)
    im = add_jitter(im, pixels=0)
    im = blend_images(im, og_im, alpha=0, strength=strength)
    return array(im)

all_filters = [
    gaussian_noise,
    perlin_noise,
    gaussian_blur,
    gaussian_high_pass,
    horizontal_blur,
    vertical_blur,
    focus_blur,
    bloom_filter,
    invert,
    pixelate,
    row_shift,
    kuwahara_filter,
    barrel_distortion,
    contrast,
    decontrast,
    saturate,
    desaturate,
    color_quantization,
    darken_shadows,
    vignette,
    hue_shift,
    screen_door_effect,
    thin_edge_enhancement,
    thick_edge_enhancement,
    chromatic_aberration,
]

filter_names = [filter.__name__ for filter in all_filters]

if __name__ == '__main__':
    parser = ArgumentParser(description='A collection of post-processing image \
    filters. This script will display a visual example of the specified \
    filter.')
    parser.add_argument('path', type=str, help='The path of the image for \
    which to apply the filter.')
    parser.add_argument('filter', type=str, help='The name of the filter to \
    display. Filters include: ' + ', '.join(filter_names))
    parser.add_argument('strength', type=float, default=1, help='The strength \
    of the filter being applied.')
    args = parser.parse_args()
    found_filter = False
    for filter in all_filters:
        if filter.__name__ == args.filter:
            found_filter = True
            image = Image.open(Path(args.path).resolve())
            image_data = array(image)
            filtered_data = filter(image_data, args.strength)
            figure_1 = pyplot.subplot(1, 2, 1)
            figure_1.imshow(image_data)
            figure_1.set_axis_off()
            figure_1.set_title('Original Image')
            figure_2 = pyplot.subplot(1, 2, 2)
            figure_2.imshow(filtered_data)
            figure_2.set_axis_off()
            figure_2.set_title(f'Filter `{filter.__name__}` at strength {args.strength}')
            pyplot.show()
    if not found_filter:
        raise ValueError(f'Filter `{args.filter}` not found.')
