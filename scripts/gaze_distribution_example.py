from matplotlib import pyplot
from numpy import exp, abs
from pathlib import Path
from PIL import Image
from utilities import to_density, is_pdf, kl_div

_, axes = pyplot.subplots(1, 2)
first, second = axes
gaze_distribution = Image.open('performance_data/Reference/saliency/1.png')
density = to_density(gaze_distribution)
log_density = exp(to_density(gaze_distribution, log=True))

gaze_distribution_2 = Image.open('performance_data/Compression_1/saliency/2.png')
density_2 = to_density(gaze_distribution_2)
log_density_2 = exp(to_density(gaze_distribution_2, log=True))

div_1 = kl_div(density, log_density_2)
div_2 = kl_div(density_2, log_density)
div_3 = kl_div(density, density_2)
div_4 = kl_div(log_density, log_density_2)
print(div_1, div_2, div_3, div_4)

density_crop = density[325:550, 700:1000]
log_density_crop = log_density[325:550, 700:1000]

first.imshow(density_crop, vmin=0, vmax=density_crop.max())
first.axis('off')
first.set_title('Normalized Density')
second.imshow(log_density_crop, vmin=0, vmax=density_crop.max())
second.axis('off')
second.set_title('LogSumExp Density')

pyplot.show()