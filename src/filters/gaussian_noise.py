def gaussian_noise(image, mean=0, sigma=1):
  numpy_image = numpy.array(image)
  noise = numpy.random.normal(mean, sigma, numpy_image.shape)
  noisy_image = numpy.clip(numpy_image + noise, 0, 255)
  return Image.fromarray(noisy_image.astype(numpy.uint8))