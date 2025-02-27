import numpy
import os
from PIL import Image



def default_filter_set():
  return {
    "gaussian_noise": gaussian_noise,
  }

def apply_filters(search_directory, verbose=False, filter_set=default_filter_set()):
  for root, _, files in os.walk(search_directory):
    path = root.split(os.sep)
    if 'base' in path and 'images' in path:
      if verbose:
        print(f"Post-processing filters for dataset {os.sep.join(path[1:-2])}")

      for filter_name, filter_function in filter_set.items():
        filter_path = os.path.join(os.sep.join(path[:-2]), filter_name, 'images')

        if not os.path.exists(filter_path):
          os.makedirs(filter_path)

        for file in files:
          if file.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(os.path.join(root, file))
            filter_function(image).save(os.path.join(filter_path, file))

if __name__ == "__main__":
  apply_filters("../data", True)
