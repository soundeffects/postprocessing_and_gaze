from argparse import ArgumentParser
from os import listdir, remove
from os.path import join, splitext, isfile
from PIL import Image

# Description of command line arguments
parser = ArgumentParser(description='Convert the format of images in a directory to PNG format.')
parser.add_argument('directory', type=str,
    help='The directory containing the images to convert.')
args = parser.parse_args()

def to_png(image_path: str, output_path: str):
    """
    Convert the format of an image to PNG format and delete the original file.
    """
    image = Image.open(image_path)
    image.save(output_path, 'PNG')
    image.close()
    remove(image_path)

def to_png_directory(directory: str):
    """
    Convert the format of all images in a directory to PNG format and delete the
    original files.
    """
    # Common image extensions to process
    valid_extensions = {'.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.png', '.bmp'}
    
    for filename in listdir(directory):
        file_path = join(directory, filename)
        if isfile(file_path):
            prefix, extension = splitext(filename)
            simplified_prefix = prefix.split('_')[0]
            if extension.lower() in valid_extensions:
                to_png(file_path, join(directory, simplified_prefix + '.png'))

to_png_directory(args.directory)
