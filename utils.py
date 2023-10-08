# libraries
import shutil
from PIL import Image
import os
import glob
import pathlib
from torchvision import datasets

# internal packages
import constants

# global variables
dataset = None

def extract_all_images(source_path = constants.IMAGES_SOURCE_DIR, dest_path = constants.IMAGES_DEST_DIR, use_torchvision = True):
    pathlib.Path(dest_path).mkdir(parents=True, exist_ok=True)
    if use_torchvision:
        d = load_dataset()
        for i in range(len(d)):
            image = d[i][0]
            if len(image.mode) == 3:
                image.save(dest_path + "%d.jpg"%i)
    else:
        # used if extracing by downloading zip manually 
        for image_dir in glob.glob(source_path):
            category = os.path.basename(image_dir)
            for image_file in glob.glob(image_dir + "/*"):
                image = Image.open(image_file)
                image_name = os.path.basename(image_file)
                image_id = "%s_%s"%(category, image_name)
                dest = "%s%s"%(dest_path,image_id)
                if 'jpg' in dest and len(image.mode) == 3:
                    shutil.copyfile(image_file, dest)

def resize_all_images(size, source_dir = constants.IMAGES_DEST_DIR, mode="rgb"):
    dir_name = constants.RESIZED_IMAGES_DEST_DIR + "%d_%d_%s"%(size[0], size[1], mode)
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    for i in glob.glob(source_dir + "*"):
        image = Image.open(i)
        if mode != "rgb":
            image = convertRGBToGreyscale(image)
        resized_image = image.resize(size)
        image_filename = os.path.basename(i)
        resized_image.save(dir_name + "/" + image_filename)

# loads dataset if not already loaded
def load_dataset():
    global dataset
    if dataset is None:
        dataset = datasets.Caltech101(constants.IMAGES_SOURCE_DIR, download = True)
    return dataset

# converts RGB image to grey scale image
def convertRGBToGreyscale(image):
    return image.convert('L')

# shows image with the image ID
def show_image(image_id):
    Image.open(constants.IMAGES_DEST_DIR + "%s.jpg"%image_id).show()