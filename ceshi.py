import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import bushing.bushing as bushi
import glob as gb
import cv2
from mrcnn.model import log

dataset_val = bushi.BalloonDataset()
dataset_val.load_balloon(r'C:\Users\Administrator\Desktop\bushingceshi', "val")
dataset_val.prepare()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
class InferenceConfig(bushi.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(os.path.realpath(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples\bushing"))  # To find local version
#% matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "G:\image recognition\logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "G:\image recognition\mask_rcnn_balloon_0080.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = r'C:\Users\Administrator\Desktop\bushingceshi\val'
class InferenceConfig(bushi.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'bushing']

# Load a random image from the images folder

#file_names = next(os.walk(r'C:\Users\Administrator\Desktop\bushing\train'))[2]
#image = skimage.io.imread(os.path.join(r'C:\Users\Administrator\Desktop\bushing\train', random.choice(file_names)))
img_path = gb.glob(r'C:\Users\Administrator\Desktop\bushingceshi\val\*.jpg')
#img_path.sort()
count = os.listdir(IMAGE_DIR)

for file in count:
    print (file)

for i in range(0, len(count)):
    path = os.path.join(IMAGE_DIR, count[i])
    if os.path.isfile(path):
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(count[i], image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        plt.show()
"""
for a_path in img_path:
    image = cv2.imread(a_path)
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'], ax=get_ax())
    plt.show()
"""
