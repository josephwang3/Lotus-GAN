# Joseph Wang
# 12/26/2021
# Pre- and post-processor scripts for images

import os
from PIL import Image

# directory where input images are saved and where output images are to be saved
IMAGE_DIR = 'testIn/'
SAVE_DIR = 'testOut/'

# bounds for cropping and resizing
CROPPED_SIZE = 500
RESIZED_SIZE = 64

# crop images to specified size
def cropImages(imageDir: str, croppedSize: int, saveDir: str) -> None:
    for filename in os.listdir(imageDir):
        imagePath = os.path.join(imageDir, filename)
        image = Image.open(imagePath)

        # crop image to save center 500x500 section
        w, h = image.size
        wMargin = (w - croppedSize) / 2
        hMargin = (h - croppedSize) / 2
        cropped = image.crop((wMargin, hMargin, croppedSize + wMargin, croppedSize + hMargin))

        # save image
        savePath = os.path.join(saveDir, filename)
        cropped.save(savePath)

# resize images to specified size
def resizeImages(imageDir: str, resizedSize: int, saveDir: str) -> None:
    for filename in os.listdir(imageDir):
        # open image
        imagePath = os.path.join(imageDir, filename)
        image = Image.open(imagePath)

        # resize image
        resized = image.resize((resizedSize, resizedSize), Image.ANTIALIAS)

        # save image
        savePath = os.path.join(saveDir, filename)
        resized.save(savePath)
    
# rename images to "image0.png" convention
def renameImages(imageDir: str, saveDir: str) -> None:
    for i, filename in enumerate(os.listdir(imageDir)):
        path = os.path.join(imageDir, filename)
        image = Image.open(path)

        new_name = "image" + str(i) + ".png"

        # save image with new name
        save_path = os.path.join(saveDir, new_name)
        image.save(save_path)

# cropImages(IMAGE_DIR, CROPPED_SIZE, SAVE_DIR)
# resizeImages(IMAGE_DIR, RESIZED_SIZE, SAVE_DIR)
renameImages(IMAGE_DIR, SAVE_DIR)