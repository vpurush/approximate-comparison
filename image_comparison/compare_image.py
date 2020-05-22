from lsh.lsh import LSH
import numpy as np
from PIL import Image
import os
from numpy import asarray

def file_path(relative_path):
    dir = os.path.dirname(os.path.abspath(__file__))
    split_path = relative_path.split("/")
    new_path = os.path.join(dir, *split_path)
    return new_path

def getImageData(image_path):
    return asarray(Image.open(open(file_path(image_path), 'rb')).resize((100, 100)))

imageNames = [
    "./images/mickey_mouse_yellow.png",
    "./images/mickey_mouse_blue.png",
    "./images/donald_duck.png",
    "./images/donald_duck_red.png"
]

images = []
for imgNames in imageNames:
    images.append(getImageData(imgNames))

alphaIgnoredImages = []
for img in images:
    imgWithoutAlpha = img[:,:,0:3]
    # print("imgWithoutAlpha", imgWithoutAlpha.shape)
    alphaIgnoredImages.append(imgWithoutAlpha)

# print("images", images, images[0].shape,  images[1].shape, images[2].shape)
# print("alphaIgnoredImages", alphaIgnoredImages)

reshapedImages = []
for img in alphaIgnoredImages:
    reshapedImg = img.reshape(1,-1)
    reshapedImages.append(reshapedImg)

print("reshapedImages", reshapedImages, "dimension", reshapedImages[0].shape[1])

lshModel = LSH(noOfHashers=25, noOfHash=10, dimension=reshapedImages[0].shape[1])

for i in range(0, len(reshapedImages)):
    lshModel.train(reshapedImages[i], { "name": imageNames[i] })

print(lshModel.isSimilar(reshapedImages[0], reshapedImages[1]))
print(lshModel.isSimilar(reshapedImages[0], reshapedImages[2]))
print(lshModel.isSimilar(reshapedImages[1], reshapedImages[2]))
print(lshModel.isSimilar(reshapedImages[2], reshapedImages[3]))