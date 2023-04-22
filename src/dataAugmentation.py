# Hao Sheng (Jack) Ning Final Project
# Used for Data Augmentation
# CS 5330 Computer Vision
# Spring 2023
# import statements
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import functional, ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Subset
import os
from torchvision.io import read_image
from random import random

magnitude = 4

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')
    
def horizontalFlipTransformFunction(image, dem, mask):
    # Resize
    resize = transforms.Resize(size=(64, 64))
    image = resize(image)
    dem = dem.resize((64,64))
    mask = mask.resize((64,64))
    

    # Random horizontal flipping
    image = transforms.functional.hflip(image)
    dem = transforms.functional.hflip(dem)
    mask = transforms.functional.hflip(mask)

    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    dem = transforms.functional.to_tensor(dem)
    mask = transforms.functional.to_tensor(mask)
    return image, dem, mask

def verticalFlipTransformFunction(image, dem, mask):
    # Resize
    resize = transforms.Resize(size=(64, 64))
    image = resize(image)
    dem = dem.resize((64,64))
    mask = mask.resize((64,64))

    # Random vertical flipping
    image = transforms.functional.vflip(image)
    dem = transforms.functional.vflip(dem)
    mask = transforms.functional.vflip(mask)

    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    dem = transforms.functional.to_tensor(dem)
    mask = transforms.functional.to_tensor(mask)
    return image, dem, mask

def randomCropTransformFunction(image, dem, mask):

    #Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(64, 64))
    image = transforms.functional.crop(image, i, j, h, w)
    dem = transforms.functional.crop(dem, i, j, h, w)
    mask = transforms.functional.crop(mask, i, j, h, w)

    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    dem = transforms.functional.to_tensor(dem)
    mask = transforms.functional.to_tensor(mask)
    return image, dem, mask
    
def transformFunction(image, dem, mask):
    # Resize
    resize = transforms.Resize(size=(128, 128))
    image = resize(image)
    dem = resize(dem)
    mask = resize(mask)

    # Shear
    image = transforms.functional.affine(image,angle=0,translate=(0,0),shear=10,scale=1)
    dem = transforms.functional.affine(dem,angle=0,translate=(0,0),shear=10,scale=1)
    mask = transforms.functional.affine(mask,angle=0,translate=(0,0),shear=10,scale=1)

    # Rnadom Scale
    randomScale = random()+1
    image = transforms.functional.affine(image,angle=0,translate=(0,0),shear=0,scale=randomScale)
    dem = transforms.functional.affine(dem,angle=0,translate=(0,0),shear=0,scale=randomScale)
    mask = transforms.functional.affine(mask,angle=0,translate=(0,0),shear=0,scale=randomScale)

    #Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(64, 64))
    image = transforms.functional.crop(image, i, j, h, w)
    dem = transforms.functional.crop(dem, i, j, h, w)
    mask = transforms.functional.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random() > 0.5:
        image = transforms.functional.hflip(image)
        dem = transforms.functional.hflip(dem)
        mask = transforms.functional.hflip(mask)

    # Random vertical flipping
    if random() > 0.5:
        image = transforms.functional.vflip(image)
        dem = transforms.functional.vflip(dem)
        mask = transforms.functional.vflip(mask)

    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    dem = transforms.functional.to_tensor(dem)
    mask = transforms.functional.to_tensor(mask)
    return image, dem, mask

class MyDataset(Dataset):
    def __init__(self, image_paths, dem_paths,target_paths, transform, train=True):
        self.image_paths = image_paths
        self.dem_paths = dem_paths
        self.target_paths = target_paths
        self.transform = transform
    

    def __getitem__(self, index):
        filenames = [name for name in os.listdir(self.image_paths) if os.path.splitext(name)[-1] == '.tif']
        image = Image.open(os.path.join(self.image_paths,filenames[index]))
        dem=Image.open(os.path.join(self.dem_paths,filenames[index]))
        mask = Image.open(os.path.join(self.target_paths,filenames[index]))
        x, y, z = self.transform(image, dem, mask)
        return x, y, z

    def __len__(self):
        return len(self.image_paths)

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    print("hello world")
    if os.path.exists("AD")==False:
        os.mkdir("AD")
    if os.path.exists("AD/images")==False:
        os.mkdir("AD/images")
    if os.path.exists("AD/images2")==False:
        os.mkdir("AD/images2")
    if os.path.exists("AD/labels")==False:
        os.mkdir("AD/labels")
    fig = plt.figure()
    data_transform_rotation = transforms.Compose([
                                     transforms.ToTensor()])
    data_transform_affine = transforms.Compose([transforms.RandomAffine(20,(0,1),(1,1.5),0.1),
                                                transforms.CenterCrop((64,64)),
                                                transforms.ToTensor()])
    data_transform_randomCrop = transforms.Compose([transforms.RandomAffine(20,scale=(1,1.5)),
                                                transforms.RandomCrop((64,64)),
                                                transforms.ToTensor()])

    transform_loader = MyDataset('data/Image_Chips_128_overlap_balanced_dem/images','data/Image_Chips_128_overlap_balanced_dem/images2','data/Image_Chips_128_overlap_balanced_dem/labels',transformFunction)
    horizontal_transform_loader = MyDataset('data/Image_Chips_128_overlap_balanced_dem/images','data/Image_Chips_128_overlap_balanced_dem/images2','data/Image_Chips_128_overlap_balanced_dem/labels',horizontalFlipTransformFunction)
    vertical_transform_loader = MyDataset('data/Image_Chips_128_overlap_balanced_dem/images','data/Image_Chips_128_overlap_balanced_dem/images2','data/Image_Chips_128_overlap_balanced_dem/labels',verticalFlipTransformFunction)
    random_crop_transform_loader = MyDataset('data/Image_Chips_128_overlap_balanced_dem/images','data/Image_Chips_128_overlap_balanced_dem/images2','data/Image_Chips_128_overlap_balanced_dem/labels',randomCropTransformFunction)

    transform = transforms.ToPILImage('RGBA')
    regTransform=transforms.ToPILImage()
    #print(str(len(myTest_loader.dataset)))
    for idx, img in enumerate(transform_loader):
        # images
        image = transform(img[0])
        rgba = image.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        rgba.putdata(newData)
        rgba.save("AD/images/file"+str(idx*magnitude+0)+".tif", "TIFF")
        # dem
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save("AD/images2/file"+str(idx*magnitude+0)+".tif", "TIFF")
        # labels
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save("AD/labels/file"+str(idx*magnitude+0)+".tif","TIFF")

    for idx, img in enumerate(horizontal_transform_loader):
        # Horizontal
        # images
        image = transform(img[0])
        rgba = image.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        rgba.putdata(newData)
        rgba.save("AD/images/file"+str(idx*magnitude+1)+".tif", "TIFF")
        # dem
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save("AD/images2/file"+str(idx*magnitude+1)+".tif", "TIFF")
        # label
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save("AD/labels/file"+str(idx*magnitude+1)+".tif","TIFF")

    for idx, img in enumerate(vertical_transform_loader):
        # Vertical
        # images
        image = transform(img[0])
        rgba = image.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        rgba.putdata(newData)
        rgba.save("AD/images/file"+str(idx*magnitude+2)+".tif", "TIFF")
        # dem
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save("AD/images2/file"+str(idx*magnitude+2)+".tif", "TIFF")
        # labels
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save("AD/labels/file"+str(idx*magnitude+2)+".tif","TIFF")
    
    for idx, img in enumerate(random_crop_transform_loader):
        # Random Crop
        # images
        image = transform(img[0])
        rgba = image.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        rgba.putdata(newData)
        rgba.save("AD/images/file"+str(idx*magnitude+3)+".tif", "TIFF")
        # DEM
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save("AD/images2/file"+str(idx*magnitude+3)+".tif", "TIFF")
        # labels
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save("AD/labels/file"+str(idx*magnitude+3)+".tif","TIFF")

    return
if __name__ == "__main__":
    main(sys.argv)
