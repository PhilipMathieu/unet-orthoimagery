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
import argparse

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


def main(args):
    images_dir = os.path.join(args.data_dir, "images/")
    dem_dir = os.path.join(args.data_dir, "images2/")
    mask_dir = os.path.join(args.data_dir, "labels/")
    images_dir_out = os.path.join(args.output_dir, "images/")
    dem_dir_out = os.path.join(args.output_dir, "images2/")
    mask_dir_out = os.path.join(args.output_dir, "labels/")
    
    transform_loader = MyDataset(images_dir, dem_dir, mask_dir, transformFunction)
    horizontal_transform_loader = MyDataset(images_dir, dem_dir, mask_dir, horizontalFlipTransformFunction)
    vertical_transform_loader = MyDataset(images_dir, dem_dir, mask_dir, verticalFlipTransformFunction)
    random_crop_transform_loader = MyDataset(images_dir, dem_dir, mask_dir, randomCropTransformFunction)

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
        rgba.save(images_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")
        # dem
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save(dem_dir_out+"file"+str(idx*magnitude+0)+".tif", "TIFF")
        # labels
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save(mask_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")

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
        rgba.save(images_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")
        # dem
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save(dem_dir_out+"file"+str(idx*magnitude+0)+".tif", "TIFF")
        # label
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save(mask_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")

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
        rgba.save(images_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")
        # dem
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save(dem_dir_out+"file"+str(idx*magnitude+0)+".tif", "TIFF")
        # labels
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save(mask_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")
    
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
        rgba.save(images_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")
        # DEM
        dem = regTransform(img[1])
        datas = dem.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        dem.putdata(newData)
        dem.save(dem_dir_out+"file"+str(idx*magnitude+0)+".tif", "TIFF")
        # labels
        label = regTransform(img[2]) 
        datas = label.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        label.putdata(newData)
        label.save(mask_dir_out + "file"+str(idx*magnitude+0)+".tif", "TIFF")

    return

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data-dir', dest='data_dir', type=os.Path, default="data/Image_Chips_128_overlap_unbalanced_dem/", help="Directory containing dataset")
    parser.add_argument('--output-dir', dest='output_dir', type=os.Path, default="AD/", help="Directory to place augmented data in")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if os.path.exists(args.output_dir)==False:
        os.mkdir(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, "images/"))==False:
        os.mkdir(os.path.join(args.output_dir, "images/"))
    if os.path.exists(os.path.join(args.output_dir, "images2/"))==False:
        os.mkdir(os.path.join(args.output_dir, "images2/"))
    if os.path.exists(os.path.join(args.output_dir, "labels/"))==False:
        os.mkdir(os.path.join(args.output_dir, "labels/"))
    main(args)
