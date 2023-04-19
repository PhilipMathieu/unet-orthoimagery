# Final Project
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
from torchvision.transforms import functional
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Subset
import os

from utils.data_loading import MEOIDataset

torch.manual_seed(17) # for repeatability, see https://pytorch.org/vision/main/transforms.html

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    if os.path.exists("AD")==False:
        os.mkdir("AD")
    if os.path.exists("AD/images")==False:
        os.mkdir("AD/images")
    if os.path.exists("AD/labels")==False:
        os.mkdir("AD/labels")

    data_dir = 'data/Image_Chips_128_overlap_balanced_dem'
    dir_img = os.path.join(data_dir, "images/")
    dir_dem = os.path.join(data_dir, "images2/")
    dir_mask = os.path.join(data_dir, "labels/")

    fig = plt.figure()
    data_transform_rotation = transforms.Compose([
                                     transforms.ToTensor()])
    data_transform_affine = transforms.Compose([transforms.RandomAffine(20,(0,0),(1,1.5),0.1),
                                                transforms.CenterCrop((64,64)),
                                                transforms.ToTensor()])
    data_transform_randomCrop = transforms.Compose([transforms.RandomAffine(20,scale=(1,1.5)),
                                                transforms.RandomCrop((64,64)),
                                                transforms.ToTensor()])
    test_data_affine = MEOIDataset(dir_img, dir_dem, dir_mask, transform=data_transform_affine)
    test_data_randomCrop = MEOIDataset(dir_img, dir_dem, dir_mask, transform=data_transform_randomCrop)

    # Affine Transform
    idx = [i for i in range(len(test_data_affine)) if test_data_affine.imgs[i][1] == test_data_affine.class_to_idx['images']]
    labelIdx = [i for i in range(len(test_data_affine)) if test_data_affine.imgs[i][1] == test_data_affine.class_to_idx['labels']]
    # build the appropriate subset
    subset = Subset(test_data_affine, idx)
    labelSubset = Subset(test_data_affine, labelIdx)
    myTest_loader = torch.utils.data.DataLoader(subset,batch_size=len(subset), shuffle=False)
    label_loader = torch.utils.data.DataLoader(labelSubset,batch_size=len(labelSubset), shuffle=False)
    myExamples = enumerate(myTest_loader)
    batch_idx, (myExample_data, myExample_targets) = next(myExamples)
    myLabels = enumerate(label_loader)
    batch_idx, (myLabels_data, myExample_targets) = next(myLabels)

    # Random Crop Transform
    idx = [i for i in range(len(test_data_randomCrop)) if test_data_randomCrop.imgs[i][1] == test_data_randomCrop.class_to_idx['images']]
    labelIdx = [i for i in range(len(test_data_randomCrop)) if test_data_randomCrop.imgs[i][1] == test_data_randomCrop.class_to_idx['labels']]
    # build the appropriate subset
    randomCropSubset = Subset(test_data_randomCrop, idx)
    randomCropLabelSubset = Subset(test_data_randomCrop, labelIdx)
    randomCropMyTest_loader = torch.utils.data.DataLoader(randomCropSubset,batch_size=len(subset), shuffle=False)
    randomCropLabel_loader = torch.utils.data.DataLoader(randomCropLabelSubset,batch_size=len(labelSubset), shuffle=False)
    randomCropMyExamples = enumerate(randomCropMyTest_loader)
    batch_idx, (randomCropMyExample_data, myExample_targets) = next(randomCropMyExamples)
    randomCropLabels = enumerate(randomCropLabel_loader)
    batch_idx, (randomCropLabels_data, myExample_targets) = next(randomCropLabels)
    transform = transforms.ToPILImage('RGBA')
    print(str(len(myTest_loader.dataset)))

    # iterate over all items in the loader
    for i in range(len(myTest_loader.dataset)):
        # Affine
        img = transform(myExample_data[i]) # apply ToPILImage('RGBA')
        rgba = img.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 0:  # finding yellow colour
                # replacing it with a transparent value
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        rgba.putdata(newData)
        rgba.save("AD/images/file"+str(i*2)+".tif", "TIFF")
        save_image(myLabels_data[i],"AD/labels/file"+str(i*2)+".tif","TIFF")

        # Random Crop
        img = transform(randomCropMyExample_data[i])
        rgba = img.convert("RGBA")
        datas = rgba.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 0:  # finding yellow colour
                # replacing it with a transparent value
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        rgba.putdata(newData)
        rgba.save("AD/images/file"+str(i*2+1)+".tif", "TIFF")
        save_image(randomCropLabels_data[i],"AD/labels/file"+str(i*2+1)+".tif","TIFF")
    print(myExample_data[0].shape)
    
    # if __name__ == "__dataAugmentation__":
    #     print("Yes")
    #     main(sys.argv)
    return
if __name__ == "__main__":
    main(sys.argv)

