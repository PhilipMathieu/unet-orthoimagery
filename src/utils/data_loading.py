# https://github.com/milesial/Pytorch-UNet/blob/master/utils/data_loading.py
# modified by: James Kim
# date: Apr 14, 2023

import logging
import os
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.tif':
        return Image.open(filename)
    elif ext == '.tfw':
        return
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
class BasicDataset(Dataset):
    def __init__(self, data_dir: str, scale: float = 1.0, mask_suffix: str= ''):
        self.images_dir = Path(os.path.join(data_dir, "images/"))
        self.mask_dir = Path(os.path.join(data_dir, "labels/"))  
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0]  for file in listdir(self.images_dir) if isfile(join(self.images_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        self.mids = [splitext(file)[0]  for file in listdir(self.mask_dir) if isfile(join(self.mask_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        
        if len(self.mids) == len(self.ids):
            self.zero_masks = False
        else:
            logging.info("Dataset contains some all-zero masks")
            self.zero_masks = True
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        self.mask_values = [0,255]
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            mask[img == 255] = 1
            return mask
        else:
            img = img.transpose((2,0,1))
            if (img > 1).any():img = img / 255.0
            return img
    
    def __getitem__(self, idx):
        img_name = self.ids[idx]
        
        img_file = list(self.images_dir.glob(img_name + '.tif'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {img_name}: {img_file}'
        img = load_image(img_file[0])

        mask_name = self.mids[idx]
        mask_file = list(self.mask_dir.glob(mask_name + self.mask_suffix + '.tif'))
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {mask_name}: {mask_file}'
        mask = load_image(mask_file[0])

        assert img.size == mask.size, \
            f'Image {img_name} and mask {mask_name} should be the same size, but are {img.size} and {mask.size}'

        
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        
        return {
            'image': torch.as_tensor(img.copy(), dtype=torch.float32).contiguous(),
            'mask': torch.as_tensor(mask.copy(), dtype=torch.long).contiguous()
        }

class MEOIDataset(BasicDataset):
    def __init__(self, data_dir, scale=1):
        super().__init__(data_dir, scale, mask_suffix='')