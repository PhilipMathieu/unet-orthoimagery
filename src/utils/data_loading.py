# https://github.com/milesial/Pytorch-UNet/blob/master/utils/data_loading.py
# modified by: James Kim
# date: Apr 14, 2023

import logging
import os
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import json

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
    
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = os.path.join(mask_dir, idx+".tif")
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    
class BasicDataset(Dataset):
    def __init__(self, data_dir: str, scale: float = 1.0, mask_suffix: str= ''):
        self.images_dir = Path(os.path.join(data_dir, "images/"))
        self.dem_dir = Path(os.path.join(data_dir, "images2/"))
        self.mask_dir = Path(os.path.join(data_dir, "labels/"))
        self.pos_weight = 1.0

        
        try:
            with open(os.path.join(data_dir, 'esri_accumulated_stats.json'), 'r') as file:
                self.stats = json.load(file)
            logging.info("Retrieved dataset stats from JSON")
            self._process_stats()
        except:
            self.stats = None
            logging.debug("Could not read dataset stats from JSON")
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0]  for file in listdir(self.images_dir) if isfile(join(self.images_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        self.dids = [splitext(file)[0]  for file in listdir(self.dem_dir) if isfile(join(self.dem_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        self.mids = [splitext(file)[0]  for file in listdir(self.mask_dir) if isfile(join(self.mask_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        
        assert len(self.ids) == len(self.dids)

        if len(self.mids) == len(self.ids):
            self.zero_masks = False
        else:
            logging.info("Dataset contains some all-zero masks")
            self.zero_masks = True
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir = self.mask_dir, mask_suffix=self.mask_suffix), self.mids),
                total = len(self.mids)
            ))
        
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)
    
    def _process_stats(self):
        self.pos_weight = (self.stats["FeatureStats"]["NumImagesTotal"]*64*64) \
            / (self.stats["FeatureStats"]["NumFeaturesPerClass"] \
              * self.stats["FeatureStats"]["FeatureAreaPerClass"][0]["Mean"])
        logging.info(f"Using pos_weight={self.pos_weight}")

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, is_dem):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if (is_mask or is_dem) else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2,0,1))
        
            if is_dem:
                img = (img - np.min(img))
                if np.max(img) > 0:
                    img = img / (np.max(img))
            else:
                if (img > 1).any():
                    img = img / 255.0
            
            return img
    
    @staticmethod
    def composite_bands(img, dem):
        return np.concatenate([img[0:3, :, :], dem], axis=0)
    
    
    def __getitem__(self, idx):
        img_name = self.ids[idx]
        dem_name = self.dids[idx]
        
        img_file = list(self.images_dir.glob(img_name + '.tif'))
        dem_file = list(self.dem_dir.glob(dem_name + '.tif'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {img_name}: {img_file}'
        assert len(dem_file) == 1, f'Either no DEM or multiple DEM found for the ID {dem_name}: {dem_file}'
        img = load_image(img_file[0])
        dem = load_image(dem_file[0])

        try:
            mask_name = self.mids[idx]
            mask_file = list(self.mask_dir.glob(mask_name + self.mask_suffix + '.tif'))
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {mask_name}: {mask_file}'
            mask = load_image(mask_file[0])
        except Exception as e:
            logging.debug(f'Generating zero mask for {img_file}')
            mask = Image.new('L', img.size) # defaults to black image

        assert img.size == mask.size, \
            f'Image {img_name} and mask {mask_name} should be the same size, but are {img.size} and {mask.size}'
        assert dem.size == mask.size, \
            f'DEM {dem_name} and mask {mask_name} should be the same size, but are {dem.size} and {mask.size}'
        
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, is_dem=False)
        dem = self.preprocess(self.mask_values, dem, self.scale, is_mask=False, is_dem=True)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, is_dem=False)
        
        img = self.composite_bands(img, dem)

        return {
            'image': torch.as_tensor(img.copy(), dtype=torch.float32).contiguous(),
            'mask': torch.as_tensor(mask.copy(), dtype=torch.bool).contiguous()
        }

class MEOIDataset(BasicDataset):
    def __init__(self, data_dir, scale=1):
        super().__init__(data_dir, scale, mask_suffix='')