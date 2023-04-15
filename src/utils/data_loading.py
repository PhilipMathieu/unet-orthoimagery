# https://github.com/milesial/Pytorch-UNet/blob/master/utils/data_loading.py
# modified by: James Kim
# date: Apr 14, 2023

import logging
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
    
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.tif'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
    
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, dem_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str= ''):
        self.images_dir = Path(images_dir)
        self.dem_dir = Path(dem_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0]  for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        self.dids = [splitext(file)[0]  for file in listdir(dem_dir) if isfile(join(dem_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        self.mids = [splitext(file)[0]  for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.') and file.endswith('.tif')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir = self.mask_dir, mask_suffix=self.mask_suffix), self.mids),
                total = len(self.ids)
            ))
        
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)
    
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
                img = img / (np.max(img) / 255)
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
        mask_name = self.mids[idx]
        img_file = list(self.images_dir.glob(img_name + '.tif'))
        dem_file = list(self.dem_dir.glob(dem_name + '.tif'))
        mask_file = list(self.mask_dir.glob(mask_name + self.mask_suffix + '.tif'))
        

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {img_name}: {img_file}'
        assert len(dem_file) == 1, f'Either no DEM or multiple DEM found for the ID {dem_name}: {dem_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {mask_name}: {mask_file}'
        mask = load_image(mask_file[0])
        dem = load_image(dem_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image {img_name} and mask {mask_name} should be the same size, but are {img.size} and {mask.size}'
        assert dem.size == mask.size, \
            f'DEM {dem_name} and mask {mask_name} should be the same size, but are {dem.size} and {mask.size}'
        
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, is_dem=False)
        dem = self.preprocess(self.mask_values, dem, self.scale, is_mask=False, is_dem=True)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, is_dem=False)
        
        img = self.composite_bands(img, dem)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class MEOIDataset(BasicDataset):
    def __init__(self, images_dir, dem_dir, mask_dir, scale=1):
        super().__init__(images_dir, dem_dir, mask_dir, scale, mask_suffix='_mask')