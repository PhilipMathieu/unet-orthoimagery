# https://github.com/milesial/Pytorch-UNet/blob/master/predict.py
# modified by: James Kim
# date: Apr 11, 2023

import argparse
import logging
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask
from utils.data_loading import MEOIDataset, BasicDataset

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = np.asarray(full_img).copy()
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2,0,1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode = 'bilinear')
        if net.n_classes == 1 :
            mask = torch.sigmoid(output)  > out_threshold
    
    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', 
                        help='Sepcify the file in which the model is stored')
    parser.add_argument('--data-dir', dest='data_dir', help="Directory containing test dataset")
    parser.add_argument('--output-dir', '-o', dest='output_dir', help="Directory to place predicted masks", default='labels_pred/')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', default=False, help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0,1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    
    if mask.ndim == 3:
        mask = np.argmax(mask, axis = 0)
    
    for i, v in enumerate(mask_values):
        out[mask==i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=4, n_classes=1, bilinear=False)
    net = net.to(memory_format=torch.channels_last)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0,1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # 1. Create Dataset
    try:
        dataset = MEOIDataset(args.data_dir, args.scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(args.data_dir, args.scale)

    # 2. Create data loaders
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=False, **loader_args)

    if not os.path.isdir(os.path.join(args.data_dir, args.output_dir)):
        os.makedirs(os.path.join(args.data_dir, args.output_dir))

    # 3. Predict masks
    net.eval()
    for i, batch in enumerate(test_loader):
        images = batch['image']
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        with torch.no_grad():
            output = net(images).cpu()
            if net.n_classes == 1 :
                mask = torch.sigmoid(output)  > args.mask_threshold
            
            pil_img = mask_to_image(mask.long().squeeze().numpy(), [0,1])
            pil_img.save(os.path.join(args.data_dir, args.output_dir, os.path.split(dataset.get_filename(i)[0])[1]))

    # 4. Copy metadata
    for file in os.listdir(os.path.join(args.data_dir, 'images/')):
        if not os.path.splitext(file)[1] == '.tif':
            shutil.copy2(os.path.join(args.data_dir, 'images/', file), os.path.join(args.data_dir, args.output_dir, os.path.split(file)[1]))
