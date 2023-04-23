# https://github.com/milesial/Pytorch-UNet/blob/master/train.py
# modified by: James Kim
# date: Apr 11, 2023
# References
#   [1] "U-Net: Convolutional Networks for Biomedical Image Segmentation"
#   [2] "Convolutional Neural Networks enable efficient, accurate and fine-grained segmentation of plant species and communities from high-resolution UAV imagery"

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader, random_split

from tqdm.autonotebook import tqdm
import wandb

from evaluate import evaluate
from unet.unet_model import UNet
from utils.dice_score import dice_loss
from utils.data_loading import MEOIDataset, BasicDataset

def train_model(
        model,
        device,
        data_dir:Path,
        epochs:int=20, # 5->20
        batch_size:int=16, # 1->16
        learning_rate:float=1e-4, # 1e-5->1e-4
        val_percent:float=0.333, # 0.1->0.333
        save_checkpoint:bool=True,
        img_scale:float=0.5, # check data_loading.py
        amp:bool=False, # not mentioned in [2], need to check [1]
        weight_decay:float=1e-8, # not mentioned in [2], need to check [1]
        momentum:float=0.999, # not mentioned in [2], need to check [1]
        gradient_clipping:float=1.0, # not mentioned in [2], need to check [1]
        dir_checkpoint="checkpoint/",
        debug=False
):
    # 1. Create Dataset
    try:
        dataset = MEOIDataset(data_dir, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(data_dir, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size = batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if debug:
        experiment = wandb.init(project="U-Net", mode="disabled")
    else:
        experiment = wandb.init(project="U-Net")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, img_sclae=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
            Epoch:          {epochs}
            Batch size:     {batch_size}
            Learning rate:  {learning_rate}
            Training size:  {n_train}
            Validation size:{n_val}
            Checkpoints:    {save_checkpoint}
            Device:         {device.type}
            Images scaling: {img_scale}
            Mixed Precision:{amp}
        ''')
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                                lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # goal: maximize Dice score # not mentioned in [2], need to check [1]
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Need to check
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([dataset.pos_weight]).to(device))
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0 
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, '\
                    f'but loaded images have {images.shape[1]} channels. Please check that '\
                    'the images are loaded correctly.'
            
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        dloss = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, dim=1).float()
                        dloss = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss + dloss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss = loss.item()
                experiment.log({
                    'BCE Loss': loss.item(),
                    'Dice Loss': dloss.item(),
                    'train loss': (loss + dloss).item(),
                    'step': global_step,
                    'epoch': epoch
                })

                pbar.set_postfix({'BCE': loss.item(), 'Dice Loss': dloss.item()})

                # Evaluation round
                division_step = (n_train // (5*batch_size)) # equivalent to floor(number of batches / 5)
                if division_step > 0: # starting after 1/5 of the batches
                    if global_step % division_step == 0: # run validation when global step is multiple of 1/5 number of batches
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0][:3,:,:].cpu()), # Log only RGB
                                'masks':{
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image((F.sigmoid(masks_pred[0]).squeeze(1) > 0.5).float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception as e:
                            logging.warn(e)
                            pass
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(f'{dir_checkpoint}checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=33.3, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--data-dir', dest='data_dir', type=Path, default="data/Image_Chips_128_overlap_unbalanced_dem/", help="Directory containing dataset")
    parser.add_argument('--dir-checkpoint', dest='dir_checkpoint', type=Path, default='./checkpoints/', help="Directory in which to store PyTorch checkpoints")
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='Use debugging mode (disable WandB uploads, etc.)')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilites you want to get per pixel
    model = UNet(n_channels=4, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "transposed conv"} upscaling')
    
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            data_dir=args.data_dir,
            debug=args.debug
        )
    except torch.cuda.OutOfMemoryError: # Giving me syntax error saying '"OutOfMemoryError" is not a valid exception class.'
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            data_dir=args.data_dir,
            dir_checkpoint=args.dir_checkpoint,
            debug=args.debug
        )
