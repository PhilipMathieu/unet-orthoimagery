# https://github.com/milesial/Pytorch-UNet/blob/master/utils/utils.py
# modified by: James Kim, Philip Mathieu
""" ..., the dice-coefficient was chosen as loss function"""

import torch
from torch import Tensor

def dice_coeff(input:Tensor, target:Tensor, reduce_batch_first: bool = False, epsilon:float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first # Need to check

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1,-2,-3)

    inter = 2 * (input * target).sum(dim=sum_dim) # 2 * number of pixels in both input and target (i.e. intersection)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) # total number of pixels in both sets
    sets_sum = torch.where(sets_sum==0, inter, sets_sum) # if that number is 0, change to inter to avoid exploding gradient
    dice = (inter + epsilon) / (sets_sum+epsilon) # divide to get Dice coefficient
    return dice.mean() # return the mean across the batch

def multiclass_dice_coeff(input:Tensor, target:Tensor, reduce_batch_first:bool=False, epsilon:float=1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0,1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input:Tensor, target:Tensor, multiclass:bool=False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True) # True here
