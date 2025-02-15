import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    
    try:
        hd95_val = hd95(output_, target_)
    except:
        hd95_val = 0
    
    return iou, dice, hd95_val

def dice_score(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    output_ = output > 0.5
    target_ = target > 0.5

    return {
        'iou': jc(output_, target_),
        'dice': dc(output_, target_),
        'hd': hd(output_, target_),
        'hd95': hd95(output_, target_),
        'recall': recall(output_, target_),
        'specificity': specificity(output_, target_),
        'precision': precision(output_, target_)
    }
