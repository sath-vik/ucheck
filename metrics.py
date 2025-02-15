import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

def iou_score(output, target):
    # Convert output probabilities to predicted class indices
    output = torch.argmax(output, dim=1)  # Reduces from [B,C,H,W] to [B,H,W]
    
    # Now both output and target are [B,H,W]
    intersection = (output & target).float().sum((1, 2))  # Intersection per sample
    union = (output | target).float().sum((1, 2))         # Union per sample
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
    return iou.mean().item()  # Return average IoU across batch

def dice_score(output, target):
    output = torch.argmax(output, dim=1)
    intersection = (output & target).float().sum((1, 2))
    return (2. * intersection / (output.float().sum((1,2)) + target.float().sum((1,2)) + 1e-6)).mean().item()


# def dice_score(output, target):
#     smooth = 1e-5
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()
#     return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

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
