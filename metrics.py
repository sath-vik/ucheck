import numpy as np
import torch
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

def calculate_metrics(output, target, num_classes, ignore_index=255):
    """
    Computes comprehensive segmentation metrics for multi-class problems
    Returns dictionary with:
    - class_iou: IoU per class
    - mean_iou: Average IoU
    - pixel_accuracy: Overall pixel accuracy
    - mean_accuracy: Average class accuracy
    - confusion_matrix: Full confusion matrix
    """
    # Convert output probabilities to predictions
    pred = output.argmax(1)
    
    # Create mask for valid pixels
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]

    # Calculate confusion matrix
    hist = torch.bincount(
        num_classes * target.flatten() + pred.flatten(),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    # Calculate metrics
    tp = torch.diag(hist)
    fp = hist.sum(0) - tp
    fn = hist.sum(1) - tp
    tn = hist.sum() - (tp + fp + fn)

    # Avoid division by zero
    iou = tp / (tp + fp + fn + 1e-10)
    pixel_acc = (tp.sum() + tn.sum()) / hist.sum()
    mean_acc = (tp / (tp + fn + 1e-10)).mean()

    return {
        'class_iou': iou.cpu().numpy(),
        'mean_iou': iou.mean().item(),
        'pixel_accuracy': pixel_acc.item(),
        'mean_accuracy': mean_acc.item(),
        'confusion_matrix': hist.cpu().numpy()
    }

def print_metrics(metrics, class_names=None):
    """Pretty-print metrics dictionary"""
    print("\nEvaluation Metrics:")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    
    if class_names and len(class_names) == len(metrics['class_iou']):
        print("\nClass-wise IoU:")
        for cls_idx, iou in enumerate(metrics['class_iou']):
            print(f"{class_names[cls_idx]:15}: {iou:.4f}")

# Legacy functions for backward compatibility
def iou_score(output, target):
    metrics = calculate_metrics(output, target, num_classes=output.shape[1])
    return metrics['mean_iou']

def dice_score(output, target):
    pred = output.argmax(1)
    intersection = (pred & target).float().sum()
    return (2. * intersection) / (pred.float().sum() + target.float().sum() + 1e-6)

def indicators(output, target):
    pred = output.argmax(1).cpu().numpy()
    target = target.cpu().numpy()
    return {
        'iou': jc(pred, target),
        'dice': dc(pred, target),
        'hd': hd(pred, target),
        'hd95': hd95(pred, target),
        'recall': recall(pred, target),
        'specificity': specificity(pred, target),
        'precision': precision(pred, target)
    }
