import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import (
    Compose, Normalize, RandomRotate90, Resize, HorizontalFlip, VerticalFlip
)
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs
import losses
from dataset import CityscapesDataset
from metrics import calculate_metrics, print_metrics, iou_score, dice_score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
import shutil

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def list_type(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('--dataseed', default=2981, type=int)
    
    # Model parameters
    parser.add_argument('--arch', default='UKAN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=19, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--input_list', type=list_type, default=[512, 640, 1024])
    
    # Dataset
    parser.add_argument('--dataset', default='cityscapes', help='dataset name')  
    parser.add_argument('--data_dir', default='inputs', help='dataset root dir')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    
    # Optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)
    
    # Scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR')
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--no_kan', action='store_true')
    
    return parser.parse_args()

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader), desc="Training")
    for img, mask, _ in train_loader:
        inputs = img.cuda()
        targets = mask.cuda()

        if config['deep_supervision']:
            outputs = model(inputs)
            loss = sum(criterion(o, targets) for o in outputs) / len(outputs)
            iou = iou_score(outputs[-1], targets)
        else:
            output = model(inputs)
            loss = criterion(output, targets)
            iou = iou_score(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), inputs.size(0))
        avg_meters['iou'].update(iou, inputs.size(0))
        
        pbar.set_postfix(OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ]))
        pbar.update(1)
    pbar.close()
    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validation")
        for img, mask, _ in val_loader:
            inputs = img.cuda()
            targets = mask.cuda()

            if config['deep_supervision']:
                outputs = model(inputs)
                loss = sum(criterion(o, targets) for o in outputs) / len(outputs)
                iou = iou_score(outputs[-1], targets)
                dice = dice_score(outputs[-1], targets).item()  # Convert to float
            else:
                output = model(inputs)
                loss = criterion(output, targets)
                iou = iou_score(output, targets)
                dice = dice_score(output, targets).item()

            avg_meters['loss'].update(loss.item(), inputs.size(0))
            avg_meters['iou'].update(iou, inputs.size(0))
            avg_meters['dice'].update(dice.item() if torch.is_tensor(dice) else dice, inputs.size(0))
            
            pbar.set_postfix(OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ]))
            pbar.update(1)
        pbar.close()
    # Change return statement in validate():
    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])

def evaluate_test(model, test_loader, num_classes):
    model.eval()
    metrics = {
        'iou': torch.zeros(num_classes).cuda(),
        'pixel_acc': 0.0,
        'mean_acc': 0.0,
        'count': 0
    }
    
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc="Testing")
        for img, mask, _ in test_loader:
            inputs = img.cuda()
            targets = mask.cuda()
            
            outputs = model(inputs)
            batch_metrics = calculate_metrics(outputs, targets, num_classes)
            
            # Convert numpy array to CUDA tensor before accumulation
            batch_iou = torch.from_numpy(batch_metrics['class_iou']).cuda()
            metrics['iou'] += batch_iou
            
            metrics['pixel_acc'] += batch_metrics['pixel_accuracy'] * inputs.size(0)
            metrics['mean_acc'] += batch_metrics['mean_accuracy'] * inputs.size(0)
            metrics['count'] += inputs.size(0)
            pbar.update(1)
        pbar.close()
    
    return {
        'class_iou': (metrics['iou'] / metrics['count']).cpu().numpy(),
        'mean_iou': (metrics['iou'].sum() / (metrics['count'] * num_classes)).item(),
        'pixel_accuracy': metrics['pixel_acc'] / metrics['count'],
        'mean_accuracy': metrics['mean_acc'] / metrics['count']
    }

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_torch()
    config = vars(parse_args())
    
    # Initialize experiment
    if not config['name']:
        config['name'] = f"{config['dataset']}_{config['arch']}_DS{config['deep_supervision']}"
    
    exp_dir = os.path.join(config['output_dir'], config['name'])
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    print('-'*20 + ' CONFIG ' + '-'*20)
    for k,v in config.items(): print(f"{k:20}: {v}")
    print('-'*48)

    # Initialize model
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision']
    ).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer setup
    param_groups = []
    for name, param in model.named_parameters():
        if 'kan' in name.lower():
            param_groups.append({'params': param, 
                               'lr': config['kan_lr'],
                               'weight_decay': config['kan_weight_decay']})
        else:
            param_groups.append({'params': param,
                               'lr': config['lr'],
                               'weight_decay': config['weight_decay']})
    
    optimizer = optim.Adam(param_groups)
    
    # Scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    else:
        scheduler = None

    # Create dataset splits
    train_ids, test_ids = CityscapesDataset.create_splits(config['data_dir'])
    
    # Initialize datasets
    train_dataset = CityscapesDataset(
        img_ids=train_ids,
        img_dir=os.path.join(config['data_dir'], 'train', 'images'),
        mask_dir=os.path.join(config['data_dir'], 'train', 'masks'),
        num_classes=config['num_classes'],
        transform=CityscapesDataset.get_train_transforms(config['input_h'])
    )

    val_dataset = CityscapesDataset(
        img_ids=[f.replace('.npy','') for f in os.listdir(os.path.join(config['data_dir'], 'val', 'images'))],
        img_dir=os.path.join(config['data_dir'], 'val', 'images'),
        mask_dir=os.path.join(config['data_dir'], 'val', 'masks'),
        num_classes=config['num_classes'],
        transform=CityscapesDataset.get_val_transforms(config['input_h'])
    )

    test_dataset = CityscapesDataset(
        img_ids=test_ids,
        img_dir=os.path.join(config['data_dir'], 'train', 'images'),
        mask_dir=os.path.join(config['data_dir'], 'train', 'masks'),
        num_classes=config['num_classes'],
        transform=CityscapesDataset.get_val_transforms(config['input_h'])
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # Training loop
    best_iou = 0
    log = OrderedDict([
        ('epoch', []), ('lr', []), ('loss', []), ('iou', []),
        ('val_loss', []), ('val_iou', []), ('val_dice', [])
    ])
    
    writer = SummaryWriter(exp_dir)
    
    for epoch in range(config['epochs']):
        print(f'Epoch [{epoch+1}/{config["epochs"]}]')
        
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)
        
        if scheduler: scheduler.step()
        
        # Update logs
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        
        # Save checkpoints
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))
            best_iou = val_log['iou']
            print(f"Saved best model (IoU: {best_iou:.4f})")
        
        # Tensorboard logging
        writer.add_scalars('loss', {
            'train': train_log['loss'],
            'val': val_log['loss']
        }, epoch)
        
        writer.add_scalars('iou', {
            'train': train_log['iou'],
            'val': val_log['iou']
        }, epoch)
        
        pd.DataFrame(log).to_csv(os.path.join(exp_dir, 'log.csv'), index=False)
        
    writer.close()
    print("Training completed!")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'model.pth')))
    
    class_names = [
        'road', 'sidewalk', 'building', 'wall', 'fence',
        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
        'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    test_metrics = evaluate_test(model, test_loader, config['num_classes'])
    print_metrics(test_metrics, class_names)

if __name__ == '__main__':
    main()
