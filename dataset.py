import os
import numpy as np
import torch
import torch.utils.data
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, 
    HueSaturationValue, RandomResizedCrop, RGBShift, RandomBrightnessContrast
)
import cv2

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, 
                 img_ext='.npy', mask_ext='.npy',
                 num_classes=19, target_size=256, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.target_size = (target_size, target_size)
        self.transform = transform or self.default_transforms()

    def __len__(self):
        return len(self.img_ids)

    def default_transforms(self):
        return Compose([
            Resize(*self.target_size, always_apply=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load and convert data types
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))

        # Convert float64 to uint8 if needed
        if img.dtype == np.float64:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if mask.dtype == np.float64:
            mask = mask.astype(np.uint8)

        # Handle class labels before validation
        mask = np.clip(mask, 0, self.num_classes-1)  # CLIP BEFORE VALIDATION

        # Validate raw data
        assert img.shape[:2] == mask.shape, \
            f"Image/Mask shape mismatch for {img_id}: {img.shape[:2]} vs {mask.shape}"
        assert np.max(mask) < self.num_classes, \
            f"Invalid class {np.max(mask)} >= {self.num_classes} in {img_id}"

        # Apply transforms
        augmented = self.transform(image=img, mask=mask)
        img, mask = augmented['image'], augmented['mask'].astype(np.int64)

        # Convert to tensors
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor, {'img_id': img_id}

    @staticmethod
    def get_train_transforms(target_size=256):
        return Compose([
            RandomResizedCrop(
                height=target_size,
                width=target_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR,
                p=0.5
            ),
            HorizontalFlip(p=0.5),
            HueSaturationValue(20, 30, 20, p=0.5),
            RGBShift(25, 25, 25, p=0.5),
            RandomBrightnessContrast(p=0.5),
            Resize(target_size, target_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms(target_size=256):
        return Compose([
            Resize(target_size, target_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def verify_dataset_structure(base_path="inputs"):
        for split in ['train', 'val']:
            split_path = os.path.join(base_path, split)
            
            for mod in ['images', 'masks']:
                path = os.path.join(split_path, mod)
                assert os.path.exists(path), f"Missing {split}/{mod}"
                assert len(os.listdir(path)) > 0, f"Empty {split}/{mod}"
                assert all(f.endswith('.npy') for f in os.listdir(path)), "Non-NPY files found"

            images = set(f.split('.')[0] for f in os.listdir(f"{split_path}/images"))
            masks = set(f.split('.')[0] for f in os.listdir(f"{split_path}/masks"))
            assert images == masks, f"Image-Mask mismatch in {split}"

        print("Dataset structure validated!")

if __name__ == '__main__':
    CityscapesDataset.verify_dataset_structure()
    
    train_dataset = CityscapesDataset(
        img_ids=[f.replace('.npy', '') for f in os.listdir("inputs/train/images")],
        img_dir="inputs/train/images",
        mask_dir="inputs/train/masks",
        transform=CityscapesDataset.get_train_transforms(256),
        num_classes=19
    )
    
    img, mask, meta = train_dataset[0]
    print(f"\nSample 0: {meta['img_id']}")
    print(f"Image shape: {img.shape}")  # Should be torch.Size([3, 256, 256])
    print(f"Mask shape: {mask.shape}")  # Should be torch.Size([256, 256])
    print(f"Unique mask values: {torch.unique(mask)}")
