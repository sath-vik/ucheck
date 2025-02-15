import os
import numpy as np
import torch
import torch.utils.data
from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90
)

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, 
                 img_ext='.npy', mask_ext='.npy',
                 num_classes=19, target_size=256, transform=None):
        """
        Custom dataset for Cityscapes .npy files with automatic resizing
        Args:
            target_size: Final output size (height, width)
            num_classes: Number of valid classes (0-18)
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.target_size = (target_size, target_size)  # Force square output
        self.transform = transform or self.default_transforms()

    def __len__(self):
        return len(self.img_ids)

    def default_transforms(self):
        """Ensures 256x256 output regardless of input size"""
        return Compose([
            Resize(*self.target_size, always_apply=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load .npy files
        img = np.load(os.path.join(self.img_dir, img_id + self.img_ext))  # (H,W,3)
        mask = np.load(os.path.join(self.mask_dir, img_id + self.mask_ext))  # (H,W)

        # Validate raw dimensions
        assert img.shape[:2] == mask.shape, \
            f"Image/Mask shape mismatch: {img.shape[:2]} vs {mask.shape} for {img_id}"
            
        # Apply transforms (resize to 256x256 + augmentations)
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        # Process mask
        mask = mask.astype(np.int64)
        mask = np.clip(mask, 0, self.num_classes-1)  # Remove negative values
        
        # Validate final dimensions
        assert img.shape[:2] == self.target_size, \
            f"Image wrong size after transform: {img.shape[:2]}"
        assert mask.shape == self.target_size, \
            f"Mask wrong size after transform: {mask.shape}"

        # Convert to tensors
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)  # (C,H,W)
        mask_tensor = torch.from_numpy(mask).long()  # (H,W)

        return img_tensor, mask_tensor, {'img_id': img_id}

    @staticmethod
    def get_train_transforms(target_size=256):
        return Compose([
            RandomRotate90(),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
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
        """Verify folder structure and file counts"""
        for split in ['train', 'val']:
            split_path = os.path.join(base_path, split)
            assert os.path.exists(split_path), f"Missing {split} directory"
            
            for modality in ['images', 'masks']:
                mod_path = os.path.join(split_path, modality)
                assert os.path.exists(mod_path), f"Missing {split}/{modality}"
                
                files = os.listdir(mod_path)
                assert len(files) > 0, f"No files in {split}/{modality}"
                assert all(f.endswith('.npy') for f in files), "Non-NPY files found"

        print("Dataset structure validated successfully!")

# Usage example (put this in your training script)
if __name__ == '__main__':
    # First verify dataset structure
    CityscapesDataset.verify_dataset_structure()
    
    # Example initialization
    train_img_dir = "inputs/train/images"
    train_img_ids = [f.replace('.npy', '') for f in os.listdir(train_img_dir)]
    
    train_dataset = CityscapesDataset(
        img_ids=train_img_ids,
        img_dir=train_img_dir,
        mask_dir="inputs/train/masks",
        transform=CityscapesDataset.get_train_transforms(256),
        num_classes=19
    )
    
    # Check first sample
    img, mask, meta = train_dataset[0]
    print(f"Sample 0: {meta['img_id']}")
    print(f"Image shape: {img.shape}")  # Should be torch.Size([3, 256, 256])
    print(f"Mask shape: {mask.shape}")   # Should be torch.Size([256, 256])
    print(f"Unique mask values: {torch.unique(mask)}")
