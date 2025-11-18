import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Remove pixel limit
Image.MAX_IMAGE_PIXELS = None

class SalientDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size=768, mode='train'):
        """
        Args:
            img_paths (list): List of image file paths.
            mask_paths (list): List of corresponding mask file paths.
            img_size (int): Target size (e.g., 768).
            mode (str): 'train' or 'val'.
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.mode = mode
        
        # ImageNet normalization (Standard for Swin Transformer)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Image and Mask
        image = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L') # Grayscale (0-255)

        # 2. Synchronized Transform (Resize & Flip)
        # We must apply the SAME transform to both Image and Mask
        
        # Resize
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BICUBIC)
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        # Data Augmentation (Only for training)
        if self.mode == 'train':
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Vertical Flip (Optional, good for plants)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # 3. Convert to Tensor
        # Image: [0, 255] -> [0.0, 1.0] -> Normalized
        img_tensor = TF.to_tensor(image)
        img_tensor = self.normalize(img_tensor)

        # Mask: [0, 255] -> [0.0, 1.0]
        mask_tensor = TF.to_tensor(mask)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'path': self.img_paths[idx] # Useful for debugging
        }

def get_loaders(data_root, img_size=768, batch_size=4, split_ratio=0.2, num_workers=4):
    """
    Automatically splits data and returns train/val loaders.
    """
    # Paths
    img_dir = os.path.join(data_root, 'images')
    mask_dir = os.path.join(data_root, 'masks')

    # Get all image files (jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png']
    img_list = []
    for ext in extensions:
        img_list.extend(glob.glob(os.path.join(img_dir, ext)))
    
    if len(img_list) == 0:
        raise ValueError(f"No images found in {img_dir}")

    # Sort to ensure consistent order before shuffling
    img_list.sort()
    
    # Pair images with masks
    # Assumption: Mask has same filename as Image (but maybe different extension)
    final_img_paths = []
    final_mask_paths = []

    for img_path in img_list:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Try finding corresponding mask (png or jpg)
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(mask_dir, name_no_ext + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path:
            final_img_paths.append(img_path)
            final_mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask not found for {filename}, skipping.")

    print(f"Total valid samples found: {len(final_img_paths)}")

    # Shuffle and Split
    indices = list(range(len(final_img_paths)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * split_ratio)
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]

    # Create Lists
    train_imgs = [final_img_paths[i] for i in train_indices]
    train_masks = [final_mask_paths[i] for i in train_indices]
    val_imgs = [final_img_paths[i] for i in val_indices]
    val_masks = [final_mask_paths[i] for i in val_indices]

    print(f"Train Size: {len(train_imgs)}, Val Size: {len(val_imgs)}")

    # Create Datasets
    train_dataset = SalientDataset(train_imgs, train_masks, img_size=img_size, mode='train')
    val_dataset = SalientDataset(val_imgs, val_masks, img_size=img_size, mode='val')

    # Create Loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True # Drop incomplete batch to avoid errors
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader