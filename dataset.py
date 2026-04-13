"""
dataset.py - ICPR2026 License Plate Dataset Loader.

This module implements a robust data pipeline for the ICPR2026 Competition.
It handles multiple scenarios (Scenario-A/B) and Brazilian/Mercosur layouts,
providing sophisticated preprocessing such as perspective rectification 
and aspect-ratio-aware padding.

Key Features:
    - Track-based splitting: Prevents data leakage by keeping video tracks isolated.
    - Perspective Rectification: Uses corner annotations to deskew plate images.
    - Aspect Ratio Padding: Ensures consistent input shapes without distortion.
    - Brazilian/Mercosur support: Handles varying plate geometries.
"""

import os
import cv2
import sys
import json
import random
import numpy as np
import torch
import albumentations as A
from typing import List, Dict, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms

from utils import padding, rectify_image


# ============================================================================
# Constants
# ============================================================================

IMG_LR = (40, 20)    # (width, height) for LR images
IMG_HR = (160, 80)   # (width, height) for HR images, 4x upscale

ASPECT_RATIO = 2.0
MIN_RATIO = ASPECT_RATIO - 0.15
MAX_RATIO = ASPECT_RATIO + 0.15
BG_COLOR = (127, 127, 127)


# ============================================================================
# Dataset Discovery
# ============================================================================

def discover_tracks(dataset_root: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Scans the ICPR2026 filesystem to discover and structure video tracks.
    
    The dataset is structured as: root > Scenario > Layout > TrackID > [Images + Json].
    This function parses the hierarchy and builds a mapping of LR-HR image pairs
    linked to their respective alphanumeric ground truth and corner coordinates.
    
    Args:
        dataset_root (Union[str, Path]): Root directory of the dataset (e.g., 'train/').
    
    Returns:
        List[Dict[str, Any]]: A list of structured track metadata dictionaries.
    """
    dataset_root = Path(dataset_root)
    tracks = []
    
    for scenario_dir in sorted(dataset_root.iterdir()):
        if not scenario_dir.is_dir() or not scenario_dir.name.startswith('Scenario'):
            continue
        scenario_name = scenario_dir.name  # e.g., "Scenario-A"
        
        for layout_dir in sorted(scenario_dir.iterdir()):
            if not layout_dir.is_dir():
                continue
            layout_name = layout_dir.name  # e.g., "Brazilian" or "Mercosur"
            
            for track_dir in sorted(layout_dir.iterdir()):
                if not track_dir.is_dir():
                    continue
                
                ann_path = track_dir / 'annotations.json'
                if not ann_path.exists():
                    continue
                
                try:
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue
                
                plate_text = ann.get('plate_text', '')
                plate_layout = ann.get('plate_layout', layout_name)
                corners = ann.get('corners', {})
                
                # Find all LR-HR pairs (lr-001 -> hr-001, etc.)
                pairs = []
                for i in range(1, 6):  # 001 to 005
                    prefix_lr = f'lr-{i:03d}'
                    prefix_hr = f'hr-{i:03d}'
                    
                    lr_path = None
                    hr_path = None
                    
                    # Check for both png and jpg
                    for ext in ['.png', '.jpg', '.jpeg']:
                        p_lr = track_dir / (prefix_lr + ext)
                        p_hr = track_dir / (prefix_hr + ext)
                        if p_lr.exists() and p_hr.exists():
                            lr_path = p_lr
                            hr_path = p_hr
                            break
                    
                    if lr_path and hr_path:
                        lr_corners = corners.get(lr_path.name, None)
                        hr_corners = corners.get(hr_path.name, None)
                        pairs.append((
                            str(lr_path),
                            str(hr_path),
                            lr_corners,
                            hr_corners
                        ))
                
                if len(pairs) > 0:
                    tracks.append({
                        'track_path': str(track_dir),
                        'plate_text': plate_text,
                        'plate_layout': plate_layout,
                        'scenario': scenario_name,
                        'pairs': pairs
                    })
    
    return tracks


def split_tracks(tracks: List[Dict[str, Any]], train_ratio: float = 0.7, 
                 seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition the dataset into training and validation sets at the TRACK level.
    
    CRITICAL: We split by track (sequence) rather than individual images to prevent
    'Data Leakage'. Since frames in a track are highly correlated, having frames 
    from the same track in both train and val would result in artificially high 
    validation accuracy and poor real-world generalization.
    
    Args:
        tracks (List[Dict[str, Any]]): Full list of discovered tracks.
        train_ratio (float): Fraction of tracks to allocate to training.
        seed (int): Random seed for deterministic reproducibility.
    
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (train_tracks, val_tracks)
    """
    rng = random.Random(seed)
    shuffled = list(tracks)
    rng.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_tracks = shuffled[:split_idx]
    val_tracks = shuffled[split_idx:]
    
    return train_tracks, val_tracks


def flatten_pairs(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts track-level metadata into a flat list of individual image pair samples.
    
    Args:
        tracks (List[Dict[str, Any]]): List of track objects.
        
    Returns:
        List[Dict[str, Any]]: List of sample objects for the PyTorch Dataset.
    """
    samples = []
    for track in tracks:
        for lr_path, hr_path, lr_corners, hr_corners in track['pairs']:
            samples.append({
                'lr_path': lr_path,
                'hr_path': hr_path,
                'plate_text': track['plate_text'],
                'plate_layout': track['plate_layout'],
                'scenario': track['scenario'],
                'lr_corners': lr_corners,
                'hr_corners': hr_corners,
            })
    return samples


# ============================================================================
# Dataset Class
# ============================================================================

class ICPR2026Dataset(Dataset):
    """
    Advanced PyTorch Dataset for License Plate Super-Resolution.
    
    This class manages the entire preprocessing lifecycle for a single image pair:
      1. Loading (RGB).
      2. Perspective Rectification (using ground truth corners).
      3. Global Augmentation (Color, Brightness, Gamma).
      4. Aspect-Ratio-Aware Padding.
      5. Resizing to fixed dimensions (LR 40x20, HR 160x80).
      6. Tensor Conversion.
    """
    
    def __init__(self, samples: List[Dict[str, Any]], augmentation: bool = True, 
                 rectify: bool = True) -> None:
        """
        Initializes the dataset with a list of flattened image pairs.
        
        Args:
            samples (List[Dict[str, Any]]): List of metadata dicts for each image pair.
            augmentation (bool): Whether to apply stochastic color augmentations.
            rectify (bool): Whether to deskew images using corner annotations.
        """
        self.samples = samples
        self.to_tensor = transforms.ToTensor()
        self.augmentation = augmentation
        self.rectify = rectify
        
        # Augmentation transforms (same as original paper)
        self.transformHR = [
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                  val_shift_limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                       brightness_by_max=True, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            None  # No augmentation (identity)
        ]
        
        self.transformLR = [
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                  val_shift_limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                       brightness_by_max=True, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            None  # No augmentation
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves and preprocesses the sample at the given index.
        
        Args:
            index (int): Index of the sample.
            
        Returns:
            Dict[str, Any]: {
                'LR': torch.Tensor,
                'HR': torch.Tensor,
                'plate_text': str,
                'plate_layout': str,
                'scenario': str,
                'file': str
            }
        """
        sample = self.samples[index]
        
        # Load images
        imgLR = np.array(Image.open(sample['lr_path']))
        imgHR = np.array(Image.open(sample['hr_path']))
        
        # 1. Perspective rectification (Deskewing)
        if self.rectify:
            if sample['lr_corners'] is not None:
                try:
                    imgLR = rectify_image(imgLR, sample['lr_corners'])
                except Exception:
                    pass  # Fallback to crop if rectification fails
            if sample['hr_corners'] is not None:
                try:
                    imgHR = rectify_image(imgHR, sample['hr_corners'])
                except Exception:
                    pass
        
        # 2. Data augmentation (Color jittering)
        if self.augmentation:
            augLR = random.choice(self.transformLR)
            augHR = random.choice(self.transformHR)
            
            if augHR is not None:
                imgHR = augHR(image=imgHR)["image"]
            if augLR is not None:
                imgLR = augLR(image=imgLR)["image"]
        
        # 3. Padding (To maintain geometry without stretching)
        imgLR, _, _ = padding(imgLR, MIN_RATIO, MAX_RATIO, color=BG_COLOR)
        imgHR, _, _ = padding(imgHR, MIN_RATIO, MAX_RATIO, color=BG_COLOR)
        
        # 4. Final Resize (Fixed batch dimensions)
        imgLR = cv2.resize(imgLR, IMG_LR, interpolation=cv2.INTER_CUBIC)
        imgHR = cv2.resize(imgHR, IMG_HR, interpolation=cv2.INTER_CUBIC)
        
        # 5. Tensor Conversion
        imgLR = self.to_tensor(imgLR)
        imgHR = self.to_tensor(imgHR)
        
        # Human-readable reference ID
        file_name = Path(sample['lr_path']).parent.name + '_' + Path(sample['lr_path']).name
        
        return {
            'LR': imgLR,
            'HR': imgHR,
            'plate_text': sample['plate_text'],
            'plate_layout': sample['plate_layout'],
            'scenario': sample['scenario'],
            'file': file_name
        }


# ============================================================================
# DataLoader Factory
# ============================================================================

def create_dataloaders(dataset_root: Union[str, Path], batch_size: int = 8, 
                       train_ratio: float = 0.7, num_workers: int = 4, 
                       pin_memory: bool = True, seed: int = 42) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Standard high-level factory to generate training and validation DataLoaders.
    
    Args:
        dataset_root (Union[str, Path]): Path to dataset root.
        batch_size (int): Training hyperparameter for batch size.
        train_ratio (float): Split percentage.
        num_workers (int): DataLoader parallelism count.
        pin_memory (bool): Whether to use CUDA pinned memory.
        seed (int): Determinism seed.
        
    Returns:
        Tuple[DataLoader, DataLoader, int, int]: 
            (TrainLoader, ValLoader, TrainSampleCount, ValSampleCount)
    """
    print(f"[Dataset] Discovering tracks in: {dataset_root}")
    tracks = discover_tracks(dataset_root)
    print(f"[Dataset] Found {len(tracks)} tracks total")
    
    train_tracks, val_tracks = split_tracks(tracks, train_ratio, seed)
    print(f"[Dataset] Split: {len(train_tracks)} train tracks, {len(val_tracks)} val tracks")
    
    train_samples = flatten_pairs(train_tracks)
    val_samples = flatten_pairs(val_tracks)
    print(f"[Dataset] Samples: {len(train_samples)} train, {len(val_samples)} val")
    
    # Count by layout
    for split_name, samples in [("Train", train_samples), ("Val", val_samples)]:
        layouts = {}
        scenarios = {}
        for s in samples:
            layouts[s['plate_layout']] = layouts.get(s['plate_layout'], 0) + 1
            scenarios[s['scenario']] = scenarios.get(s['scenario'], 0) + 1
        print(f"[Dataset] {split_name} layouts: {layouts}")
        print(f"[Dataset] {split_name} scenarios: {scenarios}")
    
    train_dataset = ICPR2026Dataset(train_samples, augmentation=False, rectify=True)
    val_dataset = ICPR2026Dataset(val_samples, augmentation=False, rectify=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, len(train_samples), len(val_samples)


def create_test_dataloader(dataset_root: str | Path, batch_size: int = 8, train_ratio: float = 0.7,
                           num_workers: int = 4, pin_memory: bool = True, seed: int = 42) -> Tuple[DataLoader, int]:
    """
    Create test/validation dataloader (uses the val split).
    """
    tracks = discover_tracks(dataset_root)
    _, val_tracks = split_tracks(tracks, train_ratio, seed)
    val_samples = flatten_pairs(val_tracks)
    
    test_dataset = ICPR2026Dataset(val_samples, augmentation=False, rectify=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return test_loader, len(val_samples)


# ============================================================================
# Debug / Test
# ============================================================================

if __name__ == '__main__':
    import sys
    
    dataset_root = ''
    
    print("Testing dataset discovery...")
    tracks = discover_tracks(dataset_root)
    print(f"Found {len(tracks)} tracks")
    
    if len(tracks) > 0:
        print(f"\nSample track:")
        t = tracks[0]
        print(f"  Path: {t['track_path']}")
        print(f"  Plate: {t['plate_text']}")
        print(f"  Layout: {t['plate_layout']}")
        print(f"  Scenario: {t['scenario']}")
        print(f"  Pairs: {len(t['pairs'])}")
        
        # Test split
        train_t, val_t = split_tracks(tracks)
        print(f"\nSplit: {len(train_t)} train, {len(val_t)} val")
        
        # Test dataset
        train_samples = flatten_pairs(train_t[:10])  # Just 10 tracks for quick test
        ds = ICPR2026Dataset(train_samples, augmentation=True)
        
        sample = ds[0]
        print(f"\nSample item:")
        print(f"  LR shape: {sample['LR'].shape}")
        print(f"  HR shape: {sample['HR'].shape}")
        print(f"  Plate: {sample['plate_text']}")
        print(f"  Layout: {sample['plate_layout']}")
        print(f"  File: {sample['file']}")
        
        print("\nDataset test passed!")
