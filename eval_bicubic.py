"""
eval_bicubic.py - Baseline Evaluation using Bicubic Interpolation.

This script calculates the baseline PSNR and SSIM metrics by upscaling 
low-resolution (LR) images using standard OpenCV Bicubic Interpolation. 
It serves as a reference point for evaluating the performance gain of 
neural Super-Resolution models.

Usage:
    python eval_bicubic.py --dataset /path/to/dataset --train_ratio 0.7
"""

import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from dataset import create_dataloaders
from utils import calculate_psnr, calculate_ssim


def main():
    parser = argparse.ArgumentParser(description='Bicubic Interpolation Baseline Evaluation')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to ICPR2026 train/ directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train/val split ratio (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # ---- Load Data ----
    print(f"--- Loading Dataset (Seed: {args.seed}) ---")
    _, val_loader, _, n_val = create_dataloaders(
        args.dataset, 
        batch_size=1, 
        train_ratio=args.train_ratio,
        num_workers=4,
        seed=args.seed
    )
    
    print(f"Validation samples: {n_val}")
    
    psnr_values = []
    ssim_values = []
    
    # ---- Evaluation Loop ----
    print("\n--- Running Bicubic Interpolation Baseline ---")
    for batch in tqdm(val_loader, desc='Bicubic Eval'):
        # Extract tensors
        # Tensors are (1, 3, H, W) and normalized to [0, 1]
        lr_tensor = batch['LR'][0] 
        hr_tensor = batch['HR'][0]
        
        # Convert to numpy uint8 (0-255) for cv2 and metric calculation
        # LR: (3, H_lr, W_lr) -> (H_lr, W_lr, 3)
        lr_np = (lr_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype('uint8')
        # HR: (3, H_hr, W_hr) -> (H_hr, W_hr, 3)
        hr_np = (hr_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype('uint8')
        
        # Get target dimensions
        target_h, target_w = hr_np.shape[0], hr_np.shape[1]
        
        # Apply Bicubic Interpolation
        bicubic_sr = cv2.resize(lr_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Calculate Metrics
        try:
            psnr = calculate_psnr(hr_np, bicubic_sr)
            ssim = calculate_ssim(hr_np, bicubic_sr)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            continue

    # ---- Results ----
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print("\n" + "="*40)
    print("  BICUBIC INTERPOLATION BASELINE RESULTS")
    print("="*40)
    print(f"  Method: Bicubic Interpolation")
    print(f"  PSNR:   {avg_psnr:.4f} dB")
    print(f"  SSIM:   {avg_ssim:.4f}")
    print("="*40)
    print(f"\nTable Row Snippet:")
    print(f"Bicubic Interpolation (Baseline) & {avg_psnr:.2f} & {avg_ssim:.4f}")


if __name__ == '__main__':
    main()
