"""
ocr_eval.py - Baseline Evaluation Pipeline for the OCR Teacher Model.

This script isolates the OCR component to verify its 'Ground Truth' performance 
on High-Resolution (HR) images. It serves as a sanity check: if the OCR model 
cannot correctly read HR images, the Super-Resolution 'Student' model 
cannot be effectively trained using it as a teacher.

Key Metrics:
    - Plate Accuracy: Perfect match percentage (case-insensitive).
    - Character Accuracy: Per-character correct predictions.
    - Average Levenshtein Distance: Edit distance normalized per sample.

Usage:
    python ocr_eval.py --dataset <path> --ocr_path <model_dir> --threshold 95.0
"""

from typing import Optional, Union, Any
import os
import sys
import argparse
import numpy as np
import cv2
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# Add current dir to path
sys.path.append(os.getcwd())

from train import OCRModule
from dataset import discover_tracks, flatten_pairs, ICPR2026Dataset
from utils import levenshtein
from typing import Optional, Union

def evaluate_ocr(dataset_path: Union[str, Path], ocr_path: Union[str, Path], 
                 device: str = 'cuda', limit_tracks: Optional[int] = None) -> float:
    """
    Evaluates the Teacher OCR model against reference HR images.
    
    This function standardizes the perspective of HR images (Deskewing) and 
    calculates baseline recognition metrics. A high threshold (e.g., 95%+) 
    indicates the Teacher is suitable for distillation.
    
    Args:
        dataset_path (Union[str, Path]): Root path to the dataset.
        ocr_path (Union[str, Path]): Path to the Keras OCR model assets.
        device (str): Device to run inference on.
        limit_tracks (Optional[int]): If set, restricts evaluation to N tracks 
                                      for faster execution during CI/CD.
                                      
    Returns:
        float: Final Plate Accuracy percentage.
    """
    print(f"--- Re-Evaluating OCR Teacher (Standardized) ---")
    print(f"Dataset: {dataset_path}")
    print(f"OCR Model: {ocr_path}")
    
    ocr = OCRModule(ocr_path, device=device)
    if ocr.OCR is None:
        print("Error: Could not load OCR model.")
        return 0.0
    
    # Use the discovery functions from dataset.py
    print("Discovering tracks...")
    all_tracks = discover_tracks(dataset_path)
    if limit_tracks:
        all_tracks = all_tracks[:limit_tracks]
    
    samples = flatten_pairs(all_tracks)
    
    # Create dataset (augmentation=False, rectify=True)
    full_dataset = ICPR2026Dataset(samples, augmentation=False, rectify=True)
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    correct_plates = 0
    total_samples = 0
    total_lev = 0
    total_chars = 0
    correct_chars = 0
    
    print(f"Scanning {len(all_tracks)} tracks ({len(full_dataset)} total images)...")
    
    to_pil = transforms.ToPILImage()
    
    for i, batch in enumerate(tqdm(dataloader, desc="OCR Eval")):
        # Batch is a dict: {'LR', 'HR', 'plate_text', 'plate_layout', 'scenario', 'file'}
        img_hr_tensor = batch['HR'] # (1, 3, 80, 160)
        gt_text = batch['plate_text'][0].upper()
        
        if not gt_text:
            continue
            
        # Convert tensor to numpy for OCR
        img_hr_np = np.array(to_pil(img_hr_tensor[0])).astype('uint8')
        
        # Predict
        pred = ocr.predict_plate(img_hr_np).upper()
        
        # Stats
        total_samples += 1
        if pred == gt_text:
            correct_plates += 1
        
        total_lev += levenshtein(gt_text, pred)
        
        # Per-Char
        total_chars += len(gt_text)
        min_len = min(len(gt_text), len(pred))
        for j in range(min_len):
            if gt_text[j] == pred[j]:
                correct_chars += 1
        
        if i % 500 == 0 and i > 0:
            print(f" [Mid-point] Current Plate Acc: {(correct_plates/total_samples)*100:.2f}%")

    if total_samples == 0:
        print("No valid plates found for evaluation.")
        return 0.0
        
    plate_acc = (correct_plates / total_samples) * 100
    char_acc = (correct_chars / total_chars) * 100
    avg_lev = total_lev / total_samples
    
    print("\n" + "="*40)
    print("OCR TEACHER EVALUATION RESULTS")
    print("="*40)
    print(f"Total Images:     {total_samples}")
    print(f"Plate Accuracy:   {plate_acc:.2f}%")
    print(f"Character Acc:    {char_acc:.2f}%")
    print(f"Avg Lev distance: {avg_lev:.4f}")
    print("="*40)
    
    if plate_acc < 95:
        print(f"\nWARNING: OCR Plate accuracy ({plate_acc:.2f}%) is below 95% threshold!")
        print("Consider retraining the OCR model on the HR data first.")
    else:
        print(f"\nSUCCESS: OCR Teacher is reliable ({plate_acc:.2f}%)!")
    
    return plate_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate OCR Teacher accuracy')
    parser.add_argument('--dataset', type=str,
        default='')
    parser.add_argument('--ocr_path', type=str,
        default='')
    parser.add_argument('--threshold', type=float, default=95.0)
    parser.add_argument('--limit_tracks', type=int, default=200)
    args = parser.parse_args()
    
    acc = evaluate_ocr(args.dataset, args.ocr_path, limit_tracks=args.limit_tracks)
    
    # Exit code: 0 = PASS (>= threshold), 1 = FAIL (< threshold)
    if acc >= args.threshold:
        sys.exit(0)
    else:
        sys.exit(1)
