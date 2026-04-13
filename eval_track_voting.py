"""
eval_track_voting.py - Robust OCR Evaluation using Temporal Voting.

This script implements a higher-order evaluation logic that mirrors real-world 
deployment. Instead of evaluating individual frames in isolation, it groups 
5-frame video tracks and uses a 'Confidence-Weighted Majority Voting' 
mechanism to determine the final plate prediction.

This approach significantly increases the robustness of the system by 
leveraging temporal consistency to overcome transient noise or motion 
blur in individual Super-Resolved frames.

Metrics Output:
    - Track-Level Full Plate Accuracy: Percentage of tracks with 100% correct match.
    - Track-Level Character Accuracy: Total correct characters across all voted tracks.
"""

from typing import List, Dict, Any, Tuple, Union, Optional
import os
import re
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from collections import defaultdict

from dataset import create_test_dataloader
from network import ImprovedNetwork
from train import OCRModule
from utils import levenshtein

# Set up logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def vote_prediction(preds: List[str], confs: List[float]) -> str:
    """
    Selects the optimal OCR prediction from a track of images.
    
    The selection logic follows:
      1. Frequency: The most frequent prediction (majority) is the winner.
      2. Confidence Tie-breaker: If frequencies are equal, the prediction with 
         the highest maximum confidence score across its occurrences wins.
    
    Args:
        preds (List[str]): List of predicted alphanumeric strings in the track.
        confs (List[float]): Corresponding list of OCR confidence scores.
        
    Returns:
        str: The voted 'Final' plate text for the entire track.
    """
    candidates: Dict[str, Dict[str, Any]] = {}
    for p, c in zip(preds, confs):
        if p not in candidates:
            candidates[p] = {'count': 0, 'max_conf': 0.0}
        candidates[p]['count'] += 1
        candidates[p]['max_conf'] = max(candidates[p]['max_conf'], c)
    
    # Sort by count (descending), then by max_conf (descending)
    sorted_candidates = sorted(candidates.items(), key=lambda x: (x[1]['count'], x[1]['max_conf']), reverse=True)
    
    return sorted_candidates[0][0] if sorted_candidates else ""

def main(args: argparse.Namespace) -> None:
    """
    Executes the track-level voting evaluation pipeline.
    
    Workflow:
      1. Batch-processes images through the SR Network.
      2. Groups SR outputs, LR inputs, and HR targets back into their original 5-frame tracks.
      3. Applies 'Confidence-Weighted Voting' to each track group.
      4. Compares the final voted results to the Ground Truth.
      5. Reports track-level 'Plate' and 'Character' accuracy stats.
      
    Args:
        args (argparse.Namespace): Command-line configuration.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---- Load OCR Model ----
    logger.info("--- Loading OCR Model ---")
    ocr = OCRModule(args.ocr_path, device=device)
    if ocr.OCR is None:
        logger.error("OCR model not loaded! Cannot perform evaluation.")
        return
        
    # ---- Load SR Model ----
    logger.info("--- Loading SR Model ---")
    model = ImprovedNetwork(3, 3).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # ---- Load Data ----
    logger.info("--- Loading Dataset ---")
    test_loader, n_test = create_test_dataloader(
        args.dataset,
        batch_size=args.batch,
        train_ratio=0.7,
        num_workers=args.workers,
        pin_memory=True,
        seed=42
    )
    
    logger.info("\n--- Running Evaluation ---")
    to_pil = transforms.ToPILImage()
    
    # Track grouping: track_id -> dict
    tracks_data = defaultdict(lambda: {
        'gt': '',
        'hr_preds': [], 'hr_confs': [],
        'lr_preds': [], 'lr_confs': [],
        'sr_preds': [], 'sr_confs': []
    })
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='[Inference]')
        for batch in pbar:
            imgs_LR_torch = batch['LR'].to(device)
            imgs_HR_torch = batch['HR'].to(device)
            
            # SR generation
            imgs_SR_torch = model(imgs_LR_torch)
            
            list_lr, list_hr, list_sr = [], [], []
            for j in range(imgs_LR_torch.size(0)):
                list_lr.append(np.array(to_pil(imgs_LR_torch[j].cpu())).astype('uint8'))
                list_hr.append(np.array(to_pil(imgs_HR_torch[j].cpu())).astype('uint8'))
                list_sr.append(np.array(to_pil(imgs_SR_torch[j].cpu())).astype('uint8'))
                
            # Batch OCR
            preds_hr, confs_hr = ocr.predict_plates_batch(list_hr, return_conf=True)
            preds_lr, confs_lr = ocr.predict_plates_batch(list_lr, return_conf=True)
            preds_sr, confs_sr = ocr.predict_plates_batch(list_sr, return_conf=True)
            
            for j in range(imgs_LR_torch.size(0)):
                file_name = batch['file'][j]
                
                # Extract Track ID robustly
                m = re.search(r'([a-zA-Z0-9]+)_lr-\d+', file_name)
                if m:
                    track_name = m.group(1)
                    if track_name.startswith('track'):
                        track_name = track_name.replace('track', '')
                else:
                    parts = file_name.rsplit('_', 1)
                    track_name = parts[0] if len(parts) > 1 else 'unknown'
                
                # Accumulate data per track
                tracks_data[track_name]['gt'] = batch['plate_text'][j]
                tracks_data[track_name]['hr_preds'].append(preds_hr[j])
                tracks_data[track_name]['hr_confs'].append(confs_hr[j])
                
                tracks_data[track_name]['lr_preds'].append(preds_lr[j])
                tracks_data[track_name]['lr_confs'].append(confs_lr[j])
                
                tracks_data[track_name]['sr_preds'].append(preds_sr[j])
                tracks_data[track_name]['sr_confs'].append(confs_sr[j])

    # ---- Track-level Voting and Accuracy ----
    logger.info("\n--- Calculating Track-Level Accuracy (Voting) ---")
    
    total_tracks = len(tracks_data)
    
    # Full plate correct counts (100% matched)
    full_correct = {'hr': 0, 'lr': 0, 'sr': 0}
    # Character correct counts
    char_correct = {'hr': 0, 'lr': 0, 'sr': 0}
    total_chars = 0
    
    for track_name, data in tracks_data.items():
        gt = data['gt']
        gt_len = max(len(gt), 1)
        total_chars += gt_len
        
        # 1. Voting implementation (Frequency + Confidence Tie-breaker)
        final_hr = vote_prediction(data['hr_preds'], data['hr_confs'])
        final_lr = vote_prediction(data['lr_preds'], data['lr_confs'])
        final_sr = vote_prediction(data['sr_preds'], data['sr_confs'])
        
        # 2. Full Plate Accuracy Verification
        if final_hr == gt: full_correct['hr'] += 1
        if final_lr == gt: full_correct['lr'] += 1
        if final_sr == gt: full_correct['sr'] += 1
        
        # 3. Character Accuracy Evaluation
        char_correct['hr'] += (gt_len - levenshtein(gt, final_hr))
        char_correct['lr'] += (gt_len - levenshtein(gt, final_lr))
        char_correct['sr'] += (gt_len - levenshtein(gt, final_sr))

    # ---- Display Results ----
    logger.info("======================================================")
    logger.info("TRACK-LEVEL VOTING EVALUATION RESULTS")
    logger.info("======================================================")
    logger.info(f"Total Tracks Evaluated: {total_tracks} tracks (Approx {total_tracks * 5} images)")
    
    logger.info("\n[1] FULL PLATE ACCURACY (100% Characters Matched):")
    logger.info(f"  - HR (High-Res GT):    {full_correct['hr']}/{total_tracks} tracks ({(full_correct['hr']/total_tracks)*100:.2f}%)")
    logger.info(f"  - LR (Low-Res Input):  {full_correct['lr']}/{total_tracks} tracks ({(full_correct['lr']/total_tracks)*100:.2f}%)")
    logger.info(f"  - SR (Super-Res Output):{full_correct['sr']}/{total_tracks} tracks ({(full_correct['sr']/total_tracks)*100:.2f}%)")
    
    logger.info("\n[2] CHARACTER ACCURACY (Correct Characters / Total Characters):")
    logger.info(f"  - HR (High-Res GT):    {char_correct['hr']}/{total_chars} chars ({(char_correct['hr']/total_chars)*100:.2f}%)")
    logger.info(f"  - LR (Low-Res Input):  {char_correct['lr']}/{total_chars} chars ({(char_correct['lr']/total_chars)*100:.2f}%)")
    logger.info(f"  - SR (Super-Res Output):{char_correct['sr']}/{total_chars} chars ({(char_correct['sr']/total_chars)*100:.2f}%)")
    logger.info("======================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate OCR Accuracy Using Track-Level Voting")
    parser.add_argument('--dataset', required=True, help="Path to ICPR2026 train/ directory")
    parser.add_argument('--model', required=True, help="Path to SR checkpoint")
    parser.add_argument('--ocr_path', required=True, help="Path to OCR model directory")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--workers', type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    main(args)
