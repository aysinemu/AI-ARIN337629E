"""
test.py - Comprehensive Evaluation Pipeline for License Plate Super-Resolution.

This script performs a rigorous assessment of the SR model's performance on the 
validation/test split of the ICPR2026 dataset. It generates quantitative metrics 
and qualitative visualizations to validate model efficacy across different scenarios.

Metrics:
    - PSNR (Peak Signal-to-Noise Ratio): Pixel-level reconstruction quality.
    - SSIM (Structural Similarity): Human-perceptual visual quality.
    - OCR Accuracy: Character-level correctness (Perfect match vs. Edit distance).
    - Confidence Scores: Model certainty for HR, LR, and SR predictions.

Deliverables:
    - results.csv: Per-image detailed metrics.
    - results_detailed.csv: Metrics with global averages.
    - accuracy_histogram.png: Visual distribution of character-level success.
    - confusion_matrix_SR.png: Mapping of character prediction errors.
    - track_grids/: 5x3 visual comparison grids for visual audit.
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import os
import re
import torch
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

# LOAD TENSORFLOW SECOND
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"TensorFlow Memory Growth Error: {e}")

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from network import ImprovedNetwork
from dataset import create_test_dataloader
from utils import (
    setup_logging, levenshtein, calculate_psnr, calculate_ssim,
    build_confusion_matrix, plot_confusion_matrix,
    load_training_state
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ============================================================================
# Import OCR Module from train.py
# ============================================================================
from train import OCRModule


# ============================================================================
# Testing Function
# ============================================================================

def test(args: argparse.Namespace) -> None:
    """
    Executes the full evaluation cycle on the verification dataset.
    
    Workflow:
      1. Initializes OCR Teacher and SR Student models.
      2. Loads the best available model checkpoint.
      3. Streams batches through the SR network.
      4. Calculates multi-modal metrics (Pixel, Perceptual, OCR).
      5. Generates statistical summaries and visual audits.
      
    Args:
        args (argparse.Namespace): Command-line configuration parameters.
    """
    
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(save_dir, name='testing')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("LP Super-Resolution Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    
    # ---- Load OCR ----
    logger.info("\n--- Loading OCR Model ---")
    ocr = OCRModule(args.ocr_path, logger=logger, device=str(device))
    
    if ocr.OCR is None:
        logger.error("OCR model not loaded! Cannot perform evaluation.")
        return
    
    # ---- Load Model ----
    logger.info("\n--- Loading SR Model ---")
    model = ImprovedNetwork(3, 3).to(device)
    
    checkpoint = load_training_state(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    best_loss = checkpoint.get('best_loss', 'N/A')
    logger.info(f"Loaded checkpoint from epoch {epoch}, best_loss: {best_loss}")
    
    # ---- Load Data ----
    logger.info("\n--- Loading Dataset ---")
    test_loader, n_test = create_test_dataloader(
        args.dataset,
        batch_size=args.batch,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        pin_memory=True,
        seed=42
    )
    logger.info(f"Test samples: {n_test}")
    
    # ---- Evaluation ----
    logger.info("\n--- Running Evaluation ---")
    
    to_pil = transforms.ToPILImage()
    
    results_keys = [
        'track', 'image', 'plate_layout', 'scenario', 'gt_plate', 'hr_pred', 'lr_pred', 'sr_pred',
        'conf_hr', 'conf_lr', 'conf_sr', 'accuracy_hr', 'accuracy_lr', 'accuracy_sr', 'psnr', 'ssim'
    ]
    
    # Initialize empty CSV with headers
    csv_path = save_dir / 'results.csv'
    pd.DataFrame(columns=results_keys).to_csv(csv_path, index=False)
    
    total_count = 0
    track_buffers = {}
    tracks_saved = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='[Evaluating]', leave=True)
        
        for batch in pbar:
            batch_results = {k: [] for k in results_keys}
            imgs_LR_torch = batch['LR'].to(device)
            imgs_HR_torch = batch['HR'].to(device)
            
            # 1. Generate SR
            imgs_SR_torch = model(imgs_LR_torch)
            
            # 2. Convert batch to numpy for OCR
            list_lr = []
            list_hr = []
            list_sr = []
            
            for j in range(imgs_LR_torch.size(0)):
                list_lr.append(np.array(to_pil(imgs_LR_torch[j].cpu())).astype('uint8'))
                list_hr.append(np.array(to_pil(imgs_HR_torch[j].cpu())).astype('uint8'))
                list_sr.append(np.array(to_pil(imgs_SR_torch[j].cpu())).astype('uint8'))
            
            # 3. Batch OCR predictions
            preds_hr, confs_hr = ocr.predict_plates_batch(list_hr, return_conf=True)
            preds_lr, confs_lr = ocr.predict_plates_batch(list_lr, return_conf=True)
            preds_sr, confs_sr = ocr.predict_plates_batch(list_sr, return_conf=True)
            
            # 4. Record results
            for j in range(imgs_LR_torch.size(0)):
                file_name = batch['file'][j]
                
                # Extract track and image correctly (e.g., from 'track_21832_lr-001.jpg' or '21832_lr-001.jpg')
                import re
                m = re.search(r'([a-zA-Z0-9]+)_lr-\d+', file_name)
                if m:
                    track_name = m.group(1)
                    # if it has 'track_' at the front, let's remove it to keep track_name generic
                    if track_name.startswith('track'):
                        track_name = track_name.replace('track', '')
                    img_name = file_name
                else:
                    parts = file_name.rsplit('_', 1)
                    track_name = parts[0] if len(parts) > 1 else 'unknown'
                    img_name = parts[1] if len(parts) > 1 else file_name
                
                plate_text = batch['plate_text'][j]
                plate_layout = batch['plate_layout'][j]
                scenario = batch['scenario'][j]
                plate_len = max(len(plate_text), 1)
                
                # Save images as track grid
                if args.save_images and tracks_saved < args.save_limit:
                    if track_name not in track_buffers:
                        track_buffers[track_name] = {'lr': [], 'sr': [], 'hr': [], 'img_names': [], 'preds': [], 'confs': [], 'gt': []}
                    
                    track_buffers[track_name]['lr'].append(list_lr[j])
                    track_buffers[track_name]['sr'].append(list_sr[j])
                    track_buffers[track_name]['hr'].append(list_hr[j])
                    track_buffers[track_name]['img_names'].append(img_name)
                    track_buffers[track_name]['preds'].append((preds_lr[j], preds_sr[j], preds_hr[j]))
                    track_buffers[track_name]['confs'].append((confs_lr[j], confs_sr[j], confs_hr[j]))
                    track_buffers[track_name]['gt'].append(plate_text)
                    
                    if len(track_buffers[track_name]['lr']) == 5:
                        _save_track_grid(track_name, track_buffers[track_name], save_dir)
                        del track_buffers[track_name]
                        tracks_saved += 1
                
                # Accuracy
                acc_hr = plate_len - levenshtein(plate_text, preds_hr[j])
                acc_lr = plate_len - levenshtein(plate_text, preds_lr[j])
                acc_sr = plate_len - levenshtein(plate_text, preds_sr[j])
                
                # PSNR / SSIM
                try:
                    psnr_val = calculate_psnr(list_hr[j], list_sr[j])
                    ssim_val = calculate_ssim(list_hr[j], list_sr[j])
                except Exception:
                    psnr_val = 0.0
                    ssim_val = 0.0
                
                # Store results
                batch_results['track'].append(track_name)
                batch_results['image'].append(img_name)
                batch_results['plate_layout'].append(plate_layout)
                batch_results['scenario'].append(scenario)
                batch_results['gt_plate'].append(plate_text)
                batch_results['hr_pred'].append(preds_hr[j])
                batch_results['lr_pred'].append(preds_lr[j])
                batch_results['sr_pred'].append(preds_sr[j])
                batch_results['conf_hr'].append(confs_hr[j])
                batch_results['conf_lr'].append(confs_lr[j])
                batch_results['conf_sr'].append(confs_sr[j])
                batch_results['accuracy_hr'].append(acc_hr)
                batch_results['accuracy_lr'].append(acc_lr)
                batch_results['accuracy_sr'].append(acc_sr)
                batch_results['psnr'].append(psnr_val)
                batch_results['ssim'].append(ssim_val)
                
                total_count += 1
            
            # Save batch results to CSV immediately
            pd.DataFrame(batch_results).to_csv(csv_path, mode='a', header=False, index=False)
            
            pbar.set_postfix({'evaluated': total_count})
    
    # ---- Load Streamed Data for Summary ----
    df = pd.read_csv(csv_path)
    
    # ---- Summary Statistics ----
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total images evaluated: {total_count}")
    
    # Mean PSNR / SSIM
    mean_psnr = df['psnr'].mean()
    mean_ssim = df['ssim'].mean()
    logger.info(f"\nMean PSNR (SR vs HR): {mean_psnr:.2f} dB")
    logger.info(f"Mean SSIM (SR vs HR): {mean_ssim:.4f}")
    
    # Confidence scores
    logger.info(f"\n--- Mean OCR Confidence ---")
    logger.info(f"  HR: {df['conf_hr'].mean()*100:.1f}%")
    logger.info(f"  LR: {df['conf_lr'].mean()*100:.1f}%")
    logger.info(f"  SR: {df['conf_sr'].mean()*100:.1f}%")
    
    # Accuracy distribution
    logger.info(f"\n--- Accuracy Distribution ---")
    for name, col in [('HR', 'accuracy_hr'), ('LR', 'accuracy_lr'), ('SR', 'accuracy_sr')]:
        mean_acc = df[col].mean()
        # Get value counts for accuracy distribution
        acc_dist = df[col].value_counts(normalize=True).sort_index() * 100
        logger.info(f"\n{name} Accuracy (mean: {mean_acc:.2f}):")
        for acc_val, pct in acc_dist.items():
            logger.info(f"  {int(acc_val)} correct: {pct:.1f}%")
    
    # By layout
    logger.info(f"\n--- Results by Layout ---")
    for layout in df['plate_layout'].unique():
        subset = df[df['plate_layout'] == layout]
        logger.info(f"\n{layout} ({len(subset)} images):")
        logger.info(f"  PSNR: {subset['psnr'].mean():.2f} dB")
        logger.info(f"  SSIM: {subset['ssim'].mean():.4f}")
        logger.info(f"  HR Acc: {subset['accuracy_hr'].mean():.2f}")
        logger.info(f"  LR Acc: {subset['accuracy_lr'].mean():.2f}")
        logger.info(f"  SR Acc: {subset['accuracy_sr'].mean():.2f}")
    
    # ---- Detailed CSV with averages ----
    # Results is already streamed to results.csv. We append the average row.
    logger.info(f"\nResults saved to: {csv_path}")
    
    # ---- Save accuracy summary CSVs ----
    for name, col in [('HR', 'accuracy_hr'), ('LR', 'accuracy_lr'), ('SR', 'accuracy_sr')]:
        acc_dist = df[col].value_counts(normalize=True).sort_index() * 100
        acc_path = save_dir / f'accuracy_{name.lower()}.csv'
        acc_dist.to_csv(acc_path, header=['Percentage'])
        logger.info(f"Accuracy dist saved to: {acc_path}")
    
    # ---- Add averages row and save detailed CSV ----
    summary_row = {
        'track': 'AVERAGE',
        'image': '',
        'plate_layout': '',
        'scenario': '',
        'gt_plate': '',
        'hr_pred': '',
        'lr_pred': '',
        'sr_pred': '',
        'conf_hr': df['conf_hr'].mean(),
        'conf_lr': df['conf_lr'].mean(),
        'conf_sr': df['conf_sr'].mean(),
        'accuracy_hr': df['accuracy_hr'].mean(),
        'accuracy_lr': df['accuracy_lr'].mean(),
        'accuracy_sr': df['accuracy_sr'].mean(),
        'psnr': mean_psnr,
        'ssim': mean_ssim,
    }
    df_with_avg = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    df_with_avg.to_csv(save_dir / 'results_detailed.csv', index=False)
    
    # ---- Confusion Matrix ----
    logger.info("\n--- Generating Confusion Matrix ---")
    
    # SR predictions vs GT
    cm_sr, classes = build_confusion_matrix(
        df['gt_plate'].tolist(),
        df['sr_pred'].tolist()
    )
    plot_confusion_matrix(
        cm_sr, classes,
        save_dir / 'confusion_matrix_SR.png',
        title='Confusion Matrix: SR Predictions vs Ground Truth',
        normalize=True
    )
    logger.info(f"SR confusion matrix saved")
    
    # LR predictions vs GT
    cm_lr, _ = build_confusion_matrix(
        df['gt_plate'].tolist(),
        df['lr_pred'].tolist()
    )
    plot_confusion_matrix(
        cm_lr, classes,
        save_dir / 'confusion_matrix_LR.png',
        title='Confusion Matrix: LR Predictions vs Ground Truth',
        normalize=True
    )
    logger.info(f"LR confusion matrix saved")
    
    # ---- Accuracy Histograms ----
    logger.info("\n--- Generating Accuracy Histograms ---")
    _plot_accuracy_histogram(df, save_dir)
    logger.info(f"Histograms saved")
    
    # ---- Confusable Character Analysis ----
    logger.info("\n--- Confusable Character Analysis ---")
    _analyze_confusable_chars(df, logger, save_dir)
    
    logger.info(f"\nEvaluation complete! All outputs saved to: {save_dir}")


# ============================================================================
# Accuracy Histogram
# ============================================================================

def _plot_accuracy_histogram(df: pd.DataFrame, save_dir: Path) -> None:
    """
    Generates a triple-histogram comparing OCR accuracy across HR, LR, and SR.
    
    Color coding:
      - Green: High accuracy (>= 6 chars correct).
      - Yellow: Moderate accuracy (4-5 chars correct).
      - Red: Low accuracy (< 4 chars correct).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (name, col) in zip(axes, [('HR', 'accuracy_hr'), ('LR', 'accuracy_lr'), ('SR', 'accuracy_sr')]):
        # Use the max plate length dynamically (supports 7-10)
        max_chars = int(df[col].max()) + 1 if len(df) > 0 else 8
        max_chars = max(max_chars, 8)  # At least 8 bins (0-7)
        
        x_counts = np.arange(0, max_chars + 1, dtype=int)
        y_counts = np.zeros(max_chars + 1)
        
        for i in range(max_chars + 1):
            y_counts[i] = (df[col] == i).sum()
        
        y_pct = (y_counts / max(len(df), 1)) * 100
        
        bars = ax.bar(x_counts, y_pct, color=['#ff6b6b' if i < 4 else '#51cf66' if i >= 6 else '#ffd43b' for i in x_counts])
        
        for bar, pct in zip(bars, y_pct):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{name} Accuracy Distribution', fontsize=13, fontweight='bold')
        ax.set_xlabel('Correct Characters', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_yticks(range(0, 110, 10))
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('OCR Accuracy Distribution (HR / LR / SR)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Confusable Character Analysis
# ============================================================================

def _analyze_confusable_chars(df: pd.DataFrame, logger: Any, save_dir: Path) -> None:
    """
    Pinpoints specific alphanumeric pairs that the model still struggles with.
    
    Identifies if '8' is being misidentified as 'B', or '0' as 'O' after SR,
    providing actionable insights for loss weighting adjustments.
    """
    from utils import CONFUSABLE_PAIRS
    
    confusable_errors = {}
    total_char_errors = 0
    confusable_char_errors = 0
    
    for _, row in df.iterrows():
        gt = row['gt_plate']
        sr = row['sr_pred']
        
        min_len = min(len(gt), len(sr))
        for i in range(min_len):
            gt_c = gt[i].upper()
            sr_c = sr[i].upper()
            
            if gt_c != sr_c:
                total_char_errors += 1
                pair = f"{gt_c}->{sr_c}"
                
                # Check if it's a confusable pair
                if gt_c in CONFUSABLE_PAIRS and CONFUSABLE_PAIRS[gt_c] == sr_c:
                    confusable_char_errors += 1
                    confusable_errors[pair] = confusable_errors.get(pair, 0) + 1
                elif sr_c in CONFUSABLE_PAIRS and CONFUSABLE_PAIRS[sr_c] == gt_c:
                    confusable_char_errors += 1
                    confusable_errors[pair] = confusable_errors.get(pair, 0) + 1
    
    logger.info(f"Total character errors: {total_char_errors}")
    logger.info(f"Confusable character errors: {confusable_char_errors}")
    
    if total_char_errors > 0:
        pct = (confusable_char_errors / total_char_errors) * 100
        logger.info(f"Confusable error ratio: {pct:.1f}%")
    
    if confusable_errors:
        logger.info("\nTop confusable pair errors (GT->Pred: Count):")
        sorted_errors = sorted(confusable_errors.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_errors[:15]:
            logger.info(f"  {pair}: {count}")
    
    # Save to CSV
    if confusable_errors:
        conf_df = pd.DataFrame([
            {'pair': k, 'count': v} for k, v in sorted(confusable_errors.items(), key=lambda x: x[1], reverse=True)
        ])
        conf_df.to_csv(save_dir / 'confusable_char_errors.csv', index=False)


# ============================================================================
# Track Grid Saving
# ============================================================================

def _save_track_grid(track_name: str, data: Dict[str, Any], save_dir: Path) -> None:
    """
    Generates a 5x3 visual diagnostic grid for a single video track.
    
    Columns:
      - LR (Source): Low-res input.
      - SR (Enhanced): Model output with OCR prediction.
      - HR (Target): Ground truth for comparison.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 12))
    fig.suptitle(f"Track: {track_name}", fontsize=16, fontweight='bold', y=0.98)
    
    # Sort data by image name to ensure order (lr-001 -> lr-005)
    sorted_indices = np.argsort(data['img_names'])
    
    for row_idx, sorted_idx in enumerate(sorted_indices):
        img_name = data['img_names'][sorted_idx]
        lr = data['lr'][sorted_idx]
        sr = data['sr'][sorted_idx]
        hr = data['hr'][sorted_idx]
        pred_lr, pred_sr, pred_hr = data['preds'][sorted_idx]
        conf_lr, conf_sr, conf_hr = data['confs'][sorted_idx]
        gt = data['gt'][sorted_idx]
        
        # Plot LR
        axes[row_idx, 0].imshow(lr)
        axes[row_idx, 0].set_title(f"{img_name} (LR)\nPred: {pred_lr} ({conf_lr*100:.1f}%) | GT: {gt}", fontsize=10, color='red' if pred_lr != gt else 'green')
        axes[row_idx, 0].axis('off')
        
        # Plot SR
        axes[row_idx, 1].imshow(sr)
        axes[row_idx, 1].set_title(f"SR\nPred: {pred_sr} ({conf_sr*100:.1f}%) | GT: {gt}", fontsize=10, color='red' if pred_sr != gt else 'green')
        axes[row_idx, 1].axis('off')
        
        # Plot HR
        axes[row_idx, 2].imshow(hr)
        axes[row_idx, 2].set_title(f"HR\nPred: {pred_hr} ({conf_hr*100:.1f}%) | GT: {gt}", fontsize=10, color='red' if pred_hr != gt else 'green')
        axes[row_idx, 2].axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_dir / f"grid_{track_name}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LP Super-Resolution Evaluation')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to ICPR2026 train/ directory')
    parser.add_argument('--save', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--ocr_path', type=str, required=True,
                        help='Path to Keras OCR model directory')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train/val split ratio to get test set (default: 0.7)')
    parser.add_argument('--save_images', action='store_true', default=False,
                        help='Save HR/LR/SR images to output directory')
    parser.add_argument('--save_limit', type=int, default=100,
                        help='Maximum number of images to save if --save_images is enabled (default: 100)')
    
    args = parser.parse_args()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test(args)
