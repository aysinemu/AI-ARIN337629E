"""
train.py - Training pipeline for Improved LP Super-Resolution.

This script implements a multi-task learning pipeline for license plate super-resolution.
It combines pixel-level reconstruction with high-level perceptual and semantic losses
to ensure that the upscaled images are not only visually sharp but also machine-readable.

Key Loss Components:
  - Pixel loss: MSE + L1 for base structural accuracy.
  - Perceptual loss: VGG19 feature-space distance for realistic textures.
  - OCR-guided loss: Leverages a pretrained Keras OCR "Teacher" to guide character clarity.
  - Latent correlation loss: Aligns the latent feature maps of SR and HR images.
  - Consistency loss: Ensures downsample(SR) matches the original LR input.
  - Total Variation (TV) loss: Reduces high-frequency noise artifacts.

Usage:
    python train.py --dataset /path/to/data --save /path/to/save --ocr_path /path/to/ocr
"""

from typing import List, Dict, Tuple, Any, Optional, Union
import os
import re
import sys
import cv2
import time
import json
import random
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm
from torch.autograd import Variable

from network import ImprovedNetwork, VGGFeatureExtractor, icnr_init  
from dataset import create_dataloaders
from utils import (
    setup_logging, padding, levenshtein, get_char_weights,
    calculate_psnr, calculate_ssim,
    plot_losses, plot_metrics,
    save_training_state, load_training_state,
    save_comparison_grid,
    CONFUSABLE_WEIGHT
)


# ============================================================================
# Suppress TF warnings (for Keras OCR)
# ============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ============================================================================
# OCR Module (Keras, same as original paper)
# ============================================================================

class OCRModule:
    """
    Keras-based OCR module used as a 'Teacher' to guide the Super-Resolution training.
    
    This module bridges the gap between PyTorch (training) and Keras (OCR). It provides
    character-level predictions and intermediate feature extraction to form the 
    semantic loss components of the SR network.
    
    Attributes:
        ocr_path (Path): Filesystem path to the OCR model directory.
        logger (Optional[logging.Logger]): Logger instance for status reporting.
        device (str): Computation device for the OCR (usually matched to PyTorch).
        OCR (Optional[tf.keras.Model]): The loaded Keras OCR model.
        feature_model (Optional[tf.keras.Model]): Sub-model for intermediate feature extraction.
    """
    
    def __init__(self, ocr_path: Union[str, Path], logger: Optional[Any] = None, device: str = 'cpu') -> None:
        """
        Initializes the OCR Module and attempts to load the Keras model.
        
        Args:
            ocr_path (Union[str, Path]): Path to directory containing model.json and weights.
            logger (Optional[Any], optional): Logger instance. Defaults to None.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.ocr_path = Path(ocr_path)
        self.to_pil = transforms.ToPILImage()
        self.logger = logger
        self.OCR = None
        self.feature_model = None
        
        if self.ocr_path.exists() and self.ocr_path.is_dir():
            self._load_ocr()
        else:
            msg = f"[WARNING] OCR model directory not found at {ocr_path}"
            if self.logger: self.logger.warning(msg)
            else: print(msg)
    
    def _log(self, msg: str, level: str = 'info') -> None:
        """
        Internal logging helper that respects the provided logger instance.
        
        Args:
            msg (str): Message to log.
            level (str, optional): Logging level ('info', 'warning', 'error'). Defaults to 'info'.
        """
        if self.logger:
            if level == 'info': self.logger.info(msg)
            elif level == 'warning': self.logger.warning(msg)
            elif level == 'error': self.logger.error(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def _get_img_to_array(self) -> Any:
        """
        Lazy-loads the Keras image-to-array utility to minimize initial startup overhead.
        
        Returns:
            Any: The Keras `img_to_array` function reference.
        """
        from tensorflow.keras.preprocessing.image import img_to_array
        return img_to_array

    def _load_ocr(self) -> None:
        """
        Loads the Keras structure and weights, and configures GPU memory growth.
        
        This method handles the complex task of initializing TensorFlow alongside PyTorch.
        It enforces 'Memory Growth' to prevent TF from monopolizing all VRAM, which 
        would otherwise crash the PyTorch training process.
        """
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    self._log(f"TF Memory growth error: {e}", 'warning')
            
            # Use tf.keras for consistency in newer TensorFlow versions
            from tensorflow.keras.models import model_from_json, Model
            import tensorflow.keras.layers as layers
            
            json_path = self.ocr_path / 'model.json'
            weights_path_v3 = self.ocr_path / 'weights_improved.weights.h5'
            weights_path_legacy = self.ocr_path / 'weights.hdf5'
            params_npy = self.ocr_path / 'parameters.npy'
            
            if not json_path.exists():
                self._log(f"Missing OCR files in {self.ocr_path}. Need model.json", 'error')
                return
            
            if weights_path_v3.exists():
                weights_path = weights_path_v3
                self._log("Using fine-tuned Keras 3 weights (weights_improved.weights.h5)")
            elif weights_path_legacy.exists():
                weights_path = weights_path_legacy
                self._log("Using legacy Keras weights (weights.hdf5)")
            else:
                self._log(f"Missing OCR weights in {self.ocr_path}", 'error')
                return

            # Load full OCR model
            with open(str(json_path), 'r') as f:
                model_json = f.read()
            
            # Define custom objects to help model_from_json locate classes
            # This is often needed when loading models across different Keras/TF versions
            custom_objects = {
                'Model': Model,
                'InputLayer': layers.InputLayer,
                'Conv2D': layers.Conv2D,
                'BatchNormalization': layers.BatchNormalization,
                'Activation': layers.Activation,
                'MaxPooling2D': layers.MaxPooling2D,
                'Flatten': layers.Flatten,
                'Dense': layers.Dense,
                'Dropout': layers.Dropout
            }
            
            try:
                self.OCR = model_from_json(model_json, custom_objects=custom_objects)
            except Exception as e:
                self._log(f"First attempt to load model_from_json failed: {e}. Trying without custom_objects...", 'warning')
                self.OCR = model_from_json(model_json)
                
            self.OCR.load_weights(str(weights_path))
            
            # Load parameters
            if params_npy.exists():
                self.parameters = np.load(str(params_npy), allow_pickle=True).item()
            else:
                self._log(f"Missing parameters.npy in {self.ocr_path}", 'error')
                return
            
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            
            # 1. Get input dimensions properly (Confirmed: (None, 60, 120, 3))
            in_shape = self.OCR.input_shape
            if isinstance(in_shape, list): in_shape = in_shape[0]
            self.IMAGE_DIMS = in_shape[1:]
            
            self.aspect_ratio = self.IMAGE_DIMS[1] / self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            
            # 2. Proper Feature extractor (Using 'activation_6' - validated from layer scan)
            try:
                target_layer = self.OCR.get_layer('activation_6')
                self.feature_model = Model(self.OCR.input, target_layer.output)
            except Exception as e:
                self._log(f"Proper feature extraction by name failed: {e}. Trying fallback search...", 'warning')
                # Fallback: find the last Activation or Conv2D layer before flatten
                target_layer = None
                for layer in reversed(self.OCR.layers):
                    if any(x in layer.__class__.__name__ for x in ['Activation', 'Conv2D', 'MaxPooling2D']):
                        if 'Flatten' not in [l.__class__.__name__ for l in self.OCR.layers[self.OCR.layers.index(layer):]]:
                            target_layer = layer
                            break
                if target_layer:
                    self.feature_model = Model(self.OCR.input, target_layer.output)
            
            self.OCR.compile()
            if self.feature_model:
                try:
                    self.feature_model.compile()
                    # Debug: print feature shape
                    dummy = np.zeros((1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2]))
                    feat_shape = self.feature_model.predict(dummy, verbose=0).shape
                    self._log(f"OCR Feature model output shape: {feat_shape}")
                except Exception as e:
                    self._log(f"Feature shape debug failed: {e}", 'warning')
            
            self._log(f"OCR model loaded properly from {self.ocr_path}")
            self._log(f"OCR Input Dims: {self.IMAGE_DIMS}")
            
        except Exception as e:
            self._log(f"Failed to load OCR model: {e}", 'error')
            self.OCR = None
    
    def predict_plate(self, img: np.ndarray, convert_to_bgr: bool = True) -> str:
        """
        Predicts the alphanumeric text of a single license plate image.
        
        Args:
            img (np.ndarray): The input image as a numpy array.
            convert_to_bgr (bool, optional): Whether to convert RGB to BGR before OCR. Defaults to True.
        
        Returns:
            str: The predicted plate text (e.g., "ABC1234").
        """
        texts, confs = self.predict_plates_batch([img], return_conf=True, convert_to_bgr=convert_to_bgr)
        return texts[0]
    
    def predict_plates_batch(self, inputs: Union[torch.Tensor, List[np.ndarray]], 
                             return_conf: bool = False, 
                             convert_to_bgr: bool = True) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Predicts plate texts for a batch of images using high-performance vectorized operations.
        
        This method is 'Consolidated': it intelligently handles both raw PyTorch Tensors 
        (used during training) and lists of Numpy images (used during evaluation/testing).
        
        Args:
            inputs (Union[torch.Tensor, List[np.ndarray]]): Batch of images.
            return_conf (bool, optional): If True, also returns confidence scores. Defaults to False.
            convert_to_bgr (bool, optional): Whether to ensure BGR color space. Defaults to True.
        
        Returns:
            Union[List[str], Tuple[List[str], List[float]]]: Predicted texts or (texts, confidences).
        """
        if self.OCR is None or inputs is None:
            batch_size = inputs.size(0) if hasattr(inputs, 'size') else len(inputs)
            if return_conf: return [""] * batch_size, [0.0] * batch_size
            return [""] * batch_size

        import torch
        img_to_array = self._get_img_to_array()
        target_h, target_w = self.IMAGE_DIMS[0], self.IMAGE_DIMS[1]

        if torch.is_tensor(inputs):
            # 1. PyTorch Tensor (B, 3, H, W) -> GPU Resize
            import torch.nn.functional as F_torch
            # Resize
            tensors_resized = F_torch.interpolate(inputs, size=(target_h, target_w),
                                                  mode='bilinear', align_corners=False)
            # Convert to Numpy batch (B, H, W, 3)
            batch_np = tensors_resized.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            # 2. List of Numpy images -> CPU Resize
            batch_np = np.zeros((len(inputs), target_h, target_w, 3), dtype='float32')
            for i, img in enumerate(inputs):
                if convert_to_bgr:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_p, _, _ = padding(img, self.min_ratio, self.max_ratio, color=(127, 127, 127))
                img_p = cv2.resize(img_p, (target_w, target_h)) # cv2 is (W, H)
                batch_np[i] = img_to_array(img_p) / 255.0

        # Keras Predict
        raw_predictions = self.OCR.predict(batch_np, verbose=0)

        # Decode results
        num_samples = batch_np.shape[0]
        results_text = [""] * num_samples
        results_conf = [0.0] * num_samples
        
        char_task_indices = []
        for task_idx, task in self.tasks.items():
            task_str = task.decode('utf-8') if isinstance(task, bytes) else str(task)
            if re.match(r'^char[1-9]\d*$', task_str):
                char_task_indices.append(task_idx)

        
        for i in range(num_samples):
            plate = ""
            conf_sum = 0.0
            for task_idx in char_task_indices:
                pp = raw_predictions[task_idx][i]
                best_idx = np.argmax(pp)
                plate += self.ocr_classes[f'char{task_idx + 1}'][best_idx]
                conf_sum += pp[best_idx]
            
            results_text[i] = plate
            results_conf[i] = conf_sum / len(char_task_indices) if char_task_indices else 0.0
            
        if return_conf:
            return results_text, results_conf
        return results_text
    
    def extract_features(self, img: np.ndarray, convert_to_bgr: bool = True) -> Optional[np.ndarray]:
        """
        Extracts intermediate semantic feature maps from the OCR model's hidden layers.
        
        Args:
            img (np.ndarray): Input RGB image.
            convert_to_bgr (bool, optional): Whether to convert to BGR. Defaults to True.
        
        Returns:
            Optional[np.ndarray]: Vectorized feature map or None if OCR is not loaded.
        """
        if self.OCR is None or self.feature_model is None:
            return None
        
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        img_p, _, _ = padding(img, self.min_ratio, self.max_ratio, color=(127, 127, 127))
        img_p = cv2.resize(img_p, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))
        
        from keras.preprocessing.image import img_to_array
        img_arr = img_to_array(img_p)
        img_arr = img_arr.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        img_arr = (img_arr / 255.0).astype('float')
        
        features = self.feature_model.predict(img_arr, verbose=0)
        return features
    
    def batch_predict(self, tensor_batch: torch.Tensor) -> List[str]:
        """
        A compatibility redirect for legacy code calling 'batch_predict'.
        
        Args:
            tensor_batch (torch.Tensor): PyTorch tensor batch.
            
        Returns:
            List[str]: Predicted texts.
        """
        return self.predict_plates_batch(tensor_batch, return_conf=False) # type: ignore


# ============================================================================
# Loss Functions
# ============================================================================

class CombinedLoss(nn.Module):
    """
    The master loss function coordinating multiple semantic and structural objectives.
    
    This class implements the 'Semantic-Guided' loss strategy where various weights
    balance architectural fidelity (Pixel/SSIM) against readability (OCR/Latent).
    
    Formula: Total = α*Pixel + β*Perceptual + γ*OCR + δ*Latent + ζ*Consistency + η*TV
    """
    
    def __init__(self, ocr_module: Optional[OCRModule] = None,
                 alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.01, 
                 delta: float = 1.0, zeta: float = 0.25, eta: float = 0.25,
                 device: str = 'cuda:0') -> None:
        """
        Initializes loss components and weights.
        
        Args:
            ocr_module (Optional[OCRModule], optional): Preloaded OCR module for semantic loss.
            alpha (float): Pixel Loss weight.
            beta (float): Perceptual Loss (VGG) weight.
            gamma (float): OCR Loss weight.
            delta (float): Latent Space Correlation weight.
            zeta (float): Consistency Loss weight.
            eta (float): Total Variation (TV) weight.
            device (str): Computation device.
        """
        super(CombinedLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.eta = eta
        self.device = device
        self.ocr_step_freq = 4  # Calculate OCR loss every 4 steps to boost speed
        self._step_counter = 0
        
        # Pixel losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Perceptual loss (VGG19)
        self.vgg = VGGFeatureExtractor().to(device)
        self.vgg.eval()
        
        # OCR module
        self.ocr = ocr_module
        
        # To numpy converter
        self.to_pil = transforms.ToPILImage()
        
        # Debug flag for printing shapes once
        self._debug_once = True
    
    def compute_pixel_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Computes the structural fidelity loss (MSE + L1 averaged).
        
        Args:
            sr (torch.Tensor): Super-resolved batch.
            hr (torch.Tensor): High-resolution batch.
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        return (self.mse_loss(sr, hr) + self.l1_loss(sr, hr)) / 2.0
    
    def compute_perceptual_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Computes the perceptual feature similarity using a frozen VGG19 backbone.
        
        Args:
            sr (torch.Tensor): Super-resolved batch.
            hr (torch.Tensor): High-resolution batch.
            
        Returns:
            torch.Tensor: L1 distance in VGG feature space.
        """
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return self.l1_loss(sr_features, hr_features)
    
    def compute_ocr_loss(self, sr: torch.Tensor, hr: torch.Tensor, gt_plates: List[str]) -> torch.Tensor:
        """
        Computes the semantic OCR loss with weighted character-level penalties.
        
        This method is gated by `self.ocr_step_freq` to ensure and boost GPU throughput, 
        as running OCR on every training step is computationally prohibitive.
        
        Args:
            sr (torch.Tensor): Super-resolved batch.
            hr (torch.Tensor): High-resolution ground truth batch.
            gt_plates (List[str]): Ground truth string texts.
            
        Returns:
            torch.Tensor: Weighted Levenshtein + Character distance.
        """
        self._step_counter += 1
        if self._step_counter % self.ocr_step_freq != 0:
            # Skip OCR loss for this step to keep GPU moving fast
            return torch.tensor(0.0, device=self.device)
            
        if self.ocr is None or self.ocr.OCR is None:
            return torch.tensor(0.0, device=self.device)
        
        # Batch inference for both SR and HR
        pred_sr_list = self.ocr.predict_plates_batch(sr)
        pred_hr_list = self.ocr.predict_plates_batch(hr)
        
        batch_loss = 0.0
        
        for pred_sr, pred_hr, gt_text in zip(pred_sr_list, pred_hr_list, gt_plates):
            reliability = 1.0
            if pred_hr.upper() != gt_text.upper():
                reliability = 0.2
            
            if len(gt_text) == 0:
                continue
                
            plate_len = max(len(gt_text), 1)
            lev_dist = levenshtein(gt_text, pred_sr)
            lev_loss = lev_dist / plate_len
            
            # Simplified weighted logic for speed
            char_weights = get_char_weights(gt_text, pred_sr, num_chars=len(gt_text))
            weighted_errors = 0.0
            min_len = min(len(gt_text), len(pred_sr))
            for i in range(min_len):
                if gt_text[i].upper() != pred_sr[i].upper():
                    weighted_errors += char_weights[i] if i < len(char_weights) else 1.0
            
            if len(gt_text) != len(pred_sr):
                weighted_errors += abs(len(gt_text) - len(pred_sr)) * CONFUSABLE_WEIGHT
                
            weighted_loss = weighted_errors / plate_len
            batch_loss += ((lev_loss + weighted_loss) / 2.0) * reliability
            
        return torch.tensor(batch_loss / max(len(gt_plates), 1), 
                            dtype=torch.float32, device=self.device)
    
    def compute_latent_loss(self, latent_sr: Optional[torch.Tensor], latent_hr: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Computes the cosine distance between the hidden latent maps of SR and HR pipelines.
        
        Args:
            latent_sr (Optional[torch.Tensor]): SR latent features.
            latent_hr (Optional[torch.Tensor]): HR latent features.
            
        Returns:
            torch.Tensor: (1.0 - CosineSimilarity) mean.
        """
        if latent_sr is None or latent_hr is None:
            return torch.tensor(0.0, device=self.device)
        
        cos_sim = F.cosine_similarity(latent_sr, latent_hr, dim=1)
        return (1.0 - cos_sim).mean()
    
    def compute_consistency_loss(self, sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        """
        Implements self-supervised 'Belief' consistency.
        
        Ensures that if we downsample our newly generated SR image, it looks
        mathematically identical to the low-resolution image that produced it.
        
        Args:
            sr (torch.Tensor): SR batch.
            lr (torch.Tensor): Original LR input batch.
            
        Returns:
            torch.Tensor: Downsample-reconstruction MSE.
        """
        # Downsample SR back to LR size
        # Assuming 4x factor based on current model config
        sr_down = F.interpolate(sr, size=(lr.size(2), lr.size(3)), mode='bicubic', align_corners=False)
        return self.mse_loss(sr_down, lr)

    def compute_tv_loss(self, img: torch.Tensor) -> torch.Tensor:
        """
        Computes Total Variation (TV) loss to suppress high-frequency noise and grains.
        
        Args:
            img (torch.Tensor): Input batch.
            
        Returns:
            torch.Tensor: TV penalty.
        """
        batch_size = img.size(0)
        h_x = img.size(2)
        w_x = img.size(3)
        count_h = self._tensor_size(img[:, :, 1:, :])
        count_w = self._tensor_size(img[:, :, :, 1:])
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w_x - 1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t: torch.Tensor) -> int:
        """Utility for calculating the total element count of a tensor window."""
        return t.size(1) * t.size(2) * t.size(3)
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor, lr: Optional[torch.Tensor] = None, 
                gt_plates: Optional[List[str]] = None, 
                latent_sr: Optional[torch.Tensor] = None, 
                latent_hr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Passes inputs through all loss components and sums them according to hyperparameter weights.
        
        Args:
            sr (torch.Tensor): Predicted SR batch.
            hr (torch.Tensor): Target HR batch.
            lr (Optional[torch.Tensor], optional): Original LR input.
            gt_plates (Optional[List[str]], optional): Target alphanumeric labels.
            latent_sr (Optional[torch.Tensor], optional): SR feature vector.
            latent_hr (Optional[torch.Tensor], optional): HR feature vector.
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: The weighted scalar loss and a breakdown dictionary.
        """
        if self._debug_once:
            print(f"\n{'='*20} [DEBUG SHAPES] {'='*20}")
            print(f"  SR Image:     {sr.shape} (Range: {sr.min():.2f} to {sr.max():.2f})")
            print(f"  HR Image:     {hr.shape} (Range: {hr.min():.2f} to {hr.max():.2f})")
            if latent_sr is not None: 
                print(f"  Latent SR:    {latent_sr.shape}")
            if latent_hr is not None: 
                print(f"  Latent HR:    {latent_hr.shape}")
            if gt_plates is not None: 
                print(f"  GT Plates:    {len(gt_plates)} samples, Sample: '{gt_plates[0]}'")
            
            # Verify OCR if active
            if self.ocr and self.ocr.OCR:
                try:
                    sr_np = np.array(self.to_pil(sr[0].detach().cpu())).astype('uint8')
                    hr_np = np.array(self.to_pil(hr[0].detach().cpu())).astype('uint8')
                    pred_sr = self.ocr.predict_plate(sr_np)
                    pred_hr = self.ocr.predict_plate(hr_np)
                    rel = "GOOD" if pred_hr.upper() == gt_plates[0].upper() else "BAD (FALLBACK ACTIVE)"
                    print(f"  OCR Test (SR[0]): '{pred_sr}' vs GT: '{gt_plates[0]}'")
                    print(f"  Teacher Read (HR): '{pred_hr}' -> Reliability: {rel}")
                except Exception as e:
                    print(f"  OCR Test failed: {e}")
            print(f"{'='*56}\n")
            self._debug_once = False
            
        # 1. Pixel loss
        pixel_loss = self.compute_pixel_loss(sr, hr)
        
        # 2. Perceptual loss
        perceptual_loss = self.compute_perceptual_loss(sr, hr)
        
        # 3. OCR loss (with reliability weighting)
        if gt_plates is not None and self.gamma > 0:
            ocr_loss = self.compute_ocr_loss(sr, hr, gt_plates)
        else:
            ocr_loss = torch.tensor(0.0, device=self.device)
        
        # 4. Latent loss
        latent_loss = self.compute_latent_loss(latent_sr, latent_hr)
        
        # 5. Consistency loss (Self-Belief)
        if lr is not None and self.zeta > 0:
            consistency_loss = self.compute_consistency_loss(sr, lr)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
        
        # 6. TV loss (Anti-Noise)
        if self.eta > 0:
            tv_loss = self.compute_tv_loss(sr)
        else:
            tv_loss = torch.tensor(0.0, device=self.device)
        
        # Combine
        total = (self.alpha * pixel_loss +
                 self.beta * perceptual_loss +
                 self.gamma * ocr_loss +
                 self.delta * latent_loss +
                 self.zeta * consistency_loss +
                 self.eta * tv_loss)
        
        loss_dict = {
            'total': total.item(),
            'pixel': pixel_loss.item(),
            'perceptual': perceptual_loss.item(),
            'ocr': ocr_loss.item(),
            'latent': latent_loss.item(),
            'consist': consistency_loss.item(),
            'tv': tv_loss.item(),
        }
        
        return total, loss_dict


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Monitors validation loss and terminates training if improvement plateaus.
    
    This prevents overfitting and saves the 'Best' iteration of weights automatically.
    
    Attributes:
        patience (int): Number of epochs to wait before killing the process.
        min_delta (float): Required minimum improvement to reset the counter.
        counter (int): Current hunger counter.
        best_loss (Optional[float]): Best loss achieved in history.
        early_stop (bool): Flag indicating stop criteria met.
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0001, best_loss: Optional[float] = None) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = best_loss
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module, save_dir: Path, epoch: int, 
                 optimizer: torch.optim.Optimizer, history: dict, logger: Optional[Any] = None) -> None:
        """
        Evaluates the current epoch's validation loss against historical bests.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_best(model, save_dir, epoch, optimizer, history)
            if logger:
                logger.info(f"[EarlyStopping] Initial best loss: {val_loss:.6f}")
            return
        
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_best(model, save_dir, epoch, optimizer, history)
            if logger:
                logger.info(f"[EarlyStopping] New best loss: {val_loss:.6f}")
        else:
            self.counter += 1
            if logger:
                logger.info(f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if logger:
                    logger.warning("[EarlyStopping] TRIGGERED! Stopping training.")
    
    def _save_best(self, model, save_dir, epoch, optimizer, history):
        save_training_state({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_loss': self.best_loss,
        }, save_dir, 'bestmodel.pt')


# ============================================================================
# Training & Validation Loops
# ============================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                    criterion: CombinedLoss, device: torch.device, logger: logging.Logger, 
                    epoch: int) -> Tuple[float, Dict[str, float]]:
    """
    Executes a single training pass over the dynamic data subset.
    
    This function handles the forward pass, loss computation, backpropagation, 
    and gradient clipping. It aggregates various loss components for reporting.
    
    Args:
        model (nn.Module): The SR network.
        dataloader (DataLoader): Training data provider.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        criterion (CombinedLoss): Multi-task loss calculator.
        device (torch.device): CPU/GPU device.
        logger (logging.Logger): Logger for progress tracking.
        epoch (int): Current epoch number.
        
    Returns:
        Tuple[float, Dict[str, float]]: Average total loss and average component losses.
    """
    model.train()
    running_loss = 0.0
    running_losses = {}
    steps = 0
    
    pbar = tqdm(dataloader, desc=f'[Train E{epoch}]', leave=True)
    
    for batch in pbar:
        imgs_LR = batch['LR'].to(device)
        imgs_HR = batch['HR'].to(device)
        gt_plates = batch['plate_text']  # List of strings
        
        optimizer.zero_grad()
        
        # Forward pass (training mode: returns sr, latent_sr, latent_hr)
        sr_output, latent_sr, latent_hr = model(imgs_LR, imgs_HR)
        
        # Range Monitoring (First Batch of First Epoch Only)
        if epoch == 1 and steps == 0:
            logger.info(f"--- [DEBUG RANGE] SR Output (pre-clamp) min: {sr_output.min():.4f}, max: {sr_output.max():.4f}, mean: {sr_output.mean():.4f}")
            logger.info(f"--- [DEBUG RANGE] LR Input min: {imgs_LR.min():.4f}, max: {imgs_LR.max():.4f}, mean: {imgs_LR.mean():.4f}")

        # Compute loss
        loss, loss_dict = criterion(
            sr_output, imgs_HR,
            lr=imgs_LR,
            gt_plates=gt_plates,
            latent_sr=latent_sr,
            latent_hr=latent_hr
        )
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        steps += 1
        
        # Accumulate component losses
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v
        
        pbar.set_postfix({
            'L': f'{running_loss/steps:.4f}',
            'pix': f'{running_losses.get("pixel", 0)/steps:.4f}',
            'vgg': f'{running_losses.get("perceptual", 0)/steps:.3f}',
            'ocr': f'{running_losses.get("ocr", 0)/steps:.3f}',
            'lat': f'{running_losses.get("latent", 0)/steps:.5f}',
            'con': f'{running_losses.get("consist", 0)/steps:.5f}',
        })
        
        # Periodic explicit log entry for history
        if steps % 100 == 0:
            logger.info(
                f"[Batch {steps}/{len(dataloader)}] "
                f"Loss: {running_loss/steps:.4f} | "
                f"P: {running_losses.get('pixel', 0)/steps:.4f} | "
                f"V: {running_losses.get('perceptual', 0)/steps:.4f} | "
                f"O: {running_losses.get('ocr', 0)/steps:.3f} | "
                f"Lat: {running_losses.get('latent', 0)/steps:.6f} | "
                f"C: {running_losses.get('consist', 0)/steps:.5f}"
            )

    avg_loss = running_loss / max(steps, 1)
    avg_losses = {k: v / max(steps, 1) for k, v in running_losses.items()}
    
    return avg_loss, avg_losses


def validate(model: nn.Module, dataloader: DataLoader, criterion: CombinedLoss, 
             device: torch.device, logger: logging.Logger, epoch: int, 
             ocr_module: Optional[OCRModule] = None) -> Tuple[float, Dict[str, float], float, float]:
    """
    Evaluates the model on the validation split and calculates image quality metrics.
    
    Args:
        model (nn.Module): The SR network.
        dataloader (DataLoader): Validation data provider.
        criterion (CombinedLoss): Multi-task loss calculator.
        device (torch.device): Computation device.
        logger (logging.Logger): Logger instance.
        epoch (int): Current epoch number.
        ocr_module (Optional[OCRModule], optional): OCR for visual grid prediction. Defaults to None.
        
    Returns:
        Tuple[float, Dict[str, float], float, float]: 
            Average validation loss, breakdown dict, average PSNR, and average SSIM.
    """
    model.eval()
    running_loss = 0.0
    running_losses = {}
    steps = 0
    
    psnr_values = []
    ssim_values = []
    
    to_pil = transforms.ToPILImage()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'[Val   E{epoch}]', leave=True)
        
        for batch in pbar:
            imgs_LR = batch['LR'].to(device)
            imgs_HR = batch['HR'].to(device)
            gt_plates = batch['plate_text']
            
            # Forward pass (inference mode: no HR passed)
            sr_output = model(imgs_LR)
            
            # Compute loss
            loss, loss_dict = criterion(
                sr_output, imgs_HR,
                lr=imgs_LR,
                gt_plates=gt_plates,
                latent_sr=None,
                latent_hr=None
            )
            
            running_loss += loss.item()
            steps += 1
            
            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + v
            
            # Compute PSNR / SSIM for each image in batch
            for j in range(sr_output.size(0)):
                sr_np = np.array(to_pil(sr_output[j].cpu())).astype('uint8')
                hr_np = np.array(to_pil(imgs_HR[j].cpu())).astype('uint8')
                
                try:
                    psnr_values.append(calculate_psnr(hr_np, sr_np))
                    ssim_values.append(calculate_ssim(hr_np, sr_np))
                except Exception:
                    pass
            
            pbar.set_postfix({
                'loss': f'{running_loss/steps:.4f}',
                'psnr': f'{np.mean(psnr_values):.2f}' if psnr_values else 'N/A',
                'ssim': f'{np.mean(ssim_values):.4f}' if ssim_values else 'N/A',
            })
    
    avg_loss = running_loss / max(steps, 1)
    avg_losses = {k: v / max(steps, 1) for k, v in running_losses.items()}
    avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
    
    # ---- Save comparison images ----
    try:
        visuals_dir = Path(logger.handlers[1].baseFilename).parent / "visuals" # type: ignore
        visuals_dir.mkdir(parents=True, exist_ok=True)
        
        save_comparison_grid(
            imgs_LR, sr_output, imgs_HR, gt_plates, ocr_module,
            visuals_dir / f"epoch_{epoch}_comparison.png",
            num_samples=4
        )
    except Exception as e:
        logger.warning(f"Failed to save comparison image: {e}")
    
    return avg_loss, avg_losses, avg_psnr, avg_ssim


def main() -> None:
    """
    Main entry point for training the License Plate Super-Resolution pipeline.
    
    This function:
      1. Parses command line arguments.
      2. Initializes data loaders, model, and loss criterion.
      3. Manages the training loop, validation frequency, and checkpointing.
      4. Handles automatic ICNR resetting for checkerboard artifact removal.
    """
    parser = argparse.ArgumentParser(description='Improved LP Super-Resolution Training')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to ICPR2026 train/ directory')
    parser.add_argument('--save', type=str, required=True,
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Max epochs (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt to resume from')
    parser.add_argument('--ocr_path', type=str, default=None,
                        help='Path to Keras OCR model directory')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train/val split ratio')
    parser.add_argument('--data_fraction', type=float, default=0.1,
                        help='Fraction of training data to use each epoch')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Frequency of validation in epochs')

    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Pixel loss weight')
    parser.add_argument('--beta', type=float, default=0.75,
                        help='Perceptual loss weight')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='OCR loss weight')
    parser.add_argument('--delta', type=float, default=1,
                        help='Latent correlation loss weight')
    parser.add_argument('--zeta', type=float, default=0.25,
                        help='Downsampling consistency loss weight')
    parser.add_argument('--eta', type=float, default=0.25,
                        help='TV loss weight')
    parser.add_argument('--force_icnr', action='store_true',
                        help='Force-apply ICNR to upsampling layers on start')
    
    args = parser.parse_args()
    
    # ---- Setup ----
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(save_dir, name='training')
    
    logger.info("=" * 60)
    logger.info("Improved LP Super-Resolution Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Loss weights: α={args.alpha}, β={args.beta}, γ={args.gamma}, δ={args.delta}, ζ={args.zeta}, η={args.eta}")
    
    # ---- Data ----
    logger.info("\n--- Loading Dataset ---")
    train_loader, val_loader, n_train, n_val = create_dataloaders(
        args.dataset, batch_size=args.batch, train_ratio=args.train_ratio,
        num_workers=args.workers, pin_memory=True, seed=42
    )
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # ---- Model ----
    logger.info("\n--- Building Model ---")
    model = ImprovedNetwork(3, 3).to(device)
    
    # ---- OCR Module ----
    ocr_module = None
    if args.ocr_path:
        logger.info(f"\n--- Loading OCR Model ---")
        ocr_module = OCRModule(args.ocr_path, logger=logger, device=str(device))
    
    # ---- Loss & Optimizer ----
    criterion = CombinedLoss(
        ocr_module=ocr_module, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        delta=args.delta, zeta=args.zeta, eta=args.eta, device=str(device)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)
    
    # ---- Training History ----
    history = {
        'train_loss': [], 'val_loss': [], 'train_losses_detail': [],
        'val_losses_detail': [], 'psnr': [], 'ssim': [], 'lr': [],
    }
    
    start_epoch = 0
    early_stopping = EarlyStopping(patience=args.patience)
    
    # ---- Resume from checkpoint ----
    if args.resume:
        logger.info(f"\n--- Resuming from: {args.resume} ---")
        checkpoint = load_training_state(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint.get('history', history)
        early_stopping.best_loss = checkpoint.get('best_loss', None)
        
        if args.force_icnr:
            logger.info("### [FLAG DETECTED] Force-applying ICNR initialization...")
            for m in model.modules():
                if isinstance(m, torch.nn.Conv2d) and m.out_channels == m.in_channels * 4:
                    icnr_init(m.weight, upscale_factor=2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
    
    # ---- Training Loop ----
    vis_subset_idx = np.random.choice(len(val_dataset), min(100, len(val_dataset)), replace=False)
    vis_loader = DataLoader(data.Subset(val_dataset, vis_subset_idx), batch_size=4, shuffle=True)
    
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        num_train = len(train_dataset)
        subset_idx = np.random.choice(num_train, int(num_train * args.data_fraction), replace=False)
        epoch_train_loader = DataLoader(data.Subset(train_dataset, subset_idx), 
                                        batch_size=args.batch, shuffle=True, num_workers=args.workers)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Samples: {len(subset_idx)}")
        
        train_loss, train_detail = train_one_epoch(model, epoch_train_loader, optimizer, criterion, device, logger, epoch+1)
        
        history['train_loss'].append(train_loss)
        history['train_losses_detail'].append(train_detail)
        history['lr'].append(current_lr)
        
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_detail, val_psnr, val_ssim = validate(model, val_loader, criterion, device, logger, epoch+1, ocr_module)
            scheduler.step(val_loss)
            history['val_loss'].append(val_loss)
            history['psnr'].append(val_psnr)
            history['ssim'].append(val_ssim)
            early_stopping(val_loss, model, save_dir, epoch+1, optimizer, history, logger)
        
        # Visuals & Plots every epoch
        model.eval()
        with torch.no_grad():
            v_batch = next(iter(vis_loader))
            v_lr, v_hr, v_plates = v_batch['LR'].to(device), v_batch['HR'].to(device), v_batch['plate_text']
            save_comparison_grid(v_lr, model(v_lr), v_hr, v_plates, ocr_module, save_dir / f'visuals/epoch_{epoch+1}.png')
        
        if len(history['train_loss']) >= 1:
            plot_losses(history['train_loss'], history['val_loss'], save_dir / 'loss_plot.png')
            plot_metrics(history['psnr'], history['ssim'], save_dir / 'metrics_plot.png')
            
            # Save history JSON
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=4)
            
        if early_stopping.early_stop: break
        
        # Save periodic backup
        save_training_state({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_loss': early_stopping.best_loss,
        }, save_dir, 'backup.pt')

    logger.info(f"\nTraining Complete! Total Time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    import logging
    import torch.utils.data as data
    from torch.utils.data import DataLoader
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

