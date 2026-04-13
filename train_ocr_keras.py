"""
train_ocr_keras.py - Fine-tuning Pipeline for the OCR Teacher Model.

This script facilitates the refinement of the Keras-based OCR model using 
High-Resolution (HR) ground truth images from the ICPR2026 dataset. 
Fine-tuning the Teacher ensures that the Super-Resolution 'Student' 
receives high-quality alphanumeric supervision during training.

Key Features:
    - Custom Objects for Cross-Version Compatibility: Loads legacy Keras 2.3 
      models into modern TensorFlow 2.x/Keras 3 environments.
    - OCRDataGenerator: Memory-efficient streaming of rectified plate images.
    - Multi-Task Loss: Categorical Cross-Entropy across 7 character positions.
    - Adaptive Training: Early stopping and learning rate reduction on plateau.
    - Format Upgrade: Saves refined weights in modern '.weights.h5' format.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import os
import json
import shutil
import numpy as np
import cv2
import sys
import random


# IMPORT TORCH BEFORE TENSORFLOW TO PREVENT CUDA SYMBOL CONFLICTS
import torch


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.layers as layers

# Add current dir for imports
sys.path.append(os.getcwd())
try:
    from dataset import discover_tracks, flatten_pairs
    from utils import rectify_image, padding
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


class OCRDataGenerator(Sequence):
    """
    Memory-efficient Keras Data Generator for large-scale OCR training.
    
    Streams rectified High-Resolution plate images and their corresponding 
    one-hot encoded character labels directly from the filesystem to prevent 
    RAM exhaustion when handling 100k+ samples.
    """
    
    def __init__(self, samples: List[Dict[str, Any]], char_mappings: Dict[str, List[str]], 
                 num_classes: Dict[str, int], batch_size: int = 32, shuffle: bool = True) -> None:
        """
        Initializes the generator with dataset metadata.
        
        Args:
            samples (List[Dict[str, Any]]): List of image-label pairs.
            char_mappings (Dict[str, List[str]]): Mapping of indices to characters for each position.
            num_classes (Dict[str, int]): Total alphanumeric categories per position.
            batch_size (int): Training batch size.
            shuffle (bool): Whether to randomize order each epoch.
        """
        super().__init__()
        self.samples = list(samples)
        self.char_mappings = char_mappings
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.floor(len(self.samples) / self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        return self._load_batch(batch_samples)

    def _load_batch(self, batch_samples):
        actual_bs = len(batch_samples)
        X = np.zeros((actual_bs, 60, 120, 3), dtype='float32')
        Y = {f'char{i+1}': np.zeros((actual_bs, self.num_classes[f'char{i+1}']), dtype='float32') 
             for i in range(7)}

        for i, s in enumerate(batch_samples):
            try:
                img = cv2.imread(s['hr_path'])
                if img is None:
                    img = np.zeros((60, 120, 3), dtype='uint8')
                else:
                    if s.get('hr_corners'):
                        try:
                            img = rectify_image(img, s['hr_corners'])
                        except:
                            pass
                    img, _, _ = padding(img, 1.85, 2.15, color=(127, 127, 127))
                    img = cv2.resize(img, (120, 60))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                X[i] = img.astype('float32') / 255.0
                
                text = s['plate_text'].upper()
                if len(text) < 7:
                    text = text.ljust(7)
                text = text[:7]
                
                for j in range(7):
                    pos_key = f'char{j+1}'
                    mapping = self.char_mappings[pos_key]
                    char = text[j]
                    idx = mapping.index(char) if char in mapping else 0
                    Y[pos_key][i] = to_categorical(idx, num_classes=self.num_classes[pos_key])
            except Exception:
                pass

        return X, Y


def load_ocr_model(ocr_dir: Union[str, Path]) -> Model:
    """
    Reconstructs the Teacher OCR model architecture and loads weights.
    
    This function specifically handles the complexity of loading legacy models
    by providing a comprehensive 'custom_objects' map for Keras 3.
    
    Args:
        ocr_dir (Union[str, Path]): Directory containing 'model.json' and weights.
        
    Returns:
        Model: The compiled Keras Model instance.
    """
    ocr_dir = Path(ocr_dir)
    json_path = ocr_dir / 'model.json'
    weights_path_v3 = ocr_dir / 'weights_improved.weights.h5'
    weights_path_legacy = ocr_dir / 'weights.hdf5'
    
    if weights_path_v3.exists():
        weights_path = weights_path_v3
        print(f"[OCR Train] Loading fine-tuned weights from {weights_path}")
    else:
        weights_path = weights_path_legacy
        print(f"[OCR Train] Loading legacy weights from {weights_path}")
    
    with open(str(json_path), 'r') as f:
        model_json = f.read()
    
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
        model = model_from_json(model_json, custom_objects=custom_objects)
        print("[OCR Train] Model architecture loaded (with custom_objects).")
    except Exception as e:
        print(f"[OCR Train] custom_objects attempt failed: {e}")
        print("[OCR Train] Trying without custom_objects...")
        model = model_from_json(model_json)
    
    print(f"[OCR Train] Loading weights from {weights_path}")
    model.load_weights(str(weights_path))
    print("[OCR Train] Model + weights loaded successfully!")
    
    return model


def train_ocr_teacher(dataset_root: Union[str, Path], ocr_dir: Union[str, Path], 
                      max_epochs: int = 30, batch_size: int = 64, 
                      patience: int = 5, initial_lr: float = 5e-5) -> None:
    """
    Orchestrates the fine-tuning training loop for the Teacher model.
    
    Args:
        dataset_root (Union[str, Path]): Path to training data.
        ocr_dir (Union[str, Path]): Path to the OCR model directory.
        max_epochs (int): Upper bound on training iterations.
        batch_size (int): Number of samples per SGD step.
        patience (int): Iterations to wait for improvement before early termination.
        initial_lr (float): Starting learning rate for the Adam optimizer.
    """
    print("=" * 60)
    print("  OCR TEACHER FINE-TUNING")
    print("=" * 60)
    
    ocr_dir = Path(ocr_dir)
    params_path = ocr_dir / 'parameters.json'
    
    # 1. Load Parameters
    print(f"\n[1/4] Loading parameters...")
    with open(params_path, 'r') as f:
        params = json.load(f)
    char_mappings = params['ocr_classes']
    num_classes = params['num_classes']
    print(f"  Char positions: {list(num_classes.keys())}")
    print(f"  Classes per pos: {list(num_classes.values())}")
    
    # 2. Load Model
    print(f"\n[2/4] Loading Keras OCR model...")
    model = load_ocr_model(ocr_dir)
    
    # 3. Prepare Data
    print(f"\n[3/4] Discovering dataset...")
    tracks = discover_tracks(dataset_root)
    samples = flatten_pairs(tracks)
    print(f"  Total: {len(tracks)} tracks, {len(samples)} images")
    
    random.seed(42)
    random.shuffle(samples)
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_gen = OCRDataGenerator(train_samples, char_mappings, num_classes, 
                                  batch_size=batch_size, shuffle=True)
    val_gen = OCRDataGenerator(val_samples, char_mappings, num_classes, 
                                batch_size=batch_size, shuffle=False)
    print(f"  Train batches: {len(train_gen)}, Val batches: {len(val_gen)}")
    
    # 4. Compile and Train
    print(f"\n[4/4] Compiling and training...")
    
    # Per-output loss and metrics
    loss_dict = {f'char{i+1}': 'categorical_crossentropy' for i in range(7)}
    metrics_dict = {f'char{i+1}': 'accuracy' for i in range(7)}
    
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss=loss_dict,
        metrics=metrics_dict
    )
    
    # Callbacks
    checkpoint_path = str(ocr_dir / 'weights_best.weights.h5')
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            mode='min'
        )
    ]
    
    print(f"\n{'='*60}")
    print(f"  TRAINING CONFIG:")
    print(f"    Max Epochs:      {max_epochs}")
    print(f"    Batch Size:      {batch_size}")
    print(f"    Initial LR:      {initial_lr}")
    print(f"    Early Stopping:  patience={patience} (on val_loss)")
    print(f"    LR Scheduler:    ReduceLROnPlateau (factor=0.5, patience=2)")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Save improved weights
    save_path = ocr_dir / 'weights_improved.weights.h5'
    model.save_weights(str(save_path))
    print(f"\n[DONE] Improved weights saved to: {save_path}")
    print(f"  OCR Teacher is now using the fine-tuned model (Keras 3 format).")
    
    # Print final per-char accuracy summary
    print(f"\n{'='*60}")
    print("  FINAL VAL ACCURACY PER CHARACTER:")
    print(f"{'='*60}")
    for i in range(7):
        key = f'char{i+1}'
        val_key = f'val_{key}_accuracy'
        if val_key in history.history:
            final_acc = history.history[val_key][-1] * 100
            print(f"  {key}: {final_acc:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune OCR Teacher on HR images')
    parser.add_argument('--dataset', type=str,
        default='')
    parser.add_argument('--ocr_dir', type=str,
        default='')
    parser.add_argument('--max_epochs', type=int, default=30,
        help='Maximum epochs (early stopping will kick in before this)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5,
        help='Early stopping patience on val_loss')
    parser.add_argument('--lr', type=float, default=5e-5,
        help='Initial learning rate (will be reduced by scheduler)')
    args = parser.parse_args()
    
    train_ocr_teacher(
        args.dataset, args.ocr_dir, 
        max_epochs=args.max_epochs, 
        batch_size=args.batch_size,
        patience=args.patience,
        initial_lr=args.lr
    )
