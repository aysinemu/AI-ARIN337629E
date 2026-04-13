# AI-ARIN337629E — License Plate Super-Resolution with OCR-Guided Training

> **Semantic-Guided Super-Resolution for License Plate Recognition**  
> Submission for the **ICPR 2026 Competition** — License Plate Character Super-Resolution & Recognition  
> Course: **ARIN337629E — Artificial Intelligence**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Module Details](#module-details)
  - [network.py — SR Network Architecture](#networkpy--sr-network-architecture)
  - [train.py — Multi-Task Training Pipeline](#trainpy--multi-task-training-pipeline)
  - [test.py — Comprehensive Evaluation](#testpy--comprehensive-evaluation)
  - [dataset.py — ICPR2026 Data Pipeline](#datasetpy--icpr2026-data-pipeline)
  - [utils.py — Core Utilities](#utilspy--core-utilities)
  - [train_ocr_keras.py — OCR Teacher Fine-Tuning](#train_ocr_keraspy--ocr-teacher-fine-tuning)
  - [eval_bicubic.py — Baseline Evaluation](#eval_bicubicpy--baseline-evaluation)
  - [eval_track_voting.py — Track-Level Voting Evaluation](#eval_track_votingpy--track-level-voting-evaluation)
  - [ocr_eval.py — OCR Teacher Validation](#ocr_evalpy--ocr-teacher-validation)
  - [run_pipeline.sh — Automated Training Orchestration](#run_pipelinesh--automated-training-orchestration)
- [Loss Functions](#loss-functions)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [Usage](#usage)
  - [Full Automated Pipeline](#full-automated-pipeline)
  - [Individual Scripts](#individual-scripts)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

This project implements an **OCR-guided Super-Resolution (SR)** system for license plate recognition. The core challenge is to recover readable alphanumeric characters from extremely low-resolution (40×20 px) license plate images by upscaling them to high-resolution (160×80 px) — a **4× upscale factor**.

The system uses a **Teacher-Student** paradigm:

| Component | Role | Framework |
|-----------|------|-----------|
| **OCR Teacher** | Pretrained Keras CNN that reads license plate characters | TensorFlow / Keras |
| **SR Student** | Deep Residual Dense Network that generates HR images from LR inputs | PyTorch |

The Teacher provides **semantic supervision** to the Student during training, ensuring that the super-resolved images are not only visually sharp (pixel-level fidelity) but also **machine-readable** (character-level clarity).

### Key Innovation

Unlike traditional SR methods that only optimize pixel-level metrics (PSNR/SSIM), this pipeline integrates **6 complementary loss objectives**:

1. **Pixel Loss** — Structural accuracy (MSE + L1)
2. **Perceptual Loss** — Texture realism (VGG19 features)
3. **OCR-Guided Loss** — Character readability (Levenshtein + weighted confusable chars)
4. **Latent Correlation Loss** — Semantic alignment (cosine similarity in latent space)
5. **Consistency Loss** — Self-supervised belief (downsample SR ≈ LR)
6. **Total Variation Loss** — Noise suppression

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │            ImprovedNetwork (SR Student)                 │
                    │                                                         │
   LR (40×20)  ──►  │  AutoEncoder ──► RDN(16 blocks) ──► ResidualModule      │
                    │       │                                    │            │
                    │       │              FeatureModule ◄───────┘            │
                    │       │                   │  (TFAM Attention)           │
                    │       │              PixelShuffle (4×)                  │
                    │       │                   │                             │
                    │       │              RDN(8 blocks, Post)                │
                    │       │                   │                             │
                    │       │              Output Conv                        │
                    │       │                   │                             │
                    │       │         + Bicubic Upsampled LR                  │
                    │       │                   │                             │
                    │       └──── Latent Encoder ────► Latent_SR              │
                    │                                                         │
                    └──────────────────────┬──────────────────────────────────┘
                                           │
                                      SR (160×80)
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
         Pixel Loss                   VGG Loss                    OCR Loss
        (MSE + L1)              (Feature Distance)          (Levenshtein +
                                                            Weighted Chars)
              │                            │                            │
              └────────────────────────────┼────────────────────────────┘
                                           │
                                    Total Weighted Loss
                                  α·Pixel + β·VGG + γ·OCR
                                + δ·Latent + ζ·Consist + η·TV
```

### Attention Mechanisms

- **TFAM (Text-Focused Attention Module)**: Custom spatial+channel attention designed specifically for alphanumeric character boundaries
- **DPCA (Dual Positional Channel Attention)**: Horizontal and vertical positional encoding combined with feature-level gating
- **AdaptiveResidualBlock**: Dual-path processing — local convolutions + downsampled wide-context path

---

## Project Structure

```
AI-ARIN337629E/
│
├── network.py              # SR network architecture (RDN + TFAM + DPCA)
├── train.py                # Multi-task training pipeline with OCR guidance
├── test.py                 # Comprehensive evaluation & metrics generation
├── dataset.py              # ICPR2026 dataset loader with perspective rectification
├── utils.py                # Utility functions (metrics, logging, plotting, etc.)
├── train_ocr_keras.py      # Fine-tuning pipeline for the OCR Teacher model
├── eval_bicubic.py         # Bicubic interpolation baseline evaluation
├── eval_track_voting.py    # Track-level temporal voting evaluation
├── ocr_eval.py             # OCR Teacher accuracy validation
├── run_pipeline.sh         # End-to-end orchestration script
├── requirements.txt        # Python dependencies
├── AI.pdf                  # Project report / documentation
├── AI.docx                 # Project report / documentation (editable)
│
├── models_ocr/             # Pretrained OCR Teacher model assets
│   ├── model.json             # Keras architecture definition
│   ├── parameters.json        # OCR class mappings and metadata
│   ├── parameters.npy         # Numpy-serialized parameters
│   ├── weights.hdf5           # Original pretrained weights (Keras 2.x)
│   ├── weights_improved.weights.h5  # Fine-tuned weights (Keras 3 format)
│   ├── weights_best.weights.h5      # Best checkpoint during fine-tuning
│   └── weights_backup.hdf5         # Safety backup of original weights
│
└── experiments/            # Training experiment outputs
    ├── v1/                 # Experiment version 1
    └── v2/                 # Experiment version 2
```

---

## Module Details

### `network.py` — SR Network Architecture

The neural network backbone implementing a **Residual Dense Network (RDN)** enhanced with custom attention modules.

| Component | Description |
|-----------|-------------|
| **`ImprovedNetwork`** | Main model hub — manages the full SR pipeline from LR input to HR output |
| **`RDN`** | Residual Dense Network with configurable block count and growth rate |
| **`RDB`** | Residual Dense Block — core building block with dense connections and residual scaling (0.1×) |
| **`DenseLayer`** | Single dense convolution layer with concatenation |
| **`DConv`** | Depthwise Separable Convolution — reduces parameters while maintaining quality |
| **`AutoEncoder`** | Initial feature extraction using PixelShuffle/Unshuffle for noise reduction |
| **`TFAM`** | Text-Focused Attention Module — spatial + channel attention for character boundaries |
| **`DPCA`** | Dual Positional Channel Attention — H/V positional encoding with feature gating |
| **`AdaptiveResidualBlock`** | Dual-path residual block (local conv + wide-context downsampled path) |
| **`ResidualConcatenationBlock`** | Multi-layer residual concatenation with pointwise fusion |
| **`LatentEncoder`** | Maps feature maps to L2-normalized latent space for correlation loss |
| **`HRFeatureEncoder`** | Extracts features from HR ground truth for latent alignment |
| **`VGGFeatureExtractor`** | Frozen VGG19 backbone for perceptual loss computation |

**Key Initialization Techniques:**

- **ICNR (Initialization Causality with Nearest-Neighbor Resize)**: Applied to PixelShuffle layers to eliminate checkerboard artifacts at initialization
- **Variance Reduction (0.1× scaling)**: All non-output Conv2D weights are scaled down by 10× to prevent signal explosion in deep networks
- **Zero-Init Output Layer**: The final conv starts at zero, making the network initially behave as a bicubic upsampler

---

### `train.py` — Multi-Task Training Pipeline

The training orchestrator implementing the complete semantic-guided learning loop.

| Class/Function | Description |
|----------------|-------------|
| **`OCRModule`** | Bridge between PyTorch training and Keras OCR — handles batch prediction and feature extraction |
| **`CombinedLoss`** | Master loss function coordinating 6 loss objectives with configurable weights |
| **`EarlyStopping`** | Monitors val loss and auto-saves best weights with configurable patience |
| **`train_one_epoch()`** | Single epoch training pass with gradient clipping (max_norm=1.0) |
| **`validate()`** | Validation loop computing loss, PSNR, SSIM, and saving comparison grids |
| **`main()`** | Entry point — parses args, manages data subset sampling, checkpointing, and plotting |

**Training Features:**

- **Dynamic Data Subset Sampling**: Each epoch uses a random fraction (`--data_fraction`) of training data for faster iteration
- **OCR Loss Gating**: OCR loss is computed every 4th step to reduce GPU overhead
- **Teacher Reliability Fallback**: If the OCR Teacher misreads the HR image, the OCR loss for that sample is downweighted to 0.2×
- **Confusable Character Weighting**: Characters like `0/O`, `8/B`, `1/I` receive 2× penalty weight

---

### `test.py` — Comprehensive Evaluation

Generates quantitative metrics and qualitative visualizations for model assessment.

**Output Deliverables:**

| Output File | Description |
|-------------|-------------|
| `results.csv` | Per-image metrics (PSNR, SSIM, OCR accuracy, confidence) |
| `results_detailed.csv` | Same as above with global average row appended |
| `accuracy_histogram.png` | Triple-bar chart comparing HR/LR/SR accuracy distributions |
| `confusion_matrix_SR.png` | 36×36 character confusion matrix for SR predictions |
| `confusion_matrix_LR.png` | Character confusion matrix for LR baseline |
| `confusable_char_errors.csv` | Ranked list of confusable character pair errors |
| `grid_<track>.png` | 5×3 visual comparison grids (LR/SR/HR per track) |

---

### `dataset.py` — ICPR2026 Data Pipeline

Handles data discovery, splitting, preprocessing, and PyTorch DataLoader creation.

| Function/Class | Description |
|----------------|-------------|
| **`discover_tracks()`** | Recursively scans `Scenario-A/B > Brazilian/Mercosur > TrackID` hierarchy |
| **`split_tracks()`** | **Track-level splitting** to prevent data leakage (correlated frames stay together) |
| **`flatten_pairs()`** | Converts track-level metadata to individual LR-HR sample pairs |
| **`ICPR2026Dataset`** | PyTorch Dataset with perspective rectification, color augmentation, and padding |
| **`create_dataloaders()`** | Factory function for train/val DataLoaders |
| **`create_test_dataloader()`** | Factory for test/evaluation DataLoader |

**Dataset Constants:**

- LR size: 40×20 pixels (width × height)
- HR size: 160×80 pixels (4× upscale)
- Aspect ratio target: 2.0 ± 0.15
- Background padding color: (127, 127, 127)

**Augmentation Pipeline** (via Albumentations):

- HueSaturationValue jittering
- RandomBrightnessContrast
- RandomGamma

---

### `utils.py` — Core Utilities

Essential helper functions used across all modules.

| Category | Functions |
|----------|-----------|
| **Image Processing** | `padding()` — aspect-ratio-aware border padding<br>`rectify_image()` — perspective transformation (deskewing) |
| **OCR Heuristics** | `get_char_weights()` — confusable character loss weighting<br>`CONFUSABLE_PAIRS` — mapping of easily confused characters (0↔O, 8↔B, etc.) |
| **Metrics** | `calculate_psnr()` — Peak Signal-to-Noise Ratio<br>`calculate_ssim()` — Structural Similarity Index<br>`levenshtein()` — Edit distance for OCR accuracy |
| **Visualization** | `plot_losses()` — dual-axis loss curve with best-epoch annotation<br>`plot_metrics()` — PSNR/SSIM progression charts<br>`plot_confusion_matrix()` — character-level confusion heatmap<br>`build_confusion_matrix()` — 36×36 matrix builder (A-Z + 0-9)<br>`save_comparison_grid()` — LR/SR/HR side-by-side visual grid |
| **Persistence** | `save_training_state()` — checkpoint serialization<br>`load_training_state()` — checkpoint restoration |
| **Logging** | `setup_logging()` — dual-pipe logger (console + file) |

---

### `train_ocr_keras.py` — OCR Teacher Fine-Tuning

Fine-tunes the pretrained Keras OCR model on HR images from the ICPR2026 dataset.

- **`OCRDataGenerator`**: Memory-efficient Keras `Sequence` generator streaming rectified plate images
- **`load_ocr_model()`**: Handles legacy Keras 2.x → Keras 3 compatibility via custom_objects mapping
- **`train_ocr_teacher()`**: Full training loop with:
  - Categorical Cross-Entropy across 7 character positions
  - EarlyStopping on `val_loss`
  - ReduceLROnPlateau scheduler
  - ModelCheckpoint for best weights
  - Saves improved weights in `.weights.h5` (Keras 3) format

---

### `eval_bicubic.py` — Baseline Evaluation

Calculates baseline PSNR and SSIM using standard **Bicubic Interpolation** (OpenCV `INTER_CUBIC`). Provides the reference point for measuring neural SR improvement.

---

### `eval_track_voting.py` — Track-Level Voting Evaluation

Implements **Confidence-Weighted Majority Voting** across 5-frame video tracks:

1. Groups SR outputs back into their original video tracks
2. For each track, collects all 5 OCR predictions with confidence scores
3. Selects the most frequent prediction (frequency-first, confidence tie-breaker)
4. Reports **Track-Level Full Plate Accuracy** and **Character Accuracy**

This mirrors real-world deployment where temporal consistency boosts robustness.

---

### `ocr_eval.py` — OCR Teacher Validation

Standalone sanity check for the OCR Teacher model. Evaluates recognition accuracy on HR images to ensure the Teacher is reliable before using it for Student distillation.

- Returns exit code 0 if accuracy ≥ threshold (default 95%), otherwise exit code 1
- Useful for CI/CD integration and automated pipeline gating

---

### `run_pipeline.sh` — Automated Training Orchestration

End-to-end bash script managing the complete training lifecycle:

```
Phase 1: OCR Teacher Evaluation & Training
    └─► Iterative loop: Evaluate → Fine-tune → Re-evaluate (up to 3 rounds)
    └─► Gate: OCR accuracy must reach ≥ 95% before proceeding

Phase 2: Super-Resolution Model Training
    └─► Launches train.py with all configured hyperparameters
    └─► Includes loss weight configuration, data fraction, etc.
```

---

## Loss Functions

The total loss is a weighted combination:

```
Total = α·Pixel + β·Perceptual + γ·OCR + δ·Latent + ζ·Consistency + η·TV
```

| Symbol | Loss | Default Weight | Description |
|--------|------|:--------------:|-------------|
| α | Pixel Loss | 1.0 | `(MSE + L1) / 2` — structural fidelity |
| β | Perceptual Loss | 0.85 | VGG19 feature-space L1 distance — texture realism |
| γ | OCR Loss | 0.01 | Weighted Levenshtein + confusable char penalties — readability |
| δ | Latent Correlation | 2.0 | `1 - cos_similarity(latent_SR, latent_HR)` — semantic alignment |
| ζ | Consistency Loss | 0.35 | `MSE(downsample(SR), LR)` — self-supervised belief |
| η | TV Loss | 0.35 | Total Variation penalty — noise suppression |

---

## Dataset Format

The ICPR2026 dataset follows this directory structure:

```
train/
├── Scenario-A/
│   ├── Brazilian/
│   │   ├── <track_id>/
│   │   │   ├── annotations.json
│   │   │   ├── lr-001.png    # Low-resolution frame 1
│   │   │   ├── hr-001.png    # High-resolution frame 1
│   │   │   ├── lr-002.png
│   │   │   ├── hr-002.png
│   │   │   └── ...           # Up to 5 LR-HR pairs per track
│   │   └── ...
│   └── Mercosur/
│       └── ...
└── Scenario-B/
    └── ...
```

**`annotations.json`** contains:

- `plate_text`: Ground truth alphanumeric string (e.g., `"AVL5215"`)
- `plate_layout`: Plate geometry type (`"Brazilian"` or `"Mercosur"`)
- `corners`: Per-image quadrilateral bounding box coordinates for perspective rectification

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: ≥ 8 GB VRAM)
- CUDA Toolkit 12.x

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AI-ARIN337629E

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0 | SR model training & inference |
| TensorFlow | 2.20.0 | OCR Teacher model (Keras) |
| torchvision | 0.25.0 | VGG19 perceptual loss & transforms |
| OpenCV | 4.11.0 | Image processing & perspective transform |
| Albumentations | 2.0.8 | Data augmentation pipeline |
| scikit-image | 0.25.2 | PSNR / SSIM metric calculation |
| Pandas | 2.3.3 | Results tabulation & CSV export |
| Matplotlib | 3.10.7 | Visualization & plotting |
| tqdm | 4.67.1 | Progress bars |

> **Note:** Both PyTorch and TensorFlow coexist in this project. The `OCRModule` in `train.py` carefully manages GPU memory growth for TensorFlow to prevent VRAM conflicts.

---

## Usage

### Full Automated Pipeline

Edit the configuration section in `run_pipeline.sh` with your paths, then:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This will automatically:

1. Evaluate the OCR Teacher accuracy
2. Fine-tune the OCR Teacher if below 95% threshold (up to 3 rounds)
3. Train the SR Student model with all configured losses

### Individual Scripts

#### Train SR Model

```bash
python train.py \
    --dataset /path/to/train/ \
    --save ./experiments/v3 \
    --ocr_path ./models_ocr \
    --batch 8 \
    --epochs 3000 \
    --lr 5e-4 \
    --data_fraction 0.1 \
    --val_freq 50 \
    --patience 10 \
    --alpha 1.0 --beta 0.85 --gamma 0.01 \
    --delta 2.0 --zeta 0.35 --eta 0.35
```

#### Evaluate SR Model

```bash
python test.py \
    --dataset /path/to/train/ \
    --save ./results/ \
    --model ./experiments/v3/bestmodel.pt \
    --ocr_path ./models_ocr \
    --batch 8 \
    --save_images
```

#### Evaluate with Track-Level Voting

```bash
python eval_track_voting.py \
    --dataset /path/to/train/ \
    --model ./experiments/v3/bestmodel.pt \
    --ocr_path ./models_ocr \
    --batch 16
```

#### Bicubic Baseline

```bash
python eval_bicubic.py \
    --dataset /path/to/train/ \
    --train_ratio 0.7
```

#### Fine-Tune OCR Teacher

```bash
python train_ocr_keras.py \
    --dataset /path/to/train/ \
    --ocr_dir ./models_ocr \
    --max_epochs 30 \
    --batch_size 64 \
    --lr 5e-5 \
    --patience 5
```

#### Validate OCR Teacher

```bash
python ocr_eval.py \
    --dataset /path/to/train/ \
    --ocr_path ./models_ocr \
    --threshold 95.0 \
    --limit_tracks 200
```

---

## Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **PSNR** | Pixel-Level | Peak Signal-to-Noise Ratio (dB) — higher is better |
| **SSIM** | Perceptual | Structural Similarity Index [0, 1] — higher is better |
| **Plate Accuracy** | OCR | Percentage of plates with 100% correct character match |
| **Character Accuracy** | OCR | Per-character correct prediction rate |
| **Levenshtein Distance** | OCR | Edit distance between predicted and ground truth strings |
| **Confidence Score** | OCR | Model certainty for each prediction |

---

## Key Design Decisions

### 1. Track-Level Data Splitting

Frames within a video track are **highly correlated**. Splitting at the track level (not image level) prevents data leakage and ensures honest validation metrics.

### 2. Teacher Reliability Weighting

If the OCR Teacher cannot correctly read the HR ground truth image, the OCR loss weight for that sample is reduced to **0.2×**. This prevents the Teacher from misleading the Student with incorrect supervision.

### 3. OCR Loss Frequency Gating

OCR inference through Keras is computationally expensive. The system gates OCR loss computation to **every 4th training step**, maintaining >95% of the semantic benefit while saving ~75% of the OCR computation cost.

### 4. Confusable Character Penalties

License plates contain characters that are easily confused (e.g., `0/O`, `8/B`, `1/I`, `5/S`). These pairs receive a **2× penalty weight** in the OCR loss to force the SR network to render these specific characters with extra clarity.

### 5. ICNR Initialization

Sub-pixel convolution (PixelShuffle) layers are initialized with **ICNR** to behave as Nearest-Neighbor upsampling at initialization, eliminating the checkerboard artifacts that typically plague newly initialized SR models.

### 6. Residual Scaling (0.1×)

All Residual Dense Blocks multiply their output by **0.1** before the residual addition. This variance reduction technique is critical for training stability in very deep networks (16+ RDB blocks).

### 7. Global Residual Bicubic Connection

The final SR output is computed as: `SR = NetworkOutput + Bicubic(LR)`. This ensures the network only needs to learn the **residual difference** from bicubic upsampling, dramatically simplifying the learning task.

---

<p align="center">
  <sub>Built with PyTorch + TensorFlow | ICPR 2026 Competition</sub>
</p>
