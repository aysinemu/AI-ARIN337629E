#!/bin/bash
# ============================================================================
# run_pipeline.sh - Orchestration Script for License Plate Super-Resolution
# ============================================================================
#
# This script manages the end-to-end training lifecycle:
#   Phase 1: Teacher Validation. Assesses OCR model accuracy.
#   Phase 2: Teacher Refinement. (Optional) Fine-tunes OCR if below threshold.
#   Phase 3: Student Training. Trains the SR network using multi-modal losses.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Configuration:
#   Adjust paths (DATASET, OCR_DIR, SAVE_DIR) and hyperparameters in the CONFIG 
#   section below before execution.
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIG - Edit these paths if needed
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET=""
OCR_DIR=""
SAVE_DIR="${SCRIPT_DIR}/experiments/v3"

# OCR Training Config
OCR_THRESHOLD=95.0
OCR_MAX_EPOCHS=30       # Each round trains up to 30 epochs (early stopping kicks in)
OCR_BATCH=64
OCR_LR=5e-5             # Initial LR (ReduceLROnPlateau will lower it)
OCR_PATIENCE=5           # Early stopping patience
OCR_EVAL_TRACKS=300      # Number of tracks to evaluate (more = more reliable)
OCR_MAX_ROUNDS=3         # Max rounds of OCR training before giving up

# SR Training Config
SR_BATCH=8
SR_EPOCHS=3000
SR_LR=5e-4
SR_DATA_FRACTION=0.1    # Use entire dataset (0.1 for quick debugging)
SR_VAL_FREQ=50           # Validate every N epochs
SR_PATIENCE=10          # Early stopping patience
SR_RESUME=""            # Path to .pt to resume training
SR_FORCE_ICNR="false"   # Set to "true" to force ICNR init on resume/start

# SR Loss Weighting Configuration (CRITICAL)
# ------------------------------------------------------------
# Alpha (1.0):   Pixel Loss (MSE) -> Basic structural reconstruction.
SR_BETA=0.85            # Perceptual Loss (VGG) -> Enhances edge sharpness and texture.
SR_GAMMA=0.01           # OCR Multi-Task Loss -> Minimizes character-level edit distance.
                        # Note: Typically kept low (or 0.0) initially for stability.
SR_DELTA=2.0            # Latent Correlation Loss -> Semantic alignment between LR/HR.
SR_ZETA=0.35            # Downsampling consistency loss weight.
SR_ETA=0.35             # Total Variation (TV) -> Reduces high-frequency artifact noise.
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0


# ============================================================================
# COLORS
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER
# ============================================================================
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo -e "${BLUE}[$(timestamp)]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}[$(timestamp)] [OK] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(timestamp)] [WARN] $1${NC}"
}

log_err() {
    echo -e "${RED}[$(timestamp)] [ERROR] $1${NC}"
}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

echo ""
echo "============================================================"
echo "  AUTOMATED TRAINING PIPELINE"
echo "  $(timestamp)"
echo "============================================================"
echo ""
log "Script dir:     ${SCRIPT_DIR}"
log "Dataset:        ${DATASET}"
log "OCR model:      ${OCR_DIR}"
log "SR save dir:    ${SAVE_DIR}"
log "OCR threshold:  ${OCR_THRESHOLD}%"
log "OCR max epochs: ${OCR_MAX_EPOCHS} (per round, with early stopping patience=${OCR_PATIENCE})"
log "OCR max rounds: ${OCR_MAX_ROUNDS}"
echo ""

# ---- Phase 1: OCR Teacher ----
echo "============================================================"
echo "  PHASE 1: OCR TEACHER EVALUATION & TRAINING"
echo "============================================================"
echo ""

OCR_PASS=false
ROUND=0

while [ "$OCR_PASS" = false ] && [ $ROUND -lt $OCR_MAX_ROUNDS ]; do
    ROUND=$((ROUND + 1))
    log "Round $ROUND/$OCR_MAX_ROUNDS: Evaluating OCR accuracy on $OCR_EVAL_TRACKS tracks..."
    
    set +e  # Don't exit on eval failure (exit code 1 = below threshold)
    python -u "${SCRIPT_DIR}/ocr_eval.py" \
        --dataset "$DATASET" \
        --ocr_path "$OCR_DIR" \
        --threshold $OCR_THRESHOLD \
        --limit_tracks $OCR_EVAL_TRACKS
    
    EVAL_EXIT=$?
    set -e
    
    if [ $EVAL_EXIT -eq 0 ]; then
        log_ok "OCR accuracy >= ${OCR_THRESHOLD}%! Teacher is ready."
        OCR_PASS=true
    else
        log_warn "OCR accuracy < ${OCR_THRESHOLD}%. Starting fine-tuning round $ROUND..."
        echo ""
        
        echo "------------------------------------------------------------"
        echo "  TRAINING OCR (Round $ROUND / max ${OCR_MAX_ROUNDS})"
        echo "  Max epochs: ${OCR_MAX_EPOCHS}, Early stop patience: ${OCR_PATIENCE}"
        echo "------------------------------------------------------------"
        
        python -u "${SCRIPT_DIR}/train_ocr_keras.py" \
            --dataset "$DATASET" \
            --ocr_dir "$OCR_DIR" \
            --max_epochs $OCR_MAX_EPOCHS \
            --batch_size $OCR_BATCH \
            --patience $OCR_PATIENCE \
            --lr $OCR_LR
        
        log_ok "OCR fine-tuning round $ROUND complete. Re-evaluating..."
        echo ""
    fi
done

if [ "$OCR_PASS" = false ]; then
    log_err "OCR did not reach ${OCR_THRESHOLD}% after $OCR_MAX_ROUNDS rounds."
    log_warn "Proceeding with SR training using the best available OCR..."
fi

echo ""
echo "============================================================"
echo "  PHASE 2: SUPER-RESOLUTION MODEL TRAINING"
echo "============================================================"
echo ""

# ---- Phase 2: Train SR Model ----
mkdir -p "$SAVE_DIR"

log "Starting SR training..."
log "  Batch: $SR_BATCH | LR: $SR_LR | Data Fraction: $SR_DATA_FRACTION"
log "  Epochs: $SR_EPOCHS | Patience: $SR_PATIENCE"
log "  Save dir: $SAVE_DIR"
echo ""

python -u "${SCRIPT_DIR}/train.py" \
    --dataset "$DATASET" \
    --save "$SAVE_DIR" \
    --batch $SR_BATCH \
    --epochs $SR_EPOCHS \
    --lr $SR_LR \
    --data_fraction $SR_DATA_FRACTION \
    --val_freq $SR_VAL_FREQ \
    --ocr_path "$OCR_DIR" \
    --alpha 1.0 \
    --beta $SR_BETA \
    --gamma $SR_GAMMA \
    --delta $SR_DELTA \
    --zeta $SR_ZETA \
    --eta $SR_ETA \
    --patience $SR_PATIENCE \
    --resume "$SR_RESUME" \
    $( [ "$SR_FORCE_ICNR" = "true" ] && echo "--force_icnr" )

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE!"
echo "  $(timestamp)"
echo "============================================================"
echo ""
log_ok "OCR Teacher trained and validated."
log_ok "SR Model training finished."
log "Results saved in: $SAVE_DIR"
echo ""
