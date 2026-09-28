#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh  —  Complete Phase 1 pipeline for vibration-ssl-pdm
#
# Runs every step in order. Safe to re-run: each step checks its own
# prerequisites and skips if files are already present.
#
# USAGE
#   chmod +x run_phase1.sh
#   ./run_phase1.sh
#
# To run individual stages only:
#   ./run_phase1.sh --from 5       # start from step 5 (SSL training)
#   ./run_phase1.sh --steps 11,12  # run only steps 11 and 12
#
# STEPS
#   1   Download CWRU .mat files
#   2   Build processed segments (segments.npz)
#   3   Dataset exploration + stats
#   4   Encoder forward-pass test
#   5   Train SSL encoder  (~20-40 min CPU)
#   6   Extract embeddings
#   7   t-SNE / UMAP visualisation
#   8   Anomaly detection (EDR + health monitor)
#   9   Downstream classification (SVM / MLP on frozen embeddings)
#   10  Supervised baseline CNN training
#   11  Comparison table + figure (SSL vs baseline)
#   12  Launch Streamlit dashboard
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

step()   { echo -e "\n${CYAN}${BOLD}══ STEP $1: $2 ══${RESET}"; }
ok()     { echo -e "  ${GREEN}✓ $1${RESET}"; }
skip()   { echo -e "  ${YELLOW}↷ Skipping: $1${RESET}"; }

FROM_STEP=1
ONLY_STEPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)   FROM_STEP="$2"; shift 2 ;;
    --steps)  ONLY_STEPS="$2"; shift 2 ;;
    *)        echo "Unknown arg: $1"; exit 1 ;;
  esac
done

should_run() {
  local s="$1"
  if [[ -n "$ONLY_STEPS" ]]; then
    echo ",$ONLY_STEPS," | grep -q ",$s," && return 0 || return 1
  fi
  [[ "$s" -ge "$FROM_STEP" ]] && return 0 || return 1
}

echo -e "\n${BOLD}Vibration SSL-PDM — Phase 1 pipeline${RESET}"
echo "Project: $SCRIPT_DIR"

# --- 0. Env setup ---
if [[ ! -x "$PY" ]]; then
  echo "Creating venv with Python 3.13..."
  /usr/local/bin/python3.13 -m venv "$VENV"
  "$PIP" install --upgrade pip -q
  "$PIP" install torch --index-url https://download.pytorch.org/whl/cpu -q
  "$PIP" install -r "$SCRIPT_DIR/requirements.txt" -q
  ok "Environment ready"
else
  "$PY" -c "import torch" 2>/dev/null || {
    "$PIP" install torch --index-url https://download.pytorch.org/whl/cpu -q
    "$PIP" install -r "$SCRIPT_DIR/requirements.txt" -q
  }
  ok "Virtual environment: $VENV"
fi

# --- Step 1: Download ---
if should_run 1; then
  step 1 "Download CWRU dataset"
  N_MAT=$(ls "$SCRIPT_DIR/data/raw/"*.mat 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$N_MAT" -ge 60 ]]; then
    skip "$N_MAT .mat files already present"
  else
    echo "  Attempting to download CWRU dataset (~300 MB)..."
    if "$PY" -m preprocessing.download_cwru --all 2>&1 | grep -qv "failed"; then
      ok "Download complete"
    else
      echo ""
      echo -e "  ${YELLOW}⚠ Network download failed. Manual download required.${RESET}"
      echo ""
      echo "  1. Open this page in your browser:"
      echo "     https://engineering.case.edu/bearingdatacenter/download-data-file"
      echo ""
      echo "  2. Or run this helper script to open all download links:"
      echo "     ./download_cwru_browser.sh"
      echo ""
      echo "  3. Save all 64 .mat files to:"
      echo "     $SCRIPT_DIR/data/raw/"
      echo ""
      echo "  4. Then re-run: ./run_phase1.sh --from 2"
      exit 0
    fi
  fi
fi

# --- Step 2: Build dataset ---
if should_run 2; then
  step 2 "Build processed segments"
  if [[ -f "$SCRIPT_DIR/data/processed/segments.npz" ]]; then
    skip "segments.npz already exists"
  else
    "$PY" -m preprocessing.build_dataset
    ok "segments.npz written"
  fi
fi

# --- Step 3: Explore ---
if should_run 3; then
  step 3 "Dataset exploration"
  "$PY" -m preprocessing.explore
  ok "dataset_summary.json written"
fi

# --- Step 4: Encoder test ---
if should_run 4; then
  step 4 "Encoder forward-pass test"
  "$PY" -c "
import sys; sys.path.insert(0,'.')
import torch
from config import CFG
from models.encoder import CNN1DEncoder
enc = CNN1DEncoder(CFG.model)
x = torch.randn(4, 1, 3000)
y = enc(x)
assert y.shape == (4, 128), f'Bad shape: {y.shape}'
print(f'  encoder: (4,1,3000) -> {tuple(y.shape)}  params: {enc.n_parameters():,}')
"
  ok "Encoder shape verified"
fi

# --- Step 5: SSL training ---
if should_run 5; then
  step 5 "SSL contrastive pretraining (50 epochs)"
  if [[ -f "$SCRIPT_DIR/checkpoints/ssl_encoder.pt" ]]; then
    skip "ssl_encoder.pt exists — delete to retrain"
  else
    echo "  This takes ~20-40 min on CPU. Watch the loss curve..."
    "$PY" -m training.train_ssl
    ok "SSL encoder trained"
  fi
fi

# --- Step 6: Embeddings ---
if should_run 6; then
  step 6 "Extract embeddings"
  if [[ -f "$SCRIPT_DIR/results/embeddings/embeddings.npz" ]]; then
    skip "embeddings.npz already exists"
  else
    "$PY" -m training.extract_embeddings
    ok "Embeddings saved"
  fi
fi

# --- Step 7: Visualisation ---
if should_run 7; then
  step 7 "t-SNE + UMAP visualisation"
  "$PY" -m evaluation.visualization
  ok "Figures written"
fi

# --- Step 8: Anomaly detection ---
if should_run 8; then
  step 8 "Anomaly detection (EDR)"
  "$PY" -m anomaly_detection.anomaly_detector
  ok "Health monitor saved"
fi

# --- Step 9: Downstream classifiers ---
if should_run 9; then
  step 9 "Downstream classification (SVM/MLP)"
  if [[ -f "$SCRIPT_DIR/results/metrics/downstream_classification.json" ]]; then
    skip "downstream_classification.json exists — delete to rerun"
  else
    "$PY" -m training.train_classifier
    ok "downstream_classification.json written"
  fi
  "$PY" -m training.train_finetune
  ok "finetune_classification.json written"
fi

# --- Step 10: Supervised baseline ---
if should_run 10; then
  step 10 "Supervised CNN baseline"
  if [[ -f "$SCRIPT_DIR/results/metrics/baseline_classification.json" ]]; then
    skip "baseline_classification.json exists — delete to rerun"
  else
    "$PY" -m training.train_baseline
    ok "baseline_classification.json written"
  fi
fi

# --- Step 11: Comparison ---
if should_run 11; then
  step 11 "SSL vs baseline comparison"
  "$PY" -m evaluation.evaluate
  ok "comparison.json + 12_ssl_vs_baseline.png"
fi

# --- Step 12: Dashboard ---
if should_run 12; then
  step 12 "Streamlit dashboard"
  echo ""
  echo -e "  ${BOLD}Opening at http://localhost:8501${RESET}"
  echo "  Press Ctrl+C to stop."
  echo ""
  PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/streamlit" run "$SCRIPT_DIR/dashboard/app.py" \
      --server.headless false \
      --browser.gatherUsageStats false
fi

echo -e "\n${GREEN}${BOLD}Phase 1 complete!${RESET}"
echo "Results:   $SCRIPT_DIR/results/"
echo "Dashboard: ./run_phase1.sh --steps 12"
