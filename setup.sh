#!/usr/bin/env bash
# =============================================================================
# Spades-AI  —  one-command setup
#
# Usage:
#     bash setup.sh              # auto-detect: conda or venv
#     bash setup.sh --help       # show help
#
# This script will:
#   1. Create a clean Python 3.12 environment (conda env or venv)
#   2. Install PyTorch (GPU if nvidia-smi found, else CPU)
#   3. Install all other dependencies
#   4. Install rlcard in editable mode
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="spades"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: bash setup.sh"
            echo ""
            echo "Auto-detects conda vs venv and creates a clean environment."
            echo "  - If conda is available: creates conda env '$ENV_NAME' (Python 3.12)"
            echo "  - Otherwise: creates .venv with system python"
            echo ""
            echo "After setup, activate with:"
            echo "  conda activate $ENV_NAME     # if conda"
            echo "  source .venv/bin/activate     # if venv"
            exit 0 ;;
        *)  echo "Unknown flag: $arg (use --help)"; exit 1 ;;
    esac
done

echo "============================================"
echo "  Spades-AI Setup"
echo "============================================"
echo ""

HAS_CONDA=false
command -v conda &>/dev/null && HAS_CONDA=true

HAS_GPU=false
command -v nvidia-smi &>/dev/null && HAS_GPU=true

# ──────────────────────────────────────────────────────────────────────────────
# 1. Create environment
# ──────────────────────────────────────────────────────────────────────────────
if [ "$HAS_CONDA" = true ]; then
    echo "[1/4] Creating conda env '$ENV_NAME' (Python 3.12) ..."

    # Remove old env if exists
    conda deactivate 2>/dev/null || true
    if conda env list | grep -qw "$ENV_NAME"; then
        echo "       Removing existing '$ENV_NAME' env ..."
        conda env remove -n "$ENV_NAME" -y -q 2>&1 | tail -1
    fi

    conda create -n "$ENV_NAME" python=3.12 -y -q 2>&1 | tail -3
    echo "       Activating ..."

    # conda activate needs shell init; use eval for script context
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$ENV_NAME"

    echo "       Python: $(python --version), $(which python)"
else
    echo "[1/4] Creating venv (.venv) ..."
    python3 -m venv .venv --clear
    source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
    echo "       Python: $(python --version), $(which python)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2. Install PyTorch
# ──────────────────────────────────────────────────────────────────────────────
echo "[2/4] Installing PyTorch ..."
pip install --upgrade pip setuptools -q

if [ "$HAS_GPU" = true ]; then
    echo "       GPU detected, installing PyTorch + CUDA 12.4 via pip ..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
else
    echo "       No GPU, installing CPU PyTorch via pip ..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi

python -c "
import torch
cuda = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'       PyTorch {torch.__version__}  CUDA={cuda}  GPU={dev}')
"

# ──────────────────────────────────────────────────────────────────────────────
# 3. Install other dependencies
# ──────────────────────────────────────────────────────────────────────────────
echo "[3/4] Installing dependencies ..."

pip install \
    numpy termcolor pyyaml matplotlib tqdm \
    flask flask-cors Django django-cors-headers \
    -q

pip install onnx onnxruntime -q 2>/dev/null || true

# ──────────────────────────────────────────────────────────────────────────────
# 4. Install rlcard (editable)
# ──────────────────────────────────────────────────────────────────────────────
echo "[4/4] Installing rlcard (editable) ..."
pip install -e rlcard --no-deps -q

# ──────────────────────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"

python -c "
import torch, numpy, rlcard
cuda = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'  PyTorch {torch.__version__}  CUDA={cuda}  GPU={dev}')
print(f'  NumPy {numpy.__version__}  RLCard {rlcard.__version__}')
"

echo ""
if [ "$HAS_CONDA" = true ]; then
    echo "Activate with:  conda activate $ENV_NAME"
else
    echo "Activate with:  source .venv/bin/activate"
fi
echo ""
echo "Train:  cd rlcard && python train_spades_selfplay_drqn.py"
echo ""
