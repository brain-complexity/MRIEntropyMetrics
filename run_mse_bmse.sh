#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="python3"
FORWARD_ARGS=()

while (($#)); do
    case "$1" in
        --python)
            if (($# < 2)); then
                echo "error: --python requires a path" >&2
                exit 2
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_BIN="${1#*=}"
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! command -v -- "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: Python interpreter not found: $PYTHON_BIN" >&2
    exit 5
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1

if ! "$PYTHON_BIN" -c 'import numpy, pandas, nibabel, scipy, joblib, EntropyHub' >/dev/null 2>&1; then
    echo "error: missing Python dependencies" >&2
    echo "required: numpy pandas nibabel scipy joblib EntropyHub" >&2
    exit 5
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/mse_bmse_pipeline.py" "${FORWARD_ARGS[@]}"
