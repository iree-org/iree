#!/bin/bash
# Sets up a venv suitable for running IREE Dispatch Profiler and executes a 
# suite of runs. This is invoked by a Github workflow and can be invoked locally.
# 
# Recommend getting default 'python' to be python 3. For example on Debian:
#   sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# Or launch with python=/some/path

set -xeuo pipefail

TD="$(cd $(dirname $0) && pwd)"

PYTHON="${PYTHON:-python3}"
DISPATCH_PROFILER_IREE_BIN_DIR="${1:-$TD/../../tools}"
DISPATCH_PROFILER_OUTPUT_DIR="${2:-$TD/dispatch_profiler_output}"

VENV_DIR="$TD/dispatch-profiler.venv"

echo "Setting up venv dir: $VENV_DIR"
echo "Python: $PYTHON"
echo "Python version: $("$PYTHON" --version)"
echo "Dispatch Profiler IREE bin dir: $DISPATCH_PROFILER_IREE_BIN_DIR"
echo "Dispatch Profiler output dir: $DISPATCH_PROFILER_OUTPUT_DIR"

function die() {
  echo "Error executing command: $*"
  exit 1
}

$PYTHON -m venv "$VENV_DIR" || die "Could not create venv."
source "$VENV_DIR/bin/activate" || die "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || die "Could not upgrade pip"
python -m pip install --upgrade -r "$TD/requirements.txt"

python "${TD}/generator.py" \
  --generated-dir "${TD}" \
  || die "Dispatch profiler failed to generate"
python "${TD}/compile.py" \
  --verbose \
  --iree-bin-dir "${DISPATCH_PROFILER_IREE_BIN_DIR}" \
  --generated-dir "${TD}" \
  || die "Dispatch profiler failed to compile"
python "${TD}/profiler.py" \
  --verbose \
  --iree-bin-dir "${DISPATCH_PROFILER_IREE_BIN_DIR}" \
  --generated-dir "${TD}" \
  --dispatches="matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync,matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_128x128_16x5_tensorcore_mmasync" \
  --output "${DISPATCH_PROFILER_OUTPUT_DIR}/matmul_perf_tensor_core_a100.csv" \
  || die "Dispatch profiler failed to profile"
