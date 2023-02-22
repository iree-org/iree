# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/bin/bash

set -euo pipefail

# Install Bazel. From https://www.tensorflow.org/install/source
npm install -g @bazel/bazelisk

# Create root dir.
ROOT_DIR=/tmp/benchmarks
rm -rf "${ROOT_DIR}"
mkdir "${ROOT_DIR}"
mkdir "${ROOT_DIR}/models"
mkdir "${ROOT_DIR}/models/tflite"
mkdir "${ROOT_DIR}/models/iree"
mkdir "${ROOT_DIR}/test_data"
mkdir "${ROOT_DIR}/output"

wget https://storage.googleapis.com/iree-model-artifacts/tflite_squad_test_data.zip -O /tmp/tflite_squad_test_data.zip
unzip /tmp/tflite_squad_test_data.zip -d "${ROOT_DIR}/test_data/"
wget https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/mobilebert_float_384_gpu.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_224_1.0_uint8.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_1.0_224.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/deeplabv3.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_static_1.0_int8.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/resnet_v2_101_1_default_1.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/asr_conformer_int8.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/albert_lite_base_squadv1_1.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_fpnlite_fp32.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_fpnlite_uint8.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/inception_v4_299_fp32.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/inception_v4_299_uint8.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_fp32_2.tflite -P "${ROOT_DIR}/models/tflite/"
wget https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite -P "${ROOT_DIR}/models/tflite/"

# Build IREE source.
SOURCE_DIR=/tmp/github
rm -rf "${SOURCE_DIR}"
mkdir "${SOURCE_DIR}"
cd "${SOURCE_DIR}"

git clone https://github.com/openxla/iree.git

cd iree
git submodule update --init
cmake -GNinja -B ../iree-build/ -S . -DCMAKE_CXX_FLAGS="-Wno-deprecated-builtins" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIREE_ENABLE_LLD=ON
cmake --build ../iree-build/

export CC=clang
export CXX=clang++
python3 configure_bazel.py

cd integrations/tensorflow
bazel build -c opt --cxxopt="-Wno-deprecated-builtins" iree_tf_compiler:iree-import-tflite
./symlink_binaries.sh

IREE_IMPORT_TFLITE_PATH="$(pwd)/bazel-bin/iree_tf_compiler/iree-import-tflite"
IREE_COMPILE_PATH="${SOURCE_DIR}/iree-build/tools/iree-compile"
TFLITE_MODEL_DIR="${ROOT_DIR}/models/tflite"
IREE_MODEL_DIR="${ROOT_DIR}/models/iree"

rm -rf "${IREE_MODEL_DIR}/cuda"
rm -rf "${IREE_MODEL_DIR}/llvm-cpu"
mkdir -p "${IREE_MODEL_DIR}/cuda"
mkdir -p "${IREE_MODEL_DIR}/llvm-cpu"

# Runs `iree-compile` on all TFLite files in directory. If compilation fails, we
# keep going.
for i in $(ls ${ROOT_DIR}/models/tflite/); do
  MODEL_NAME=$(basename $i .tflite)
  echo "Processing ${MODEL_NAME} ..."

  ${IREE_IMPORT_TFLITE_PATH} "${TFLITE_MODEL_DIR}/${MODEL_NAME}.tflite" -o "${IREE_MODEL_DIR}/${MODEL_NAME}.mlir" || true
  # Build for CUDA.
  echo "Compiling ${MODEL_NAME}.vmfb for cuda..."
  "${IREE_COMPILE_PATH}" \
    --iree-input-type=tosa \
    --iree-hal-target-backends=cuda \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-llvm-debug-symbols=false \
    --iree-vm-bytecode-module-strip-source-map=true \
    --iree-vm-emit-polyglot-zip=false \
    "${IREE_MODEL_DIR}/${MODEL_NAME}.mlir" \
    --o "${IREE_MODEL_DIR}/cuda/${MODEL_NAME}.vmfb" || true

  echo "Compiling ${MODEL_NAME}_fp16.vmfb for cuda..."
  "${IREE_COMPILE_PATH}" \
    --iree-input-type=tosa \
    --iree-hal-target-backends=cuda \
    --iree-hal-cuda-llvm-target-arch=sm_80 \
    --iree-flow-demote-f32-to-f16 \
    --iree-llvm-debug-symbols=false \
    --iree-vm-bytecode-module-strip-source-map=true \
    --iree-vm-emit-polyglot-zip=false \
    "${IREE_MODEL_DIR}/${MODEL_NAME}.mlir" \
    --o "${IREE_MODEL_DIR}/cuda/${MODEL_NAME}_fp16.vmfb" || true

  # Build for x86.
  echo "Compiling ${MODEL_NAME}.vmfb for llvm-cpu..."
  "${IREE_COMPILE_PATH}" \
    --iree-input-type=tosa \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvm-target-cpu=cascadelake \
    --iree-llvm-target-triple=x86_64-unknown-linux-gnu \
    --iree-llvm-debug-symbols=false \
    --iree-vm-bytecode-module-strip-source-map=true \
    --iree-vm-emit-polyglot-zip=false \
    "${IREE_MODEL_DIR}/${MODEL_NAME}.mlir" \
    --o "${IREE_MODEL_DIR}/llvm-cpu/${MODEL_NAME}.vmfb" || true

  echo "Compiling ${MODEL_NAME}_padfuse.vmfb for llvm-cpu..."
  "${IREE_COMPILE_PATH}" \
    --iree-input-type=tosa \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvm-target-cpu=cascadelake \
    --iree-llvm-target-triple=x86_64-unknown-linux-gnu \
    --iree-flow-enable-fuse-padding-into-linalg-consumer-ops \
    --iree-llvm-debug-symbols=false \
    --iree-vm-bytecode-module-strip-source-map=true \
    --iree-vm-emit-polyglot-zip=false \
    "${IREE_MODEL_DIR}/${MODEL_NAME}.mlir" \
    --o "${IREE_MODEL_DIR}/llvm-cpu/${MODEL_NAME}_padfuse.vmfb" || true
done

cp "${SOURCE_DIR}/iree-build/tools/iree-benchmark-module" "${ROOT_DIR}/"

# Build TFLite benchmark.
sudo apt-get install libgles2-mesa-dev

export CC=clang
export CXX=clang++

cd "${SOURCE_DIR}"
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
# Select defaults and answer No to all questions.
python3 configure.py

bazel build -c opt --copt=-DCL_DELEGATE_NO_GL \
  --copt=-DMESA_EGL_NO_X11_HEADERS=1 \
  tensorflow/lite/tools/benchmark:benchmark_model

cp "${SOURCE_DIR}/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model" "${ROOT_DIR}/"

# Run benchmark.
cd "${SOURCE_DIR}/iree"
python3.9 build_tools/benchmarks/comparisons/run_benchmarks.py \
  --device_name=desktop --base_dir=${ROOT_DIR} \
  --output_dir=${ROOT_DIR}/output --mode=desktop

cat "${ROOT_DIR}/output/results.csv"
