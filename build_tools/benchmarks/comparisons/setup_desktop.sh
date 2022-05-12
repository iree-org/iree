#!/bin/bash

# Install Bazel. From https://www.tensorflow.org/install/source
npm install -g @bazel/bazelisk

# Create root dir.
ROOT_DIR=/tmp/mobilebert_benchmarks
mkdir ${ROOT_DIR}
mkdir ${ROOT_DIR}/models
mkdir ${ROOT_DIR}/models/tflite
mkdir ${ROOT_DIR}/models/iree
mkdir ${ROOT_DIR}/test_data
mkdir ${ROOT_DIR}/output

wget https://storage.googleapis.com/iree-model-artifacts/tflite_squad_test_data.zip -O /tmp/tflite_squad_test_data.zip
unzip /tmp/tflite_squad_test_data.zip -d ${ROOT_DIR}/test_data/
wget https://storage.googleapis.com/iree-model-artifacts/mobilebert_float_384_gpu.tflite -O ${ROOT_DIR}/models/tflite/mobilebert_float_384_gpu.tflite

# Build IREE source.
SOURCE_DIR=/tmp/github
mkdir ${SOURCE_DIR}
cd ${SOURCE_DIR}

#git clone https://github.com/google/iree.git
git clone https://github.com/mariecwhite/iree.git

cd iree
git checkout origin/comparisons
git submodule update --init
# Below is only needed if CUDA is not setup.
#export IREE_CUDA_DEPS_DIR="/usr/local/iree_cuda_deps"
cmake -GNinja -B ../iree-build/ -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DIREE_ENABLE_LLD=ON -DIREE_HAL_DRIVER_CUDA=ON -DIREE_TARGET_BACKEND_CUDA=ON
cmake --build ../iree-build/

export CC=clang
export CXX=clang++
python configure_bazel.py

cd integrations/tensorflow
bazel build -c opt iree_tf_compiler:iree-import-tflite

IREE_COMPILE_PATH=${SOURCE_DIR}/iree-build/iree/tools/iree-compile

TFLITE_MODEL_DIR=${ROOT_DIR}/models/tflite
IREE_MODEL_DIR=${ROOT_DIR}/models/iree
mkdir -p ${IREE_MODEL_DIR}/cuda
mkdir -p ${IREE_MODEL_DIR}/dylib

MODEL_NAME="mobilebert_float_384_gpu"
bazel-bin/iree_tf_compiler/iree-import-tflite ${TFLITE_MODEL_DIR}/${MODEL_NAME}.tflite -o ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir
# Build for CUDA.
echo "Compiling ${MODEL_NAME}.vmfb for cuda..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=cuda \
  --iree-hal-cuda-llvm-target-arch=sm_80 \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/cuda/${MODEL_NAME}.vmfb
# Build for x86.
echo "Compiling ${MODEL_NAME}.vmfb for dylib..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-target-backends=dylib-llvm-aot \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/dylib/${MODEL_NAME}.vmfb
# Build mm4td for x86.
echo "Compiling ${MODEL_NAME}_mmt4d.vmfb for dylib..."
${IREE_COMPILE_PATH} --iree-input-type=tosa --iree-mlir-to-vm-bytecode-module \
  --iree-llvm-target-cpu-features=host \
  --iree-hal-target-backends=dylib-llvm-aot \
  --iree-llvm-debug-symbols=false \
  --iree-vm-bytecode-module-strip-source-map=true \
  --iree-vm-emit-polyglot-zip=false \
  "--iree-flow-mmt4d-target-options=arch=aarch64 features=+dotprod" \
  --iree-llvm-target-cpu-features=+dotprod \
  ${IREE_MODEL_DIR}/${MODEL_NAME}.mlir \
  --o ${IREE_MODEL_DIR}/dylib/${MODEL_NAME}_mmt4d.vmfb

cp ${SOURCE_DIR}/iree-build/iree/tools/iree-benchmark-module ${ROOT_DIR}/

# Build TFLite benchmark.
sudo apt-get install libgles2-mesa-dev

export CC=clang
export CXX=clang++

cd ${SOURCE_DIR}
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
# Select defaults and answer No to all questions.
python configure.py

bazel build -c opt --copt=-DCL_DELEGATE_NO_GL \
  --copt=-DMESA_EGL_NO_X11_HEADERS=1 \
  tensorflow/lite/tools/benchmark:benchmark_model

cp ${SOURCE_DIR}/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model ${ROOT_DIR}/

# Run benchmark.
cd ${SOURCE_DIR}/iree
python3.9 build_tools/benchmarks/comparisons/run_benchmarks.py \
  --device_name=desktop --base_dir=${ROOT_DIR} \
  --output_dir=${ROOT_DIR}/output --mode=desktop

cat ${ROOT_DIR}/output/results.csv
