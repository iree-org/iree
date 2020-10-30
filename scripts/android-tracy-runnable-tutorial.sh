#!/bin/bash

# Stop on error.
set -e

function print_status {
  echo -e "\e[96m$@\e[39m"
}

# Environment variables. They can be set manually, or will use the following defaults.
# IREE_ROOT and ANDROID_NDK are empty by default, must be defined by the user.

: ${IREE_ROOT:=""}
: ${IREE_BUILD_ANDROID:="$HOME/iree-build-android"}
: ${TRACY_ROOT:="$HOME/tracy"}
: ${PYTHON_BIN:=python3}
: ${CC:=clang}
: ${CXX:=clang++}
: ${IREE_INPUT_MLIR:=/tmp/iree/modules/MobileBertSquad/iree_input.mlir}
: ${ANDROID_NDK:=""}
: ${IREE_LLVMAOT_LINKER_PATH:="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++ -static-libstdc++ -O3"}

# Validation of the environment variables.

if [ -z "${IREE_ROOT}" ]
then
  print_status "Please define IREE_ROOT to point to your IREE git clone."
  print_status "Example:"
  print_status "  IREE_ROOT=\$HOME/iree $0"
  exit 1
fi

if [ ! -f "${IREE_ROOT}/iree/base/CMakeLists.txt" ]
then
  print_status "FATAL: bad IREE_ROOT (${IREE_ROOT})."
  exit 1
fi

if [ -z "${ANDROID_NDK}" ]
then
  print_status "Please define ANDROID_NDK to point to your Android NDK."
  print_status "Example:"
  print_status "  ANDROID_NDK=\$HOME/android-ndk-r21d $0"
  exit 1
fi

if [ ! -d "${ANDROID_NDK}/toolchains/llvm" ]
then
  print_status "FATAL: bad ANDROID_NDK (${ANDROID_NDK})."
  exit 1
fi

if ! $CC --version 1>/dev/null 2>/dev/null
then
  print_status "FATAL: Install $CC and set CC to it."
  exit 1
fi

if ! $CXX --version 1>/dev/null 2>/dev/null
then
  print_status "FATAL: Install $CXX and set CXX to it."
  exit 1
fi

if ! $IREE_LLVMAOT_LINKER_PATH --version 1>/dev/null 2>/dev/null
then
  print_status "Bad IREE_LLVMAOT_LINKER_PATH (${IREE_LLVMAOT_LINKER_PATH}). Rerun with it correctly set."
  exit 1
fi

print_status "Running with the following environment variables:"
print_status "ANDROID_NDK=${ANDROID_NDK}"
print_status "IREE_LLVMAOT_LINKER_PATH=${IREE_LLVMAOT_LINKER_PATH}"
print_status "IREE_ROOT=${IREE_ROOT}"
print_status "IREE_BUILD_ANDROID=${IREE_BUILD_ANDROID}"
print_status "TRACY_ROOT=${TRACY_ROOT}"
print_status "PYTHON_BIN=${PYTHON_BIN}"
print_status "CC=${CC}"
print_status "CXX=${CXX}"
print_status "IREE_INPUT_MLIR=${IREE_INPUT_MLIR}"
echo

print_status "Ensuring that we have Tracy source code..."
if [ ! -d ${TRACY_ROOT}/profiler/build/unix/ ]
then
  print_status "Tracy not found at ${TRACY_ROOT}. Either set TRACY_ROOT, or we're going to clone the git repository now."
  read -p "Press the return key..."
  git clone https://github.com/wolfpld/tracy "${TRACY_ROOT}"
fi

echo

print_status "Ensuring that the Tracy profiler is built..."
if [ ! -x "${TRACY_ROOT}/profiler/build/unix/Tracy-release" ]
then
  print_status "Checking Tracy dependencies - assuming Debian."
  TRACY_DEPS="libcapstone-dev libtbb-dev libglfw3-dev libfreetype6-dev libgtk-3-dev"
  TRACY_DEPS_COUNT=5
  TRACY_DEPS_INSTALLED="$(apt list $TRACY_DEPS) 2>/dev/null | grep installed | wc -l"
  if [ $TRACY_DEPS_INSTALLED != $TRACY_DEPS_COUNT ]
  then
    print_status "Installing dependencies now - assuming Debian."
    sudo apt install $TRACY_DEPS
  fi
  make -C "${TRACY_ROOT}/profiler/build/unix" -j12 release
fi

echo

print_status "Ensuring that we have the input MLIR file..."
if [ ! -f "${IREE_INPUT_MLIR}" ]
then
  print_status "Set IREE_INPUT_MLIR to point to some iree input MLIR file. Not found at current value ${IREE_INPUT_MLIR}."
  if [ "${IREE_INPUT_MLIR}" == "/tmp/iree/modules/MobileBertSquad/iree_input.mlir" ]
  then
    print_status "Okay, we actually know how to generate that file, ${IREE_INPUT_MLIR}, but it will take a while."
    print_status "Press the return key to continue..."
    pushd "${IREE_ROOT}"
    scripts/get_e2e_artifacts.py --test_suites=mobile_bert_squad_tests
    popd
  fi
fi

if [ ! -f "${IREE_INPUT_MLIR}" ]
then
  print_status "FATAL: we should have ${IREE_INPUT_MLIR} by that point."
  exit 1
fi

echo

print_status "Building IREE for Android in ${IREE_BUILD_ANDROID}..."

mkdir -p "${IREE_BUILD_ANDROID}"
pushd "${IREE_BUILD_ANDROID}"

cmake -G Ninja ../iree \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=OFF  \
  -DIREE_HOST_C_COMPILER=`which clang` \
  -DIREE_HOST_CXX_COMPILER=`which clang++` \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DIREE_ENABLE_RUNTIME_TRACING=ON \
  -DIREE_ENABLE_ASAN=ON

cmake --build .

popd
echo

print_status "Compiling the input MLIR file into a IREE module..."

IREE_COMPILED_MODULE=/tmp/android_module.fbvm
IREE_LOG=/tmp/iree-translate.log

rm -rf "${IREE_COMPILED_MODULE}"

IREE_LLVMAOT_LINKER_PATH="${IREE_LLVMAOT_LINKER_PATH}" \
  iree-build-android/host/bin/iree-translate \
    --iree-hal-target-backends=dylib-llvm-aot \
    --iree-mlir-to-vm-bytecode-module \
    --iree-llvm-target-triple=aarch64-linux-android \
    /tmp/iree/modules/MobileBertSquad/iree_input.mlir \
    -o /tmp/android_module.fbvm \
    2>"${IREE_LOG}"

if [ ! -f "${IREE_COMPILED_MODULE}" ]
then
  print_status "iree-translate failed to produce ${IREE_COMPILED_MODULE}. Log saved in ${IREE_LOG}. First few lines:"
  # The whole log might be enormous if it contains a big MLIR dump.
  head -n10 "${IREE_LOG}"
  print_status "tip: check if IREE_LLVMAOT_LINKER_PATH was correctly set."
  exit 1
fi

echo

print_status "Pushing the compiled module to the device..."
adb push /tmp/android_module.fbvm /data/local/tmp
echo

print_status "Pushing the IREE benchmarking program to the device..."
adb push iree-build-android/iree/tools/iree-benchmark-module /data/local/tmp
echo

print_status "Setting up TCP port forwarding to let Tracy connect with the benchmark running on the device..."
adb forward tcp:8086 tcp:8086
echo

print_status "Now you can launch the Tracy UI in another shell and hit \"Connect\", while we run the benchmark on the device in this shell..."
print_status "Run this command in another shell:"
print_status "  ${TRACY_ROOT}/profiler/build/unix/Tracy-release"
echo

print_status "Running the benchmark... hit Ctrl-C to terminate it after Tracy is done with it."
# `TRACY_NO_EXIT=1` is to prevent it from exiting at the end: that's needed for profiling
# short-running tasks.
# `taskset 80` selects which CPU core to run on. On Pixel4, `taskset 80` gives the biggest
# core, which can get some reproducibility, as long as we don't run into thermal issues. 
# `taskset 0f` would give the little cores, avoiding thermal issues but running slower and
# requiring different optimization work to be efficient on.
# `--driver=dylib` is to use the LLVM AOT generated code backend.
adb shell \
  TRACY_NO_EXIT=1 \
    taskset 80 \
      data/local/tmp/iree-benchmark-module \
        --driver=dylib \
        --module_file=/data/local/tmp/android_module.fbvm \
        --function_inputs='1x384xi32,1x384xi32,1x384xi32' \
        --entry_function=serving_default
