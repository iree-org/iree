#!/bin/bash

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Compiles the given model to modules. The targets are expected to be a
# comma-separated list, e.g., --targets=vmla,vulkan,dylib
#
# If dylib target is set, the script assumes ANDROID_NDK env is also set.

PREFIX="module"
while [[ $# -gt 0 ]]; do
  token="$1"
  case $token in
    --model=*)
      MODEL=${1#*=}
      shift
      ;;
    --targets=*)
      TARGETS=${1#*=}
      shift
      ;;
    --prefix=*)
      PREFIX=${1#*=}
      shift
      ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "Must set --model flag.";
  exit 1
fi

export IREE_LLVMAOT_LINKER_PATH="${ANDROID_NDK?}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ -static-libstdc++ -O3"

IFS=',' read -ra targets_array <<< "$TARGETS"
for target in "${targets_array[@]}"
do
  echo "Compile the module for ${target}..."
  module_name="${PREFIX}-${target}.vmfb"
  case "${target}" in
    "vulkan")
      build/host/iree/tools/iree-translate \
        --iree-mlir-to-vm-bytecode-module \
        --iree-hal-target-backends=vulkan-spirv \
        ${MODEL} \
        -o "${module_name}"
      ;;
    "vmla")
      build/host/iree/tools/iree-translate \
        --iree-mlir-to-vm-bytecode-module \
        --iree-hal-target-backends=vmla \
        ${MODEL} \
        -o "${module_name}"
      ;;
    "dylib")
      build/host/iree/tools/iree-translate \
        --iree-mlir-to-vm-bytecode-module \
        --iree-hal-target-backends=dylib-llvm-aot \
        --iree-llvm-target-triple=aarch64-linux-android \
        ${MODEL} \
        -o "${module_name}"
        ;;
    *)
      echo "Not supported target"
      exit 1
      ;;
  esac
done
