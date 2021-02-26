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
# comma-separated list, e.g., --targets=vmla,vulkan-spirv,dylib-llvm-aot
#
# The scripts is used for benchmarking automation, and it assumes:
#   1) ANDROID_NDK env is set.
#   2) IREE is built for the host in `build-host`, e.g. build with
#      build_tools/cmake/build_android.sh script.

set -e
set -o pipefail
set -o xtrace

prefix="module"
while [[ $# -gt 0 ]]; do
  token="$1"
  case $token in
    --model=*)
      model=${1#*=}
      shift
      ;;
    --targets=*)
      targets=${1#*=}
      shift
      ;;
    --prefix=*)
      prefix=${1#*=}
      shift
      ;;
  esac
done

if [[ -z "${model}" ]]; then
  echo "Must set --model flag.";
  exit 1
fi

IFS=',' read -ra targets_array <<< "$targets"
for target in "${targets_array[@]}"
do
  echo "Compile the module for ${target}..."
  module_name="${prefix}-${target}.vmfb"
  extra_flags=()
  case "${target}" in
    "dylib-llvm-aot")
      # Note: this should match the phones the benchmarks will be run on.
      # We could pass it in as an option or query the min/max in $ANDROID_NDK,
      # but that will only be worth it if this script is used in different
      # environments or for benchmarking on different phones.
      extra_flags+=('--iree-llvm-target-triple=aarch64-none-linux-android29')
      ;;
    *)
      ;;
  esac
  build-host/iree/tools/iree-translate \
    --iree-mlir-to-vm-bytecode-module \
    --iree-hal-target-backends="${target}" \
    "${extra_flags[@]}" \
    "${model}" \
    -o "${module_name}"
done
