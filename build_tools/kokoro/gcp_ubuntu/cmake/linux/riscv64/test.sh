#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled RISCV targets using Kokoro.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Environment variable used by the emulator and iree-compile for the
# dylib-llvm-aot bytecode codegen.
export RISCV_TOOLCHAIN_ROOT=${RISCV_RV64_LINUX_TOOLCHAIN_ROOT?}


function generate_dylib_vmfb {
  local target="${1}"; shift
  local translate_arg=(
    --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot
    --iree-llvm-embedded-linker-path=${BUILD_HOST_DIR?}/install/bin/lld
    --iree-llvm-target-triple=riscv64
    --iree-llvm-target-cpu=generic-rv64
    --iree-llvm-target-abi=lp64d
  )
  if [[ "${target}" == "mhlo" ]]; then
    translate_arg+=(
      --iree-input-type=mhlo
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
    )
  elif [[ "${target}" == "tosa" ]]; then
    local input_file="${1}"; shift
    iree-import-tflite -o "${BUILD_RISCV_DIR?}/tosa.mlir" "${input_file}"
    translate_arg+=(
      --iree-input-type=tosa
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
      "${BUILD_RISCV_DIR?}/tosa.mlir"
    )
  elif [[ "${target}" == "tosa-rvv" ]]; then
    local input_file="${1}"; shift
    iree-import-tflite -o "${BUILD_RISCV_DIR?}/tosa.mlir" "${input_file}"
    translate_arg+=(
      --iree-input-type=tosa
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c,+v"
      --riscv-v-fixed-length-vector-lmul-max=8
      --riscv-v-vector-bits-min=256
      "${BUILD_RISCV_DIR?}/tosa.mlir"
    )
  fi
  "${BUILD_HOST_DIR?}/install/bin/iree-compile" "${translate_arg[@]}" "$@"
}

generate_dylib_vmfb mhlo \
  "${ROOT_DIR?}/iree/tools/test/iree-run-module.mlir" \
  -o "${BUILD_RISCV_DIR?}/iree-run-module-llvm_aot.vmfb"

wget -P "${BUILD_RISCV_DIR?}/" https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite

generate_dylib_vmfb tosa \
  "${BUILD_RISCV_DIR?}/person_detect.tflite" \
  -o "${BUILD_RISCV_DIR?}/person_detect.vmfb"

generate_dylib_vmfb tosa-rvv \
  "${BUILD_RISCV_DIR?}/person_detect.tflite" \
  -o "${BUILD_RISCV_DIR?}/person_detect_rvv.vmfb"

${PYTHON_BIN?} "${ROOT_DIR?}/third_party/llvm-project/llvm/utils/lit/lit.py" \
  -v --path "${BUILD_HOST_DIR?}/third_party/llvm-project/llvm/bin" \
  "${ROOT_DIR?}/build_tools/kokoro/gcp_ubuntu/cmake/linux/riscv64/tests"
