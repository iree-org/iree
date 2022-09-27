#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled RISCV 64-bit Linux targets.

set -xeuo pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"

RISCV_ARCH="${RISCV_ARCH:-rv64}"

if [[ "${RISCV_ARCH}" == "rv64" ]]; then
  IREE_IMPORT_TFLITE_BIN="${IREE_IMPORT_TFLITE_BIN:-iree-import-tflite}"
  LLVM_BIN_DIR="${LLVM_BIN_DIR}"
fi

# Environment variable used by the emulator and iree-compile for the
# llvm-cpu bytecode codegen.
export RISCV_TOOLCHAIN_ROOT="${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}"

function generate_llvm_cpu_vmfb {
  local target="${1}"; shift
  local compile_args=(
    --iree-hal-target-backends=llvm-cpu
    --iree-llvm-embedded-linker-path="${IREE_HOST_BINARY_ROOT}/bin/lld"
    --iree-llvm-target-triple=riscv64
    --iree-llvm-target-cpu=generic-rv64
    --iree-llvm-target-abi=lp64d
  )
  if [[ "${target}" == "mhlo" ]]; then
    compile_args+=(
      --iree-input-type=mhlo
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
    )
  elif [[ "${target}" == "tosa" ]]; then
    local input_file="${1}"; shift
    "${IREE_IMPORT_TFLITE_BIN}" -o "${BUILD_RISCV_DIR}/tosa.mlir" "${input_file}"
    compile_args+=(
      --iree-input-type=tosa
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
      "${BUILD_RISCV_DIR}/tosa.mlir"
    )
  elif [[ "${target}" == "tosa-rvv" ]]; then
    local input_file="${1}"; shift
    "${IREE_IMPORT_TFLITE_BIN}" -o "${BUILD_RISCV_DIR}/tosa.mlir" "${input_file}"
    compile_args+=(
      --iree-input-type=tosa
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c,+v"
      --riscv-v-fixed-length-vector-lmul-max=8
      --riscv-v-vector-bits-min=512
      "${BUILD_RISCV_DIR}/tosa.mlir"
    )
  fi
  "${IREE_HOST_BINARY_ROOT}/bin/iree-compile" "${compile_args[@]}" "$@"
}

if [[ "${RISCV_ARCH}" == "rv64" ]]; then
  generate_llvm_cpu_vmfb mhlo \
    "${ROOT_DIR}/tools/test/iree-run-module.mlir" \
    -o "${BUILD_RISCV_DIR}/iree-run-module-llvm_cpu.vmfb"

  wget -P "${BUILD_RISCV_DIR}/" "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"

  generate_llvm_cpu_vmfb tosa \
    "${BUILD_RISCV_DIR}/person_detect.tflite" \
    -o "${BUILD_RISCV_DIR}/person_detect.vmfb"

  generate_llvm_cpu_vmfb tosa-rvv \
    "${BUILD_RISCV_DIR}/person_detect.tflite" \
    -o "${BUILD_RISCV_DIR}/person_detect_rvv.vmfb"

  ${PYTHON_BIN} "${ROOT_DIR}/third_party/llvm-project/llvm/utils/lit/lit.py" \
    -v --path "${LLVM_BIN_DIR}" "${ROOT_DIR}/tests/riscv64"
fi

ctest_args=(
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

declare -a label_exclude_args=(
  "^nokokoro$"
  "^driver=vulkan$"
  "^driver=cuda$"
  "^vulkan_uses_vk_khr_shader_float16_int8$"
  "^requires-filesystem$"
  "^requires-dtz$"
)

# Excluding mobilebert, fp16, and lowering_config regression
# tests for now.
# TODO(#10462): Investigate the lowering_config test issue.
declare -a test_exclude_args=(
  "bert"
  "fp16"
  "regression_llvm-cpu_lowering_config"
)

# Test runtime unit tests
runtime_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
runtime_ctest_args=(
  "--test-dir ${BUILD_RISCV_DIR}/runtime/"
  ${ctest_args[@]}
  "--label-exclude ${runtime_label_exclude_regex}"
)
echo "******** Running runtime CTest ********"
ctest ${runtime_ctest_args[@]}

if [[ "${RISCV_ARCH}" == "rv32-linux" ]]; then
  # mhlo.power is also disabled because musl math library is not compiled for
  # 32-bit.
  test_exclude_args+=(
    "xla.*llvm-cpu.*pow"
  )
fi

tests_label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
tests_exclude_regex="($(IFS="|" ; echo "${test_exclude_args[*]}"))"
test_ctest_args=(
  "--test-dir ${BUILD_RISCV_DIR}/tests/e2e"
  ${ctest_args[@]}
  "--label-exclude ${tests_label_exclude_regex}"
  "--exclude-regex ${tests_exclude_regex}"
)
echo "******** Running e2e CTest ********"
ctest  ${test_ctest_args[@]}
