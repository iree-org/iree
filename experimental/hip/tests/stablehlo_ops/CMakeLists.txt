# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unset(IREE_HIP_TEST_COMPILER_FLAGS)
if(IREE_ROCM_LINK_BC)
  list(APPEND IREE_HIP_TEST_COMPILER_FLAGS "--iree-rocm-link-bc=true")
else()
  list(APPEND IREE_HIP_TEST_COMPILER_FLAGS "--iree-rocm-link-bc=false")
endif()
list(APPEND IREE_HIP_TEST_COMPILER_FLAGS
  "--iree-rocm-target-chip=${IREE_ROCM_TARGET_CHIP}"
  "--iree-rocm-bc-dir=${IREE_ROCM_BC_DIR}")

iree_check_single_backend_test_suite(
  NAME
    check_hip_stream
  SRCS
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/abs.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/add.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/batch_norm_inference.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/bitcast_convert.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/broadcast.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/broadcast_add.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/broadcast_in_dim.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/clamp.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/compare.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/complex.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/concatenate.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/constant.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/convert.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/convolution.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/cosine.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/divide.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/dot.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/dot_bf16.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/dot_general.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/dynamic_slice.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/dynamic_update_slice.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/exponential.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/exponential_fp16.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/exponential_minus_one.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/fft.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/finite.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/floor.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/gather.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/householder.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/iota.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/log.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/log_plus_one.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/maximum.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/minimum.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/multiply.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/negate.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/pad.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/philox.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/pow.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/reduce.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/reduce_window.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/remainder.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/reshape.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/reverse.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/rng_normal.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/rng_uniform.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/round.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/rsqrt.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/scatter.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/scatter_dynamic.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/select.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/sine.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/slice.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/sort.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/sqrt.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/subtract.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/tanh.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/three_fry.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/torch_index_select.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/transpose.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/stablehlo_ops/while.mlir"
  TARGET_BACKEND
    "rocm"
  DRIVER
    "hip"
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  INPUT_TYPE
    "stablehlo"
  RUNNER_ARGS
    "--hip_use_streams=true"
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-amd"
)
