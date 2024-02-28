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
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/abs.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/add.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/arithmetic_right_shift.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/bitwise_and.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/bitwise_or.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/bitwise_xor.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/ceil.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/clamp.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/clz.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/const.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/equal.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/exp.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/floor.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/fully_connected.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/gather.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/greater.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/greater_equal.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/if.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/log.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/logical_left_shift.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/logical_right_shift.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/logical_right_shift_16.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/matmul.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/max_pool.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/maximum.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/minimum.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/mul.mlir"
    #"${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/mul_shift.mlir" (compilation failure)
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/negate.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/pad.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/reciprocal.mlir"
    #"${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/reduce.mlir" (compilation failure)
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/reshape.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/rsqrt.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/select.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/sigmoid.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/sub.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/table.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/tanh.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/transpose.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/while.mlir"
  TARGET_BACKEND
    "rocm"
  DRIVER
    "hip"
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  INPUT_TYPE
    "tosa"
  RUNNER_ARGS
    "--hip_use_streams=true"
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-amd"
)

iree_check_single_backend_test_suite(
  NAME
    check_hip_graph
  SRCS
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/abs.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/add.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/arithmetic_right_shift.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/bitwise_and.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/bitwise_or.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/bitwise_xor.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/ceil.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/clamp.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/clz.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/const.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/equal.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/exp.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/floor.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/fully_connected.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/gather.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/greater.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/greater_equal.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/if.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/log.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/logical_left_shift.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/logical_right_shift.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/logical_right_shift_16.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/matmul.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/max_pool.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/maximum.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/minimum.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/mul.mlir"
    #"${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/mul_shift.mlir" (compilation failure)
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/negate.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/pad.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/reciprocal.mlir"
    #"${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/reduce.mlir" (compilation failure)
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/reshape.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/rsqrt.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/select.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/sigmoid.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/sub.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/table.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/tanh.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/transpose.mlir"
    "${IREE_SOURCE_DIR}/tests/e2e/tosa_ops/while.mlir"
  TARGET_BACKEND
    "rocm"
  DRIVER
    "hip"
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  INPUT_TYPE
    "tosa"
  RUNNER_ARGS
    "--hip_use_streams=false"
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-amd"
)
