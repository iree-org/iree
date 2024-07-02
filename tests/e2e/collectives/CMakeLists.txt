# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_COMPILER OR NOT IREE_ENABLE_COLLECTIVE_RUNTIME_TESTS)
  return()
endif()

if(IREE_TARGET_BACKEND_CUDA AND IREE_HAL_DRIVER_CUDA)

  set(COMMON_ARGS
    "--target_backend=cuda"
    "--driver=cuda"
    "--iree_compiler_args=--iree-hal-cuda-llvm-target-arch=sm_53"
  )

  set(COMMON_LABELS
    "requires-gpu-nvidia"
    "driver=cuda"
  )

  iree_py_test(
    NAME
      collectives_test_cuda_1_gpu
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "SingleRank"
      ${COMMON_ARGS}
    LABELS
      ${COMMON_LABELS}
  )

  iree_py_test(
    NAME
      collectives_test_cuda_2_gpus
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "TwoRanks"
      ${COMMON_ARGS}
    LABELS
      ${COMMON_LABELS}
      # The NCCL collectives backend requires 1 GPU per rank.
      # To properly test for 2 ranks we need 2 GPUs.
      "requires-multiple-devices"
  )

  iree_py_test(
    NAME
      collectives_test_cuda_4_gpus
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "FourRanks"
      ${COMMON_ARGS}
    LABELS
      ${COMMON_LABELS}
      "requires-multiple-devices"
  )
endif()

if(IREE_TARGET_BACKEND_ROCM AND IREE_HAL_DRIVER_HIP AND IREE_HIP_TEST_TARGET_CHIP)
  set(COMMON_ARGS
    "--target_backend=rocm"
    "--driver=hip"
    "--iree_compiler_args=--iree-rocm-target-chip=${IREE_HIP_TEST_TARGET_CHIP}"
  )

  set(COMMON_LABELS
    "requires-gpu-amd"
    "driver=hip"
  )

  iree_py_test(
    NAME
      collectives_test_hip_1_gpu
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "SingleRank"
      ${COMMON_ARGS}
    LABELS
      ${COMMON_LABELS}
  )

  iree_py_test(
    NAME
      collectives_test_hip_2_gpus
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "TwoRanks"
      ${COMMON_ARGS}
    LABELS
      ${COMMON_LABELS}
      # The NCCL collectives backend requires 1 GPU per rank.
      # To properly test for 2 ranks we need 2 GPUs.
      "requires-multiple-devices"
  )

  iree_py_test(
    NAME
      collectives_test_hip_4_gpus
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "FourRanks"
      ${COMMON_ARGS}
    LABELS
      ${COMMON_LABELS}
      "requires-multiple-devices"
  )
endif()
