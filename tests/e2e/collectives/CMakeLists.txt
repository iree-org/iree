# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# These tests perform linking via the Compiler API, which is only supported
# in bundled-LLVM builds at the moment (#14086).
if(NOT IREE_BUILD_BUNDLED_LLVM)
  return()
endif()


set(CUDA_COMMON_ARGS
  "--target_backend=cuda"
  "--driver=cuda"
  "--iree_compiler_args=--iree-hal-cuda-llvm-target-arch=sm_53"
)

set(CUDA_COMMON_LABELS
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
    ${CUDA_COMMON_ARGS}
  LABELS
    ${CUDA_COMMON_LABELS}
)

iree_py_test(
  NAME
    collectives_test_cuda_2_gpus
  SRCS
    "collectives_test.py"
  ARGS
    "-k" "TwoRanks"
    ${CUDA_COMMON_ARGS}
  LABELS
    ${CUDA_COMMON_LABELS}
    # The NCCL collectives backend requires 1 GPU per rank.
    # To properly test for 2 ranks we need 2 GPUs.
    "requires-multiple-devices"
    "requires-2-devices"
)

iree_py_test(
  NAME
    collectives_test_cuda_4_gpus
  SRCS
    "collectives_test.py"
  ARGS
    "-k" "FourRanks"
    ${CUDA_COMMON_ARGS}
  LABELS
    ${CUDA_COMMON_LABELS}
    "requires-multiple-devices"
    "requires-4-devices"
)


if(IREE_HIP_TEST_TARGET_CHIP)
  set(HIP_COMMON_ARGS
    "--target_backend=rocm"
    "--driver=hip"
    "--iree_compiler_args=--iree-rocm-target-chip=${IREE_HIP_TEST_TARGET_CHIP}"
  )

  set(HIP_COMMON_LABELS
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
      ${HIP_COMMON_ARGS}
    LABELS
      ${HIP_COMMON_LABELS}
  )

  iree_py_test(
    NAME
      collectives_test_hip_2_gpus
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "TwoRanks"
      ${HIP_COMMON_ARGS}
    LABELS
      ${HIP_COMMON_LABELS}
      # The RCCL collectives backend requires 1 GPU per rank.
      # To properly test for 2 ranks we need 2 GPUs.
      "requires-multiple-devices"
      "requires-2-devices"
  )

  iree_py_test(
    NAME
      collectives_test_hip_4_gpus
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "FourRanks"
      ${HIP_COMMON_ARGS}
    LABELS
      ${HIP_COMMON_LABELS}
      "requires-multiple-devices"
      "requires-4-devices"
  )
endif()
