# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# These tests perform linking via the Compiler API, which is only supported
# in bundled-LLVM builds at the moment (#14086).
if(NOT IREE_BUILD_BUNDLED_LLVM OR NOT IREE_ENABLE_COLLECTIVE_RUNTIME_TESTS)
  return()
endif()

if(IREE_TARGET_BACKEND_CUDA AND IREE_HAL_DRIVER_CUDA)
  iree_py_test(
    NAME
      collectives_test_single_gpu
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "SingleRank"
      "--target_backend=cuda"
      "--driver=cuda"
    LABELS
      "requires-gpu-nvidia"
      "driver=cuda"
  )

  iree_py_test(
    NAME
      collectives_test_multi_gpu
    SRCS
      "collectives_test.py"
    ARGS
      "-k" "TwoRanks"
      "--target_backend=cuda"
      "--driver=cuda"
    LABELS
      "requires-gpu-nvidia"
      "driver=cuda"
      # The NCCL collectives backend requires 1 GPU per rank.
      # To properly test for 2 ranks we need 2 GPUs.
      "requires-multiple-devices"
  )
endif()
