# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda1
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/cuda/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_cuda_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  DEPS
    iree::hal::drivers::cuda::registration
  EXCLUDED_TESTS
    # Semaphores are not fully implemented in the CUDA backend yet.
    "semaphore"
  LABELS
    driver=cuda
    requires-gpu-nvidia
)

# Variant test suite using graph command buffers (--cuda_use_streams=0)
iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda1
  VARIANT_SUFFIX
    graph
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/cuda/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_cuda_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--cuda_use_streams=0"
  DEPS
    iree::hal::drivers::cuda::registration
  INCLUDED_TESTS
    "command_buffer"
    "command_buffer_dispatch"
  LABELS
    driver=cuda
    requires-gpu-nvidia
)
