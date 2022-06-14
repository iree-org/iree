# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda
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
    # This test depends on iree_hal_cuda_stream_command_buffer_update_buffer
    # via iree_hal_buffer_view_allocate_buffer, which is not implemented yet.
    "command_buffer_dispatch"
    # Non-push descriptor sets are not implemented in the CUDA backend yet.
    "descriptor_set"
    # Semaphores are not implemented in the CUDA backend yet.
    "semaphore_submission"
    "semaphore"
)

# Variant test suite using graph command buffers (--cuda_use_streams=0)
iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda
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
    # This test depends on iree_hal_cuda_stream_command_buffer_update_buffer
    # via iree_hal_buffer_view_allocate_buffer, which is not implemented yet.
    # "command_buffer_dispatch"
)
