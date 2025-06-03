# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx")
  return()
endif()

unset(IREE_AMDGPU_TEST_COMPILER_FLAGS)
list(APPEND IREE_AMDGPU_TEST_COMPILER_FLAGS
  "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    amdgpu
  VARIANT_SUFFIX
    all
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/amdgpu/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_amdgpu_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  COMPILER_TARGET_DEVICE
    "amdgpu"
  EXECUTABLE_FORMAT
    "\"amdgcn-amd-amdhsa--${IREE_HIP_TEST_TARGET_CHIP}\""
  COMPILER_FLAGS
    ${IREE_AMDGPU_TEST_COMPILER_FLAGS}
  # ARGS
  #   "--..."
  DEPS
    iree::hal::drivers::amdgpu::registration
  EXCLUDED_TESTS
    "allocator"
    "buffer_mapping"
    "command_buffer"
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_dispatch_constants"
    "command_buffer_fill_buffer"
    "command_buffer_update_buffer"
    # HAL event is unimplemented for now.
    "event"
    "file"
  LABELS
    driver=amdgpu
    requires-gpu-amd
)
