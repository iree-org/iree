# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    level_zero
  DRIVER_REGISTRATION_HDR
    "experimental/level_zero/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_level_zero_driver_module_register"
  COMPILER_TARGET_BACKEND
    "opencl-spirv"
  EXECUTABLE_FORMAT
    "\"ZERO\""
  DEPS
    iree::experimental::level_zero::registration
  EXCLUDED_TESTS
    # This test depends on iree_hal_level_zero_direct_command_buffer_update_buffer
    # via iree_hal_buffer_view_allocate_buffer, which is not implemented yet.
    "command_buffer_dispatch"
    # Non-push descriptor sets are not implemented in the level_zero backend yet.
    "descriptor_set"
    # Semaphores are not implemented in the level_zero backend yet.
    "semaphore_submission"
    "semaphore"
)
