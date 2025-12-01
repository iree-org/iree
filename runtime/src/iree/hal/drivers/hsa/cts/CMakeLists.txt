# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx")
  return()
endif()

unset(IREE_HSA_TEST_COMPILER_FLAGS)
list(APPEND IREE_HSA_TEST_COMPILER_FLAGS
  "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hsa
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hsa/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hsa_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HSA_TEST_COMPILER_FLAGS}
  DEPS
    iree::hal::drivers::hsa::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # TODO: implement executable reflection.
    "executable"
    # Semaphore timepoint import/export not implemented.
    "semaphore"
    # semaphore_submission requires async execution (SubmitWithWait hangs).
    "semaphore_submission"
    # Collectives not implemented.
    "command_buffer_collective"
    # These tests require features not yet implemented in the HSA backend.
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
    # Driver test hits HSA resource limits when creating many devices in sequence.
    # QueryAndCreateAvailableDevicesByOrdinal fails after QueryAndCreateAvailableDevicesByID.
    "driver"
  LABELS
    driver=hsa
    requires-gpu-amd
)

