# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx")
  return()
endif()

unset(IREE_HIP_TEST_COMPILER_FLAGS)
list(APPEND IREE_HIP_TEST_COMPILER_FLAGS
  "--iree-rocm-target-chip=${IREE_HIP_TEST_TARGET_CHIP}"
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    stream
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=true"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=hip
    requires-gpu-amd
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    graph
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=false"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=hip
)
