# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unset(COMPILER_FLAGS)
if(IREE_ROCM_LINK_BC)
  list(APPEND COMPILER_FLAGS "--iree-rocm-link-bc=true")
else()
  list(APPEND COMPILER_FLAGS "--iree-rocm-link-bc=false")
endif()
list(APPEND COMPILER_FLAGS
  "--iree-rocm-target-chip=${IREE_ROCM_TARGET_CHIP}"
  "--iree-rocm-bc-dir=${IREE_ROCM_BC_DIR}")

iree_hal_cts_test_suite(
  DRIVER_NAME
    rocm
  VARIANT_SUFFIX
    direct
  DRIVER_REGISTRATION_HDR
    "experimental/rocm/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_rocm_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--rocm_use_graph=false"
  COMPILER_FLAGS
    ${COMPILER_FLAGS}
  DEPS
    iree::experimental::rocm::registration
  EXCLUDED_TESTS
    # This test depends on iree_hal_rocm_direct_command_buffer_update_buffer
    # via iree_hal_buffer_view_allocate_buffer_copy, which is not implemented yet.
    "command_buffer_dispatch"
    # HAL event is unimplemented for now.
    "event"
    # Semaphores are not implemented in the ROCm backend yet.
    "semaphore_submission"
    "semaphore"
  LABELS
    driver=rocm
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    rocm
  VARIANT_SUFFIX
    graph 
  DRIVER_REGISTRATION_HDR
    "experimental/rocm/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_rocm_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--rocm_use_graph=true"
  COMPILER_FLAGS
    ${COMPILER_FLAGS}
  DEPS
    iree::experimental::rocm::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # Semaphores are not implemented in the ROCm backend yet.
    "semaphore_submission"
    "semaphore"
  LABELS
    driver=rocm
)
