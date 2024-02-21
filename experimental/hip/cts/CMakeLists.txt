# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unset(IREE_HIP_TEST_COMPILER_FLAGS)
if(IREE_ROCM_LINK_BC)
  list(APPEND IREE_HIP_TEST_COMPILER_FLAGS "--iree-rocm-link-bc=true")
else()
  list(APPEND IREE_HIP_TEST_COMPILER_FLAGS "--iree-rocm-link-bc=false")
endif()
list(APPEND IREE_HIP_TEST_COMPILER_FLAGS
  "--iree-rocm-target-chip=${IREE_ROCM_TARGET_CHIP}"
  "--iree-rocm-bc-dir=${IREE_ROCM_BC_DIR}")

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    stream
  DRIVER_REGISTRATION_HDR
    "experimental/hip/registration/driver_module.h"
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
    iree::experimental::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=hip
)
