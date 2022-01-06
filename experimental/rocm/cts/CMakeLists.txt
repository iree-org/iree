# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    rocm
  DRIVER_REGISTRATION_HDR
    "experimental/rocm/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_rocm_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  DEPS
    experimental::rocm::registration
)
