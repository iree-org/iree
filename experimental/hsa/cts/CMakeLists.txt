# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unset(IREE_HSA_TEST_COMPILER_FLAGS)
list(APPEND IREE_HSA_TEST_COMPILER_FLAGS
  "--iree-hip-target=${IREE_HIP_TARGET_CHIP}"
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hsa
  VARIANT_SUFFIX
    queue
  DRIVER_REGISTRATION_HDR
    "experimental/hsa/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hsa_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HSA_TEST_COMPILER_FLAGS}
  ARGS
    ""
  DEPS
    iree::experimental::hsa::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=hsa
)
