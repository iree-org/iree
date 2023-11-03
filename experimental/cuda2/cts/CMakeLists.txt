# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda2
  VARIANT_SUFFIX
    graph
  DRIVER_REGISTRATION_HDR
    "experimental/cuda2/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_cuda2_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--cuda2_use_streams=false"
  DEPS
    iree::experimental::cuda2::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=cuda2
    requires-gpu-nvidia
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda2
  VARIANT_SUFFIX
    stream
  DRIVER_REGISTRATION_HDR
    "experimental/cuda2/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_cuda2_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--cuda2_use_streams=true"
  DEPS
    iree::experimental::cuda2::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=cuda2
    requires-gpu-nvidia
)
