# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    webgpu
  DRIVER_REGISTRATION_HDR
    "experimental/webgpu/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_webgpu_driver_module_register"
  COMPILER_TARGET_BACKEND
    "webgpu"
  EXECUTABLE_FORMAT
    "\"webgpu-wgsl-fb\""
  DEPS
    iree::experimental::webgpu::registration
)
