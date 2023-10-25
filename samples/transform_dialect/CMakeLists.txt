# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV)
  return()
endif()

iree_lit_test_suite(
  NAME
    transform_example
  SRCS
    "example_module.mlir"
  DATA
    "transform_library.mlir"
  TOOLS
    FileCheck
    iree-compile
)
