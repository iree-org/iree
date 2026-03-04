# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CTS2 test binaries for the WebGPU HAL driver (hand-written CMake, no Bazel).
# Non-executable tests only — the webgpu-spirv compiler backend is not
# available in most build configurations.

iree_cc_library(
  NAME
    backends
  SRCS
    "backends.cc"
  DEPS
    iree::hal
    iree::hal::cts2::util::registry
    iree::experimental::webgpu::registration
  TESTONLY
  ALWAYSLINK
  PUBLIC
)

iree_hal_cts2_test_suite(
  BACKENDS_LIB
    ::backends
  LABELS
    "driver=webgpu"
)
