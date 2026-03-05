# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CTS test binaries for the Metal HAL driver (hand-written CMake, no Bazel).

iree_cc_library(
  NAME
    backends
  SRCS
    "backends.cc"
  DEPS
    iree::hal
    iree::hal::cts::util::registry
    iree::hal::drivers::metal::registration
  TESTONLY
  ALWAYSLINK
  PUBLIC
)

iree_hal_cts_testdata(
  FORMAT_NAME
    metal_spirv
  TARGET_DEVICE
    "metal"
  IDENTIFIER
    "iree_cts_testdata_metal_spirv"
  BACKEND_NAME
    "metal"
  FORMAT_STRING
    [=["MTLE"]=]
  TESTDATA_DIR
    "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/testdata"
)

iree_hal_cts_test_suite(
  BACKENDS_LIB
    ::backends
  TESTDATA_LIBS
    ::testdata_metal_spirv_lib
  LABELS
    "driver=metal"
)
