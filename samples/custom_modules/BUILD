# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "module",
    srcs = ["module.cc"],
    hdrs = ["module.h"],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:cc",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:cc",
    ],
)

iree_cmake_extra_content(
    content = """
if(NOT IREE_TARGET_BACKEND_VMVX OR
   NOT IREE_HAL_DRIVER_VMVX)
  return()
endif()
""",
)

iree_bytecode_module(
    name = "custom_modules_test_module",
    src = "custom_modules_test.mlir",
    c_identifier = "iree_samples_custom_modules_test_module",
    flags = [
        "--iree-mlir-to-vm-bytecode-module",
        "--iree-hal-target-backends=vmvx",
    ],
    translate_tool = "//samples/custom_modules/dialect:custom-translate",
)

cc_test(
    name = "custom_modules_test",
    srcs = ["custom_modules_test.cc"],
    deps = [
        ":custom_modules_test_module_c",
        ":module",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:logging",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/vmvx/registration",
        "//runtime/src/iree/modules/hal",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm",
        "//runtime/src/iree/vm:bytecode_module",
        "//runtime/src/iree/vm:cc",
    ],
)
