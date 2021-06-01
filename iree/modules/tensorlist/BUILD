# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
# absl has thread dependency.
# TODO(#3848): Remove this condition.
if(NOT ${IREE_ENABLE_THREADING})
  return()
endif()
""",
    inline = True,
)

cc_library(
    name = "native_module",
    srcs = ["native_module.cc"],
    hdrs = ["native_module.h"],
    deps = [
        "//iree/base",
        "//iree/base:status",
        "//iree/hal",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:cc",
        "@com_google_absl//absl/types:span",
    ],
)

iree_cmake_extra_content(
    content = """
if (NOT ${IREE_BUILD_COMPILER})
  return()
endif()
""",
    inline = True,
)

iree_bytecode_module(
    name = "tensorlist_test_module",
    testonly = True,
    src = "tensorlist_test.mlir",
    c_identifier = "iree_tensorlist_test_module",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=dylib-llvm-aot",
    ],
)

cc_test(
    name = "tensorlist_test",
    srcs = ["tensorlist_test.cc"],
    deps = [
        ":native_module",
        ":tensorlist_test_module_c",
        "//iree/base",
        "//iree/base:logging",
        "//iree/hal",
        "//iree/hal/vmvx/registration",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:cc",
        "@com_google_absl//absl/types:span",
    ],
)
