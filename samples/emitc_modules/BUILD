# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_test")
load("//build_tools/bazel:iree_c_module.bzl", "iree_c_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_c_module(
    name = "add_module",
    src = "add.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "add_module.h",
)

iree_runtime_cc_test(
    name = "add_module_test",
    srcs = ["add_module_test.cc"],
    deps = [
        ":add_module",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm",
    ],
)

iree_c_module(
    name = "import_module_a",
    src = "import_module_a.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "import_module_a.h",
)

iree_c_module(
    name = "import_module_b",
    src = "import_module_b.mlir",
    flags = [
        "--compile-mode=vm",
    ],
    h_file_output = "import_module_b.h",
)

iree_runtime_cc_test(
    name = "import_module_test",
    srcs = ["import_module_test.cc"],
    deps = [
        ":import_module_a",
        ":import_module_b",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
        "//runtime/src/iree/vm",
    ],
)
