# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_binary", "iree_runtime_cc_test")
load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_binary(
    name = "dynamic_library_test_library.so",
    testonly = True,
    srcs = ["dynamic_library_test_library.cc"],
    linkshared = True,
)

c_embed_data(
    name = "dynamic_library_test_library",
    testonly = True,
    srcs = [":dynamic_library_test_library.so"],
    c_file_output = "dynamic_library_test_library_embed.c",
    flatten = True,
    h_file_output = "dynamic_library_test_library_embed.h",
)

iree_runtime_cc_test(
    name = "dynamic_library_test",
    srcs = ["dynamic_library_test.cc"],
    tags = [
        # Disabled because of unknown CI failures only with bazel on the CI.
        # This passes locally and with cmake.
        # See #10034 for more information.
        "nokokoro",
    ],
    deps = [
        ":dynamic_library_test_library",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:dynamic_library",
        "//runtime/src/iree/base/internal:file_io",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
