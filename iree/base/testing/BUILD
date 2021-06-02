# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
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

cc_test(
    name = "dynamic_library_test",
    srcs = ["dynamic_library_test.cc"],
    deps = [
        ":dynamic_library_test_library",
        "//iree/base",
        "//iree/base:logging",
        "//iree/base/internal:dynamic_library",
        "//iree/base/internal:file_io",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
