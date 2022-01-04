# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "buffer_transfer",
    srcs = ["buffer_transfer.c"],
    hdrs = ["buffer_transfer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//iree/base",
        "//iree/base:tracing",
        "//iree/hal",
    ],
)

cc_library(
    name = "deferred_command_buffer",
    srcs = ["deferred_command_buffer.c"],
    hdrs = ["deferred_command_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":resource_set",
        "//iree/base",
        "//iree/base:tracing",
        "//iree/base/internal:arena",
        "//iree/hal",
    ],
)

cc_library(
    name = "resource_set",
    srcs = ["resource_set.c"],
    hdrs = ["resource_set.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//iree/base",
        "//iree/base:tracing",
        "//iree/base/internal:arena",
        "//iree/hal",
    ],
)

cc_test(
    name = "resource_set_benchmark",
    srcs = ["resource_set_benchmark.c"],
    deps = [
        ":resource_set",
        "//iree/base",
        "//iree/base/internal:prng",
        "//iree/hal",
        "//iree/testing:benchmark",
    ],
)

cc_test(
    name = "resource_set_test",
    srcs = ["resource_set_test.cc"],
    deps = [
        ":resource_set",
        "//iree/base",
        "//iree/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
