# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Common types and utilities used in the IREE codebase.

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library", "iree_runtime_cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "base",
    srcs = [
        "allocator.c",
        "allocator.h",
        "assert.h",
        "bitfield.c",
        "bitfield.h",
        "loop.c",
        "loop.h",
        "loop_inline.c",
        "loop_inline.h",
        "status.c",
        "status.h",
        "status_cc.h",
        "string_builder.c",
        "string_builder.h",
        "string_view.c",
        "string_view.h",
        "time.c",
        "time.h",
        "wait_source.c",
        "wait_source.h",
    ],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":core_headers",
        ":tracing",
    ],
)

iree_runtime_cc_test(
    name = "bitfield_test",
    srcs = ["bitfield_test.cc"],
    deps = [
        ":base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "loop_inline_test",
    srcs = [
        "loop_inline_test.cc",
    ],
    deps = [
        ":base",
        ":loop_test_hdrs",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_library(
    name = "loop_test_hdrs",
    testonly = True,
    hdrs = [
        "loop_test.h",
    ],
    deps = [
        ":base",
        ":tracing",
        "//runtime/src/iree/base/internal:wait_handle",
        "//runtime/src/iree/testing:gtest",
    ],
)

iree_runtime_cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "string_builder_test",
    srcs = ["string_builder_test.cc"],
    deps = [
        ":base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "string_view_test",
    srcs = ["string_view_test.cc"],
    deps = [
        ":base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Core headers (platform detection, compiler compat, etc)
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "core_headers",
    hdrs = [
        "alignment.h",
        "attributes.h",
        "config.h",
        "target_platform.h",
    ],
)

iree_runtime_cc_library(
    name = "target_platform",
    hdrs = ["target_platform.h"],
)

#===------------------------------------------------------------------------===#
# Internal IREE C++ wrappers and utilities
#===------------------------------------------------------------------------===#

iree_runtime_cc_library(
    name = "loop_sync",
    srcs = ["loop_sync.c"],
    hdrs = ["loop_sync.h"],
    deps = [
        ":base",
        ":tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:wait_handle",
    ],
)

iree_runtime_cc_test(
    name = "loop_sync_test",
    srcs = [
        "loop_sync_test.cc",
    ],
    deps = [
        ":base",
        ":loop_sync",
        ":loop_test_hdrs",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_library(
    name = "tracing",
    srcs = ["tracing.cc"],
    hdrs = ["tracing.h"],
    deps = [
        ":core_headers",
        "@tracy_client//:runtime_impl",
    ],
)
