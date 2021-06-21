# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Common types and utilities used in the IREE codebase.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

cc_library(
    name = "base",
    srcs = [
        "allocator.c",
        "allocator.h",
        "api.c",
        "status.c",
        "status.h",
        "string_view.c",
        "string_view.h",
        "time.c",
        "time.h",
    ],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":core_headers",
        ":tracing",
    ],
)

# TODO(benvanik): make these srcs and only expose an api_cc.h.
cc_library(
    name = "cc",
    srcs = [
        "status_cc.cc",
    ],
    hdrs = [
        "status_cc.h",
    ],
    deps = [
        ":base",
        ":core_headers",
        ":logging",
    ],
)

cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":base",
        ":cc",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "string_view_test",
    srcs = ["string_view_test.cc"],
    deps = [
        ":base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Core headers (platform detection, compiler compat, etc)
#===------------------------------------------------------------------------===#

cc_library(
    name = "core_headers",
    hdrs = [
        "alignment.h",
        "attributes.h",
        "config.h",
        "target_platform.h",
    ],
)

cc_library(
    name = "target_platform",
    hdrs = ["target_platform.h"],
)

#===------------------------------------------------------------------------===#
# Internal IREE C++ wrappers and utilities
#===------------------------------------------------------------------------===#

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    linkopts = select({
        "//iree:iree_is_android": [
            "-llog",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":core_headers",
        ":tracing",
        "//iree/base/internal:flags",
    ],
)

cc_library(
    name = "tracing",
    hdrs = ["tracing.h"],
    deps = [
        ":core_headers",
    ],
)
