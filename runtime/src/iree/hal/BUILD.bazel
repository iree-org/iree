# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# HAL (Hardware Abstraction Layer).
# Subdirectories contain implementations for different hardware and
# software backends.

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
    name = "hal",
    srcs = [
        "allocator.c",
        "allocator.h",
        "allocator_heap.c",
        "buffer.c",
        "buffer.h",
        "buffer_heap.c",
        "buffer_heap_impl.h",
        "buffer_view.c",
        "buffer_view.h",
        "buffer_view_util.c",
        "buffer_view_util.h",
        "channel.c",
        "channel.h",
        "command_buffer.c",
        "command_buffer.h",
        "command_buffer_validation.c",
        "command_buffer_validation.h",
        "detail.h",
        "device.c",
        "device.h",
        "driver.c",
        "driver.h",
        "driver_registry.c",
        "driver_registry.h",
        "event.c",
        "event.h",
        "executable.c",
        "executable.h",
        "executable_cache.c",
        "executable_cache.h",
        "fence.c",
        "fence.h",
        "pipeline_layout.c",
        "pipeline_layout.h",
        "resource.h",
        "semaphore.c",
        "semaphore.h",
        "string_util.c",
        "string_util.h",
    ],
    hdrs = [
        "api.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:path",
        "//runtime/src/iree/base/internal:synchronization",
    ],
)

iree_runtime_cc_test(
    name = "string_util_test",
    srcs = ["string_util_test.cc"],
    deps = [
        ":hal",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base/internal:span",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
