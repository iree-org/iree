# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# HAL (Hardware Abstraction Layer).
# Subdirectories contain implementations for different hardware and
# software backends.

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

cc_library(
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
        "command_buffer.c",
        "command_buffer.h",
        "command_buffer_validation.c",
        "descriptor_set.c",
        "descriptor_set.h",
        "descriptor_set_layout.c",
        "descriptor_set_layout.h",
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
        "executable_layout.c",
        "executable_layout.h",
        "resource.h",
        "semaphore.c",
        "semaphore.h",
        "string_util.c",
        "string_util.h",
    ],
    hdrs = [
        "api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal:synchronization",
    ],
)

cc_test(
    name = "string_util_test",
    srcs = ["string_util_test.cc"],
    deps = [
        ":hal",
        "//iree/base",
        "//iree/base:cc",
        "//iree/base/internal:span",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
