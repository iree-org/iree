# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_CUDA})
  return()
endif()
""",
)

cc_library(
    name = "cuda",
    srcs = [
        "api.h",
        "context_wrapper.h",
        "cuda_allocator.c",
        "cuda_allocator.h",
        "cuda_buffer.c",
        "cuda_buffer.h",
        "cuda_device.c",
        "cuda_device.h",
        "cuda_driver.c",
        "cuda_event.c",
        "cuda_event.h",
        "descriptor_set_layout.c",
        "descriptor_set_layout.h",
        "event_semaphore.c",
        "event_semaphore.h",
        "executable_layout.c",
        "executable_layout.h",
        "graph_command_buffer.c",
        "graph_command_buffer.h",
        "native_executable.c",
        "native_executable.h",
        "nop_executable_cache.c",
        "nop_executable_cache.h",
        "status_util.c",
        "status_util.h",
    ],
    hdrs = [
        "api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dynamic_symbols",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/base/internal",
        "//iree/base/internal:flatcc",
        "//iree/base/internal:synchronization",
        "//iree/hal",
        "//iree/schemas:cuda_executable_def_c_fbs",
    ],
)

cc_library(
    name = "dynamic_symbols",
    srcs = [
        "cuda_headers.h",
        "dynamic_symbols.c",
    ],
    hdrs = [
        "dynamic_symbols.h",
    ],
    textual_hdrs = [
        "dynamic_symbol_tables.h",
    ],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:tracing",
        "//iree/base/internal:dynamic_library",
        "@cuda//:cuda_headers",
    ],
)

cc_test(
    name = "dynamic_symbols_test",
    srcs = ["dynamic_symbols_test.cc"],
    tags = ["driver=cuda"],
    deps = [
        ":dynamic_symbols",
        "//iree/base",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
