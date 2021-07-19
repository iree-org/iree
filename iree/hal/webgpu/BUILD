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
if(NOT ("${IREE_HAL_DRIVER_WEBGPU_EMSCRIPTEN}" OR
        "${IREE_HAL_DRIVER_WEBGPU_NATIVE}"))
  return()
endif()
""",
)

cc_library(
    name = "webgpu",
    srcs = [
        "api.h",
        "bind_group_cache.c",
        "bind_group_cache.h",
        "buffer.c",
        "buffer.h",
        "builtins.c",
        "builtins.h",
        "command_buffer.c",
        "command_buffer.h",
        "descriptor_set_layout.c",
        "descriptor_set_layout.h",
        "executable.c",
        "executable.h",
        "executable_layout.c",
        "executable_layout.h",
        "nop_event.c",
        "nop_event.h",
        "nop_executable_cache.c",
        "nop_executable_cache.h",
        "nop_semaphore.c",
        "nop_semaphore.h",
        "simple_allocator.c",
        "simple_allocator.h",
        "staging_buffer.c",
        "staging_buffer.h",
        "webgpu_device.c",
        "webgpu_device.h",
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
        "//iree/base/internal/flatcc:parsing",
        "//iree/hal",
        "//iree/hal/webgpu/platform",
        "//iree/hal/webgpu/shaders",
        "//iree/schemas:wgsl_executable_def_c_fbs",
        "@webgpu_headers",
    ],
)
