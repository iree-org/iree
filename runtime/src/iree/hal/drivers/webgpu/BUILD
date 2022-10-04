# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
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
        "executable.c",
        "executable.h",
        "nop_event.c",
        "nop_event.h",
        "nop_executable_cache.c",
        "nop_executable_cache.h",
        "nop_semaphore.c",
        "nop_semaphore.h",
        "pipeline_layout.c",
        "pipeline_layout.h",
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
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:arena",
        "//runtime/src/iree/base/internal/flatcc:parsing",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/drivers/webgpu/platform",
        "//runtime/src/iree/hal/drivers/webgpu/shaders",
        "//runtime/src/iree/hal/utils:buffer_transfer",
        "//runtime/src/iree/schemas:wgsl_executable_def_c_fbs",
        "@webgpu_headers",
    ],
)
