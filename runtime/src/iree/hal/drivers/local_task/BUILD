# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Default implementations for HAL types that use the host resources.
# These are generally just wrappers around host heap memory and host threads.

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_runtime_cc_library(
    name = "task_driver",
    srcs = [
        "task_command_buffer.c",
        "task_device.c",
        "task_driver.c",
        "task_event.c",
        "task_queue.c",
        "task_queue_state.c",
        "task_semaphore.c",
    ],
    hdrs = [
        "task_command_buffer.h",
        "task_device.h",
        "task_driver.h",
        "task_event.h",
        "task_queue.h",
        "task_queue_state.h",
        "task_semaphore.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:arena",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:event_pool",
        "//runtime/src/iree/base/internal:synchronization",
        "//runtime/src/iree/base/internal:wait_handle",
        "//runtime/src/iree/hal",
        "//runtime/src/iree/hal/local",
        "//runtime/src/iree/hal/local:executable_environment",
        "//runtime/src/iree/hal/local:executable_library",
        "//runtime/src/iree/hal/utils:buffer_transfer",
        "//runtime/src/iree/hal/utils:resource_set",
        "//runtime/src/iree/hal/utils:semaphore_base",
        "//runtime/src/iree/task",
    ],
)
