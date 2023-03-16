# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//build_tools/bazel:build_defs.oss.bzl", "iree_cmake_extra_content", "iree_runtime_cc_library", "iree_runtime_cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
# Task-based executor requires threading support.
if(NOT IREE_ENABLE_THREADING)
  return()
endif()

# cpuinfo can be conditionally disabled when it is not supported.
# If disabled then by default the task system will use 1 thread.
set(IREE_CPUINFO_TARGET)
if(IREE_ENABLE_CPUINFO)
  set(IREE_CPUINFO_TARGET cpuinfo)
endif()
""",
    inline = True,
)

iree_runtime_cc_library(
    name = "api",
    srcs = ["api.c"],
    hdrs = ["api.h"],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:flags",
    ],
)

iree_runtime_cc_library(
    name = "task",
    srcs = [
        "executor.c",
        "executor_impl.h",
        "list.c",
        "poller.c",
        "pool.c",
        "post_batch.c",
        "post_batch.h",
        "queue.c",
        "scope.c",
        "submission.c",
        "task.c",
        "task_impl.h",
        "topology.c",
        "topology_cpuinfo.c",
        "worker.c",
        "worker.h",
    ],
    hdrs = [
        "affinity_set.h",
        "executor.h",
        "list.h",
        "poller.h",
        "pool.h",
        "queue.h",
        "scope.h",
        "submission.h",
        "task.h",
        "topology.h",
        "tuning.h",
    ],
    deps = [
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:core_headers",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal",
        "//runtime/src/iree/base/internal:atomic_slist",
        "//runtime/src/iree/base/internal:cpu",
        "//runtime/src/iree/base/internal:event_pool",
        "//runtime/src/iree/base/internal:fpu_state",
        "//runtime/src/iree/base/internal:prng",
        "//runtime/src/iree/base/internal:synchronization",
        "//runtime/src/iree/base/internal:threading",
        "//runtime/src/iree/base/internal:wait_handle",
        "@cpuinfo",
    ],
)

iree_runtime_cc_test(
    name = "executor_demo",
    srcs = ["executor_demo.cc"],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/base:tracing",
        "//runtime/src/iree/base/internal:prng",
        "//runtime/src/iree/task/testing:test_util",
    ],
)

iree_runtime_cc_test(
    name = "executor_test",
    srcs = ["executor_test.cc"],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/task/testing:test_util",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "list_test",
    srcs = ["list_test.cc"],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/task/testing:test_util",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "pool_test",
    srcs = ["pool_test.cc"],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/task/testing:test_util",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "queue_test",
    srcs = ["queue_test.cc"],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/task/testing:test_util",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "scope_test",
    srcs = [
        "scope_test.cc",
        "task_impl.h",
    ],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/task/testing:test_util",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "task_tests",
    srcs = [
        "task_test_barrier.cc",
        "task_test_call.cc",
        "task_test_dispatch.cc",
        "task_test_fence.cc",
        "task_test_nop.cc",
        "task_test_wait.cc",
    ],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/task/testing:task_test",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)

iree_runtime_cc_test(
    name = "topology_test",
    srcs = ["topology_test.cc"],
    tags = [
        "noasan",  # TODO(8469): Does not work on machines with large numbers of cores.
    ],
    deps = [
        ":task",
        "//runtime/src/iree/base",
        "//runtime/src/iree/testing:gtest",
        "//runtime/src/iree/testing:gtest_main",
    ],
)
