# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Implementations for iree/base/.
# These are not part of the IREE API. Though they may be used by external
# projects their API may change at any time.

load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")
load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")
load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Core headers (internal)
#===------------------------------------------------------------------------===#
# Put files here that large percentages of the code include only; adding
# infrequently used files here will cause extraneous large rebuilds.

cc_library(
    name = "internal",
    srcs = [
        "atomics_clang.h",
        "atomics_gcc.h",
        "atomics_msvc.h",
    ],
    hdrs = [
        "atomics.h",
        "debugging.h",
        "inline_array.h",
        "math.h",
    ],
    deps = [
        "//iree/base:core_headers",
    ],
)

cc_test(
    name = "atomics_test",
    srcs = ["atomics_test.cc"],
    deps = [
        ":internal",
        "//iree/base:core_headers",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "math_test",
    srcs = ["math_test.cc"],
    deps = [
        ":internal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Utilities
#===------------------------------------------------------------------------===#

cc_library(
    name = "arena",
    srcs = ["arena.c"],
    hdrs = ["arena.h"],
    deps = [
        ":atomic_slist",
        ":synchronization",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_library(
    name = "atomic_slist",
    srcs = ["atomic_slist.c"],
    hdrs = ["atomic_slist.h"],
    deps = [
        ":internal",
        ":synchronization",
        "//iree/base:core_headers",
    ],
)

cc_test(
    name = "atomic_slist_test",
    srcs = ["atomic_slist_test.cc"],
    deps = [
        ":atomic_slist",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "dynamic_library",
    srcs = [
        "dynamic_library_posix.c",
        "dynamic_library_win32.c",
    ],
    hdrs = ["dynamic_library.h"],
    deps = [
        ":file_path",
        ":internal",
        ":synchronization",
        "//build_tools:default_linkopts",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_library(
    name = "file_io",
    srcs = ["file_io.c"],
    hdrs = ["file_io.h"],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_test(
    name = "file_io_test",
    srcs = ["file_io_test.cc"],
    deps = [
        ":file_io",
        "//iree/base:cc",
        "//iree/base:logging",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "file_path",
    srcs = ["file_path.c"],
    hdrs = ["file_path.h"],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
    ],
)

cc_test(
    name = "file_path_test",
    srcs = [
        "file_path_test.cc",
    ],
    deps = [
        ":file_path",
        "//iree/base:core_headers",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.c"],
    hdrs = ["flags.h"],
    deps = [
        ":file_io",
        ":internal",
        "//iree/base",
        "//iree/base:tracing",
    ],
)

cc_binary(
    name = "flags_demo",
    srcs = ["flags_demo.c"],
    deps = [
        ":flags",
        "//iree/base",
    ],
)

iree_lit_test_suite(
    name = "flags_test",
    srcs = ["flags_test.txt"],
    data = [
        ":flags_demo",
        "//iree/tools:IreeFileCheck",
    ],
    tags = ["hostonly"],
)

cc_library(
    name = "flatcc",
    hdrs = ["flatcc.h"],
    deps = [
        ":flatcc_dummy",
        "@com_github_dvidelabs_flatcc//:runtime",
    ],
)

iree_flatbuffer_c_library(
    name = "flatcc_dummy",
    srcs = ["flatcc.fbs"],
    flatcc_args = [
        "--reader",
        "--builder",
        "--verifier",
        "--json",
    ],
)

cc_library(
    name = "fpu_state",
    srcs = ["fpu_state.c"],
    hdrs = ["fpu_state.h"],
    deps = [
        ":internal",
        "//iree/base",
        "//iree/base:core_headers",
    ],
)

cc_test(
    name = "fpu_state_benchmark",
    srcs = ["fpu_state_benchmark.cc"],
    deps = [
        ":fpu_state",
        "//iree/base",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "fpu_state_test",
    srcs = ["fpu_state_test.cc"],
    deps = [
        ":fpu_state",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "main",
    srcs = [
        "main_posix.cc",
        "main_win32.cc",
    ],
    hdrs = ["main.h"],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:logging",
    ],
)

cc_library(
    name = "prng",
    hdrs = ["prng.h"],
    deps = [
        ":internal",
        "//iree/base:core_headers",
    ],
)

cc_test(
    name = "prng_test",
    srcs = ["prng_test.cc"],
    deps = [
        ":prng",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "span",
    hdrs = ["span.h"],
)

cc_library(
    name = "synchronization",
    srcs = [
        "synchronization.c",
    ],
    hdrs = [
        "call_once.h",
        "synchronization.h",
    ],
    deps = [
        ":internal",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_binary(
    name = "synchronization_benchmark",
    testonly = True,
    srcs = ["synchronization_benchmark.cc"],
    deps = [
        ":synchronization",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

run_binary_test(
    name = "synchronization_benchmark_test",
    args = ["--benchmark_min_time=0"],
    test_binary = ":synchronization_benchmark",
)

cc_test(
    name = "synchronization_test",
    srcs = ["synchronization_test.cc"],
    deps = [
        ":synchronization",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Utilities with thread dependencies
#===------------------------------------------------------------------------===#

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_ENABLE_THREADING})
  return()
endif()
""",
    inline = True,
)

cc_library(
    name = "threading",
    srcs = [
        "threading.c",
        "threading_darwin.c",
        "threading_impl.h",
        "threading_pthreads.c",
        "threading_win32.c",
    ],
    hdrs = ["threading.h"],
    deps = [
        ":internal",
        ":synchronization",
        "//build_tools:default_linkopts",
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_test(
    name = "threading_test",
    srcs = [
        "threading_impl.h",
        "threading_test.cc",
    ],
    deps = [
        ":internal",
        ":synchronization",
        ":threading",
        "//iree/base:cc",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "wait_handle",
    srcs = [
        "wait_handle.c",
        "wait_handle_epoll.c",
        "wait_handle_impl.h",
        "wait_handle_kqueue.c",
        "wait_handle_poll.c",
        "wait_handle_posix.c",
        "wait_handle_posix.h",
        "wait_handle_win32.c",
    ],
    hdrs = ["wait_handle.h"],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_test(
    name = "wait_handle_test",
    srcs = ["wait_handle_test.cc"],
    deps = [
        ":wait_handle",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)
