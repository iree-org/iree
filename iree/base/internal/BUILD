# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Implementations for iree/base/.
# These are not part of the IREE API. Though they may be used by external
# projects their API may change at any time.

load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")
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
        ":threading",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_library(
    name = "file_io",
    srcs = ["file_io.cc"],
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
        "//iree/base:core_headers",
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
    name = "status_internal",
    srcs = [
        "status.cc",
        "statusor.cc",
    ],
    hdrs = [
        "status.h",
        "statusor.h",
    ],
    deps = [
        "//iree/base",
        "//iree/base:core_headers",
        "//iree/base:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
    ],
)

cc_library(
    name = "synchronization",
    srcs = ["synchronization.c"],
    hdrs = ["synchronization.h"],
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
    name = "threading_benchmark",
    srcs = ["threading_benchmark.cc"],
    deps = [
        ":threading",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "threading_test",
    srcs = [
        "threading_impl.h",
        "threading_test.cc",
    ],
    deps = [
        ":synchronization",
        ":threading",
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
