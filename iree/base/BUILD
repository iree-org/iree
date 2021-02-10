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

# Common types and utilities used in the IREE codebase.

load("//build_tools/bazel:iree_flatcc.bzl", "iree_flatbuffer_c_library")
load("//build_tools/bazel:run_binary_test.bzl", "run_binary_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

cc_library(
    name = "api",
    srcs = ["api.c"],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":core_headers",
        ":tracing",
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
        "target_platform.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
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
    name = "dynamic_library",
    srcs = [
        "dynamic_library_posix.cc",
        "dynamic_library_win32.cc",
    ],
    hdrs = ["dynamic_library.h"],
    deps = [
        ":core_headers",
        ":logging",
        ":status",
        ":tracing",
        "//build_tools:default_linkopts",
        "//iree/base/internal:file_path",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
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
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        ":core_headers",
        ":tracing",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "signature_mangle",
    srcs = ["signature_mangle.cc"],
    hdrs = ["signature_mangle.h"],
    deps = [
        ":status",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "signature_mangle_test",
    srcs = ["signature_mangle_test.cc"],
    deps = [
        ":signature_mangle",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    deps = [
        "//iree/base/internal:status_internal",
    ],
)

cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "synchronization",
    srcs = ["synchronization.c"],
    hdrs = ["synchronization.h"],
    deps = [
        ":api",
        ":core_headers",
        ":tracing",
        "//iree/base/internal",
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
        ":api",
        ":core_headers",
        ":synchronization",
        ":tracing",
        "//build_tools:default_linkopts",
        "//iree/base/internal",
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
    name = "tracing",
    hdrs = ["tracing.h"],
    deps = [
        ":core_headers",
    ],
)
