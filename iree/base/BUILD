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
        "atomics.h",
        "bitfield.h",
        "math.h",
        "memory.h",
        "target_platform.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",  # bitfield.h
    ],
)

cc_test(
    name = "bitfield_test",
    srcs = ["bitfield_test.cc"],
    deps = [
        ":core_headers",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Internal IREE C++ wrappers and utilities
#===------------------------------------------------------------------------===#

cc_library(
    name = "arena",
    srcs = ["arena.cc"],
    hdrs = ["arena.h"],
    deps = [
        ":logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "arena_test",
    srcs = ["arena_test.cc"],
    deps = [
        ":arena",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "dynamic_library",
    srcs = [
        "dynamic_library_posix.cc",
        "dynamic_library_win32.cc",
    ],
    hdrs = ["dynamic_library.h"],
    deps = [
        ":core_headers",
        ":file_path",
        ":logging",
        ":status",
        ":tracing",
        "//build_tools:default_linkopts",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "file_io",
    hdrs = ["file_io.h"],
    deps = [
        ":core_headers",
        ":status",
        "//iree/base/internal:file_io_internal",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "file_io_hdrs",
    hdrs = ["file_io.h"],
    deps = [
        ":status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "file_io_test",
    srcs = ["file_io_test.cc"],
    deps = [
        ":file_io",
        ":file_path",
        ":logging",
        ":status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_mapping",
    hdrs = ["file_mapping.h"],
    deps = [
        ":ref_ptr",
        ":status",
        "//iree/base/internal:file_mapping_internal",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "file_mapping_hdrs",
    hdrs = ["file_mapping.h"],
    deps = [
        ":ref_ptr",
        ":status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "file_path",
    srcs = ["file_path.cc"],
    hdrs = ["file_path.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "file_path_test",
    srcs = [
        "file_path_test.cc",
    ],
    deps = [
        ":file_path",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    deps = [
        ":api",
        "@com_google_absl//absl/flags:parse",
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
    name = "intrusive_list",
    hdrs = [
        "intrusive_list.h",
        "intrusive_list_ref_ptr.inc",
        "intrusive_list_unique_ptr.inc",
    ],
    deps = [
        ":logging",
        ":ref_ptr",
    ],
)

cc_test(
    name = "intrusive_list_test",
    srcs = [
        "intrusive_list_ref_ptr_test.cc",
        "intrusive_list_test.cc",
        "intrusive_list_unique_ptr_test.cc",
    ],
    deps = [
        ":intrusive_list",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        ":tracing",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:str_format",
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
        ":core_headers",
        ":logging",
    ],
)

cc_library(
    name = "ref_ptr",
    hdrs = ["ref_ptr.h"],
    deps = [
        ":logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_test(
    name = "ref_ptr_test",
    size = "small",
    srcs = ["ref_ptr_test.cc"],
    deps = [
        ":ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
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
    ],
)

cc_test(
    name = "synchronization_benchmark",
    srcs = ["synchronization_benchmark.cc"],
    tags = [
        "nowindows",  # TODO(#3615) make this link on windows
    ],
    deps = [
        ":synchronization",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "synchronization_test",
    srcs = ["synchronization_test.cc"],
    tags = [
        "nowindows",  # TODO(#3615) make this link on windows
    ],
    deps = [
        ":synchronization",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    deps = [
        ":api",
    ],
)

cc_test(
    name = "time_test",
    srcs = ["time_test.cc"],
    deps = [
        ":time",
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
    ],
)

cc_test(
    name = "threading_benchmark",
    srcs = ["threading_benchmark.cc"],
    tags = [
        "nowindows",  # TODO(#3615) make this link on windows
    ],
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
    tags = [
        "nowindows",  # TODO(#3615) make this link on windows
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
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "wait_handle",
    srcs = [
        "wait_handle.c",
        "wait_handle_epoll.c",
        "wait_handle_kqueue.c",
        "wait_handle_poll.c",
        "wait_handle_posix.c",
        "wait_handle_posix.h",
        "wait_handle_win32.c",
    ],
    hdrs = ["wait_handle.h"],
    copts = [
        "-D_GNU_SOURCE=1",
    ],
    deps = [
        ":api",
        ":target_platform",
        ":tracing",
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
