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

# Implementations for iree/base/

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
        "//iree/base:core_headers",
        "//iree/base:synchronization",
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
    name = "file_handle_win32",
    srcs = ["file_handle_win32.cc"],
    hdrs = ["file_handle_win32.h"],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_io",
    hdrs = ["file_io.h"],
    deps = [
        "//iree/base:core_headers",
        "//iree/base:status",
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
        "//iree/base:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "file_io_test",
    srcs = ["file_io_test.cc"],
    deps = [
        ":file_io",
        ":file_path",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_io_internal",
    srcs = [
        "file_io_posix.cc",
        "file_io_win32.cc",
    ],
    deps = [
        ":file_handle_win32",
        ":file_io_hdrs",
        ":file_path",
        "//iree/base:core_headers",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
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
        "//iree/base:api",
        "@com_google_absl//absl/flags:parse",
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
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:logging",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
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
        "//iree/base:api",
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
