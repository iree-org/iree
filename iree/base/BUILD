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

load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")
load("//iree:build_defs.oss.bzl", "platform_trampoline_deps")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "alignment",
    hdrs = ["alignment.h"],
    deps = [
        ":target_platform",
    ],
)

cc_library(
    name = "api",
    srcs = [
        "api.c",
    ],
    hdrs = ["api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":api_hdrs",
        ":target_platform",
        ":tracing",
    ],
)

cc_library(
    name = "api_hdrs",
    hdrs = ["api.h"],
)

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
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "atomics",
    hdrs = ["atomics.h"],
    deps = [
        ":target_platform",
    ],
)

cc_library(
    name = "bitfield",
    hdrs = ["bitfield.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "bitfield_test",
    srcs = ["bitfield_test.cc"],
    deps = [
        ":bitfield",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "dynamic_library",
    srcs = [
        "dynamic_library_posix.cc",
        "dynamic_library_win32.cc",
    ],
    hdrs = ["dynamic_library.h"],
    linkopts = ["-ldl"],
    deps = [
        ":logging",
        ":status",
        ":target_platform",
        ":tracing",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_binary(
    name = "dynamic_library_test_library.so",
    testonly = True,
    srcs = ["dynamic_library_test_library.cc"],
    linkshared = True,
)

cc_embed_data(
    name = "dynamic_library_test_library",
    testonly = True,
    srcs = [":dynamic_library_test_library.so"],
    cc_file_output = "dynamic_library_test_library_embed.cc",
    cpp_namespace = "iree",
    flatten = True,
    h_file_output = "dynamic_library_test_library_embed.h",
)

cc_test(
    name = "dynamic_library_test",
    srcs = ["dynamic_library_test.cc"],
    deps = [
        ":dynamic_library",
        ":dynamic_library_test_library",
        ":file_io",
        ":status",
        ":target_platform",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "file_io",
    hdrs = ["file_io.h"],
    deps = [
        ":status",
        ":target_platform",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + platform_trampoline_deps("file_io"),
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + platform_trampoline_deps("file_mapping"),
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
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "flatbuffer_util",
    srcs = ["flatbuffer_util.cc"],
    hdrs = ["flatbuffer_util.h"],
    deps = [
        ":file_mapping",
        ":memory",
        ":ref_ptr",
        ":status",
        ":tracing",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "init",
    hdrs = ["init.h"],
    deps = platform_trampoline_deps("init"),
)

cc_library(
    name = "initializer",
    srcs = ["initializer.cc"],
    hdrs = ["initializer.h"],
    deps = [
        ":target_platform",
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
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

# This library is a no-op in OSS, but exists for upstream compatibility
cc_library(
    name = "localfile",
    deps = platform_trampoline_deps("localfile"),
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    deps = platform_trampoline_deps("logging"),
)

cc_library(
    name = "main",
    srcs = [
        "main_posix.cc",
        "main_win32.cc",
    ],
    hdrs = ["main.h"],
    deps = [
        ":logging",
        ":target_platform",
    ],
)

cc_library(
    name = "math",
    hdrs = ["math.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "memory",
    hdrs = ["memory.h"],
    deps = [
        "@com_google_absl//absl/types:span",
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
    name = "target_platform",
    hdrs = ["target_platform.h"],
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

# Dependent code has been removed and wait_handle is currently incompatible
# with Windows, so excluding entirely.
# See google/iree/65
# cc_library(
#     name = "wait_handle",
#     srcs = ["wait_handle.cc"],
#     hdrs = ["wait_handle.h"],
#     deps = [
#         ":logging",
#         ":ref_ptr",
#         ":status",
#         ":time",
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/container:fixed_array",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/time",
#         "@com_google_absl//absl/types:span",
#     ],
# )

# cc_test(
#     name = "wait_handle_test",
#     srcs = ["wait_handle_test.cc"],
#     deps = [
#         ":status",
#         ":wait_handle",
#         "@com_google_absl//absl/time",
#         "//iree/testing:gtest",
#         "//iree/testing:gtest_main",
#     ],
# )
