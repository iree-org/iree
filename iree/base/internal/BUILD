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

load("//iree:build_defs.oss.bzl", "iree_build_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "file_handle_win32",
    srcs = ["file_handle_win32.cc"],
    hdrs = ["file_handle_win32.h"],
    deps = [
        "//iree/base:status",
        "//iree/base:target_platform",
        "@com_google_absl//absl/memory",
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
        "//iree/base:file_io_hdrs",
        "//iree/base:file_path",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:tracing",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_mapping_internal",
    srcs = [
        "file_mapping_posix.cc",
        "file_mapping_win32.cc",
    ],
    deps = [
        ":file_handle_win32",
        "//iree/base:file_mapping_hdrs",
        "//iree/base:target_platform",
        "//iree/base:tracing",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "init_internal",
    srcs = ["init_internal.cc"],
    hdrs = ["init_internal.h"],
    deps = [
        "//iree/base:initializer",
        "//iree/base:target_platform",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "logging_internal",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "//iree/base:target_platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
    ],
)

# This library is a no-op in OSS, but exists for upstream compatibility
cc_library(
    name = "localfile_internal",
)

cc_library(
    name = "ostringstream",
    srcs = ["ostringstream.cc"],
    hdrs = ["ostringstream.h"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "status_internal",
    srcs = [
        "status.cc",
        "status_builder.cc",
        "statusor.cc",
    ],
    hdrs = [
        "status.h",
        "status_builder.h",
        "statusor.h",
    ],
    deps = [
        ":ostringstream",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/base:target_platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

iree_build_test(
    name = "internal_build_test",
    targets = [
        ":file_handle_win32",
        ":file_io_internal",
        ":file_mapping_internal",
        ":init_internal",
        ":logging_internal",
        ":status_internal",
    ],
)
