# Copyright 2020 Google LLC
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

# HAL implementation for executing functions provided by dynamic libraries.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "dylib_device",
    srcs = ["dylib_device.cc"],
    hdrs = ["dylib_device.h"],
    deps = [
        ":dylib_executable_cache",
        "//iree/base:tracing",
        "//iree/hal/host:host_local_device",
    ],
)

cc_library(
    name = "dylib_driver",
    srcs = ["dylib_driver.cc"],
    hdrs = ["dylib_driver.h"],
    deps = [
        ":dylib_device",
        "//iree/hal:device_info",
        "//iree/hal:driver",
        "//iree/hal/host:serial_scheduling_model",
    ],
)

cc_library(
    name = "dylib_driver_module",
    srcs = ["dylib_driver_module.cc"],
    deps = [
        ":dylib_driver",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:driver_registry",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dylib_executable",
    srcs = ["dylib_executable.cc"],
    hdrs = ["dylib_executable.h"],
    deps = [
        ":memref_runtime",
        "//iree/base:dynamic_library",
        "//iree/base:file_io",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:executable",
        "//iree/hal:executable_spec",
        "//iree/hal/host:host_executable",
        "//iree/schemas:dylib_executable_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dylib_executable_cache",
    srcs = ["dylib_executable_cache.cc"],
    hdrs = ["dylib_executable_cache.h"],
    deps = [
        ":dylib_executable",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:executable",
        "//iree/hal:executable_cache",
        "//iree/hal:executable_format",
    ],
)

cc_library(
    name = "memref_runtime",
    hdrs = [
        "memref_runtime.h",
    ],
)
