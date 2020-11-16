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

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_DYLIB})
  return()
endif()
""",
)

cc_library(
    name = "dylib",
    srcs = [
        "dylib_device.cc",
        "dylib_driver.cc",
        "dylib_executable.cc",
        "dylib_executable_cache.cc",
    ],
    hdrs = [
        "dylib_device.h",
        "dylib_driver.h",
        "dylib_executable.h",
        "dylib_executable_cache.h",
    ],
    deps = [
        "//iree/base:dynamic_library",
        "//iree/base:file_io",
        "//iree/base:file_path",
        "//iree/base:flatcc",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal",
        "//iree/hal/host:host_executable",
        "//iree/hal/host:host_local_device",
        "//iree/hal/host/serial:serial_scheduling_model",
        "//iree/schemas:dylib_executable_def_c_fbs",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dylib_driver_module",
    srcs = ["dylib_driver_module.cc"],
    deps = [
        ":dylib",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:driver_registry",
    ],
    alwayslink = 1,
)
