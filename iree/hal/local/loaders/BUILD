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

# Default implementations for HAL types that use the host resources.
# These are generally just wrappers around host heap memory and host threads.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "legacy_library_loader",
    srcs = ["legacy_library_loader.cc"],
    hdrs = ["legacy_library_loader.h"],
    defines = [
        "IREE_HAL_HAVE_LEGACY_LIBRARY_LOADER=1",
    ],
    deps = [
        "//iree/base:api",
        "//iree/base:dynamic_library",
        "//iree/base:flatcc",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/base/internal:file_path",
        "//iree/hal:api",
        "//iree/hal/local",
        "//iree/schemas:dylib_executable_def_c_fbs",
    ],
)

cc_library(
    name = "system_library_loader",
    srcs = ["system_library_loader.c"],
    hdrs = ["system_library_loader.h"],
    defines = [
        "IREE_HAL_HAVE_SYSTEM_LIBRARY_LOADER=1",
    ],
    deps = [
        "//iree/base:api",
        "//iree/base:flatcc",
        "//iree/base:tracing",
        "//iree/base/internal:file_io",
        "//iree/hal:api",
        "//iree/hal/local",
    ],
)

iree_cmake_extra_content(
    content = """
if(${IREE_HAL_DRIVER_VMLA})
""",
    inline = True,
)

cc_library(
    name = "vmla_module_loader",
    srcs = ["vmla_module_loader.cc"],
    hdrs = ["vmla_module_loader.h"],
    defines = [
        "IREE_HAL_HAVE_VMLA_MODULE_LOADER=1",
    ],
    deps = [
        "//iree/base:api",
        "//iree/base:flatcc",
        "//iree/base:tracing",
        "//iree/hal:api",
        "//iree/hal/local",
        "//iree/modules/vmla:op_module",
        "//iree/schemas:vmla_executable_def_c_fbs",
        "//iree/vm",
        "//iree/vm:bytecode_module",
    ],
)

iree_cmake_extra_content(
    content = """
endif()
""",
    inline = True,
)
