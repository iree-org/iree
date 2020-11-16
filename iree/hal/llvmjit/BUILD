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

# HAL implementation for jitting CPU code from LLVMIR.

load("//iree:build_defs.oss.bzl", "iree_cmake_extra_content")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_cmake_extra_content(
    content = """
if(NOT ${IREE_HAL_DRIVER_LLVM})
  return()
endif()
""",
)

cc_library(
    name = "llvmjit",
    srcs = [
        "llvmjit_device.cc",
        "llvmjit_driver.cc",
        "llvmjit_executable.cc",
        "llvmjit_executable_cache.cc",
    ],
    hdrs = [
        "llvmjit_device.h",
        "llvmjit_driver.h",
        "llvmjit_executable.h",
        "llvmjit_executable_cache.h",
    ],
    deps = [
        "//iree/base:flatcc",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal",
        "//iree/hal/host:host_executable",
        "//iree/hal/host:host_local_device",
        "//iree/hal/host/serial:serial_scheduling_model",
        "//iree/schemas:llvmir_executable_def_c_fbs",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "llvmjit_driver_module",
    srcs = ["llvmjit_driver_module.cc"],
    deps = [
        ":llvmjit",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:driver_registry",
        "@llvm-project//llvm:Support",
        #TODO(ataei): Link with native target dep.
        "@llvm-project//llvm:X86CodeGen",
    ],
    alwayslink = 1,
)
