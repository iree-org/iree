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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "llvmjit_device",
    srcs = ["llvmjit_device.cc"],
    hdrs = ["llvmjit_device.h"],
    deps = [
        ":llvmjit_executable_cache",
        "//iree/base:tracing",
        "//iree/hal/host:host_local_device",
    ],
)

cc_library(
    name = "llvmjit_driver",
    srcs = ["llvmjit_driver.cc"],
    hdrs = ["llvmjit_driver.h"],
    deps = [
        ":llvmjit_device",
        "//iree/hal:device_info",
        "//iree/hal:driver",
        "//iree/hal/host/serial:serial_scheduling_model",
        "@llvm-project//llvm:ExecutionEngine",
    ],
)

cc_library(
    name = "llvmjit_driver_module",
    srcs = ["llvmjit_driver_module.cc"],
    deps = [
        ":llvmjit_driver",
        "//iree/base:init",
        "//iree/base:status",
        "//iree/hal:driver_registry",
        "@llvm-project//llvm:Support",
        #TODO(ataei): Link with native target dep.
        "@llvm-project//llvm:X86CodeGen",
    ],
    alwayslink = 1,
)

cc_library(
    name = "llvmjit_executable",
    srcs = ["llvmjit_executable.cc"],
    hdrs = ["llvmjit_executable.h"],
    deps = [
        ":memref_runtime",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:buffer",
        "//iree/hal:executable",
        "//iree/hal:executable_spec",
        "//iree/hal/host:host_executable",
        "//iree/schemas:llvmir_executable_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "llvmjit_executable_cache",
    srcs = ["llvmjit_executable_cache.cc"],
    hdrs = ["llvmjit_executable_cache.h"],
    deps = [
        ":llvmjit_executable",
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
