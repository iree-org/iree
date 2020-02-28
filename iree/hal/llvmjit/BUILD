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
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "llvmjit_executable",
    srcs = ["llvmjit_executable.cc"],
    hdrs = ["llvmjit_executable.h"],
    deps = [
        "//iree/base:status",
        "//iree/hal:allocator",
        "//iree/hal:executable",
        "//iree/hal:executable_spec",
        "//iree/schemas:llvmir_executable_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:asm_parser",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:execution_engine",
        "@llvm-project//llvm:orc_jit",
        "@llvm-project//llvm:support",
    ],
)

cc_library(
    name = "llvmjit_command_processor",
    srcs = ["llvmjit_command_processor.cc"],
    hdrs = ["llvmjit_command_processor.h"],
    deps = [
        ":llvmjit_executable",
        ":memref_runtime",
        "//iree/base:tracing",
        "//iree/hal:buffer",
        "//iree/hal/host:host_local_command_processor",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:execution_engine",
        "@llvm-project//llvm:orc_jit",
        "@llvm-project//llvm:support",
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
        "//iree/hal:allocator",
        "//iree/hal:executable",
        "//iree/hal:executable_cache",
        "//iree/hal:executable_format",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:orc_jit",
    ],
)

cc_library(
    name = "llvmjit_device",
    srcs = ["llvmjit_device.cc"],
    hdrs = ["llvmjit_device.h"],
    deps = [
        ":llvmjit_command_processor",
        ":llvmjit_executable_cache",
        "//iree/base:memory",
        "//iree/base:status",
        "//iree/base:tracing",
        "//iree/hal:command_buffer_validation",
        "//iree/hal:command_queue",
        "//iree/hal:device",
        "//iree/hal:fence",
        "//iree/hal/host:async_command_queue",
        "//iree/hal/host:host_event",
        "//iree/hal/host:host_local_allocator",
        "//iree/hal/host:host_submission_queue",
        "//iree/hal/host:inproc_command_buffer",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:orc_jit",
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
        "@llvm-project//llvm:core",
        "@llvm-project//llvm:execution_engine",
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
        "@llvm-project//llvm:support",
        #TODO(ataei): Link with native target dep.
        "@llvm-project//llvm:x86_code_gen",
    ],
    alwayslink = 1,
)

cc_library(
    name = "memref_runtime",
    hdrs = [
        "memref_runtime.h",
    ],
)
