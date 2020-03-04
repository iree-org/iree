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

load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//iree:build_defs.oss.bzl", "IREE_DRIVER_MODULES", "PLATFORM_VULKAN_DEPS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "check_test_module",
    src = "check_test.mlir",
    cc_namespace = "iree::check",
    flags = ["-iree-mlir-to-vm-bytecode-module"],
    translate_tool = "//iree/modules/check/dialect:check-translate",
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cc"],
    deps = [
        ":check_test_module_cc",
        ":native_module",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:ref",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "iree-check-module",
    testonly = True,
    srcs = ["check_module_main.cc"],
    deps = [
        ":native_module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:file_io",
        "//iree/base:init",
        "//iree/base:localfile",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/tools:vm_util",
        "//iree/vm:bytecode_module",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_library(
    name = "native_module",
    testonly = True,
    srcs = ["native_module.cc"],
    hdrs = ["native_module.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:buffer_string_util",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/vm",
        "//iree/vm:module_abi_cc",
        "@com_google_absl//absl/strings",
    ],
)
