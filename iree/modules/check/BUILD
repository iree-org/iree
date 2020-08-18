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

load(
    "//iree:build_defs.oss.bzl",
    "IREE_DRIVER_MODULES",
    "PLATFORM_VULKAN_DEPS",
    iree_cc_binary = "cc_binary",
)

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "check_test",
    srcs = ["check_test.cc"],
    deps = [
        ":native_module",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/hal:api",
        "//iree/hal/vmla:vmla_driver_module",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:ref_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

iree_cc_binary(
    name = "iree-check-module",
    testonly = True,
    srcs = ["check_module_main.cc"],
    deps = [
        ":native_module",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "//iree/base:api",
        "//iree/base:file_io",
        "//iree/base:init",
        "//iree/base:localfile",
        "//iree/base:status",
        "//iree/base:target_platform",
        "//iree/base:tracing",
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
        "//iree/base:status",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/testing:gtest",
        "//iree/vm",
        "//iree/vm:module_abi_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)
