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

load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "custom_modules_test_module",
    src = "custom_modules_test.mlir",
    cc_namespace = "iree::samples::custom_modules",
    flags = ["-iree-mlir-to-vm-bytecode-module"],
    translate_tool = "//iree/samples/custom_modules/dialect:custom-translate",
)

cc_test(
    name = "custom_modules_test",
    srcs = ["custom_modules_test.cc"],
    deps = [
        ":custom_modules_test_module_cc",
        ":native_module",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/hal:api",
        "//iree/hal/vmla:vmla_driver_module",
        "//iree/modules/hal",
        "//iree/testing:gtest_main",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "//iree/vm:ref",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "native_module",
    srcs = ["native_module.cc"],
    hdrs = ["native_module.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:buffer_string_util",
        "//iree/base:shape",
        "//iree/base:shaped_buffer_string_util",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:module_abi_cc",
    ],
)
