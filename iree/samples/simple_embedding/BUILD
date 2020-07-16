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

load("//iree:build_defs.oss.bzl", "PLATFORM_VULKAN_TEST_DEPS")
load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "simple_embedding_test_bytecode_module",
    src = "simple_embedding_test.mlir",
    cc_namespace = "iree::samples",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vmla",
        "-iree-hal-target-backends=vulkan-spirv",
    ],
)

cc_test(
    name = "simple_embedding_test",
    srcs = ["simple_embedding_test.cc"],
    data = [
        # For AddressSanitizer when using Vulkan + a local Nvidia GPU
        "//iree/tools:sanitizer_suppressions.txt",
    ],
    tags = ["driver=vulkan"],
    deps = [
        ":simple_embedding_test_bytecode_module_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:logging",
        "//iree/hal:api",
        "//iree/modules/hal",
        "//iree/vm:bytecode_module",
        "//iree/vm",

        # These are the drivers we support running with and can produce
        # executables for from the source MLIR.
        "//iree/hal/vmla:vmla_driver_module",  # build-cleaner: keep
        "//iree/hal/vulkan:vulkan_driver_module",  # build-cleaner: keep
    ] + PLATFORM_VULKAN_TEST_DEPS,
)
