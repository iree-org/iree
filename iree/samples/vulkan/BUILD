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

# Samples demonstrating use of the VM and HAL APIs with Vulkan.

load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "simple_mul_bytecode_module",
    src = "simple_mul.mlir",
    cc_namespace = "iree::samples::vulkan",
    flags = [
        "-iree-mlir-to-vm-bytecode-module",
        "-iree-hal-target-backends=vulkan-spirv",
    ],
)

cc_binary(
    name = "vulkan_inference_gui",
    srcs = ["vulkan_inference_gui.cc"],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": [
            "-SUBSYSTEM:WINDOWS",
        ],
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "nokokoro",
    ],
    deps = [
        ":simple_mul_bytecode_module_cc",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/base:main",
        "//iree/hal:api",
        "//iree/hal/vulkan:api",
        "//iree/modules/hal",
        "//iree/vm",
        "//iree/vm:bytecode_module",
        "@com_google_absl//absl/base:core_headers",
        "@dear_imgui",
        "@dear_imgui//:imgui_sdl_vulkan",
        "@iree_vulkan_headers//:vulkan_headers_no_prototypes",
        "@sdl2//:SDL2",
        "@vulkan_sdk//:sdk",
    ],
)
