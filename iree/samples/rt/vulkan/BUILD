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

# Samples demonstrating use of the RT API with Vulkan.

load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

iree_bytecode_module(
    name = "simple_mul_bytecode_module",
    srcs = ["simple_mul.mlir"],
    cc_namespace = "iree::rt::samples",
)

cc_binary(
    name = "vk_graphics_integration",
    srcs = ["vk_graphics_integration.cc"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":simple_mul_bytecode_module_cc",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/hal:api",
        "//iree/hal/vulkan:api",
        "//iree/rt:api",
        "//iree/vm:api",
        "@com_google_absl//absl/base:core_headers",
        "@dear_imgui",
        "@dear_imgui//:imgui_sdl_vulkan",
        "@sdl2//:SDL2",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
        "@vulkan_sdk//:sdk",
    ],
)
