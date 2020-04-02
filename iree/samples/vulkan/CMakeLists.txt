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

iree_bytecode_module(
  NAME
    simple_mul_bytecode_module
  SRC
    "simple_mul.mlir"
  CC_NAMESPACE
    "iree::samples::vulkan"
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=vulkan-spirv"
)

# bazel_to_cmake: DO NOT EDIT
# TODO(scotttodd): handle vulkan in bazel_to_cmake
# Statically link against Vulkan.
# Requires installing the Vulkan SDK from https://vulkan.lunarg.com/.
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(FATAL_ERROR "Vulkan library was not found")
endif()

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
  set(_GUI_LINKOPTS "-SUBSYSTEM:WINDOWS")
else()
  set(_GUI_LINKOPTS "")
endif()

iree_cc_binary(
    NAME
      vulkan_inference_gui
    SRCS
      "vulkan_inference_gui.cc"
    DEPS
      absl::base
      dear_imgui::dear_imgui
      dear_imgui::impl_sdl
      dear_imgui::impl_vulkan
      iree::base::api
      iree::base::logging
      iree::base::main
      iree::hal::api
      iree::hal::vulkan::api
      iree::modules::hal
      iree::samples::vulkan::simple_mul_bytecode_module_cc
      iree::vm
      iree::vm::bytecode_module
      SDL2-static
      Vulkan::Vulkan
    LINKOPTS
      "${_GUI_LINKOPTS}"
)
