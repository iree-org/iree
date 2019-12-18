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

add_custom_command(
  OUTPUT simple_mul_bytecode_module.emod
  COMMAND iree-translate
          --mlir-to-iree-module "${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.mlir"
          -o simple_mul_bytecode_module.emod
  DEPENDS iree_tools_iree_translate
)

add_custom_command(
  OUTPUT simple_mul_bytecode_module.h simple_mul_bytecode_module.cc
  COMMAND generate_cc_embed_data simple_mul_bytecode_module.emod
          --output_header=simple_mul_bytecode_module.h
          --output_impl=simple_mul_bytecode_module.cc
          --identifier=simple_mul_bytecode_module
          --cpp_namespace=iree::rt::samples
  DEPENDS generate_cc_embed_data
          simple_mul_bytecode_module.emod
)

add_library(
  simple_mul_bytecode_module STATIC
  simple_mul_bytecode_module.cc
)

# Statically link against Vulkan.
# Requires installing the Vulkan SDK from https://vulkan.lunarg.com/.
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(FATAL_ERROR "Vulkan library was not found")
endif()

iree_cc_binary(
    NAME
      vk_graphics_integration
    SRCS
      "vk_graphics_integration.cc"
    DEPS
      absl::base
      dear_imgui::dear_imgui
      dear_imgui::impl_sdl
      dear_imgui::impl_vulkan
      iree::base::api
      iree::base::logging
      iree::hal::api
      iree::hal::vulkan::api
      iree::rt::api
      iree::vm::api
      SDL2-static
      simple_mul_bytecode_module
      Vulkan::Vulkan
)
