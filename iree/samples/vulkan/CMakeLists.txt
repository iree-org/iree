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
  TRANSLATE_TOOL
    iree_tools_iree-translate
  TRANSLATION
    "-iree-mlir-to-vm-bytecode-module"
  PUBLIC
)

iree_cc_binary(
  NAME
    vulkan_inference_gui
  SRCS
    "vulkan_inference_gui.cc"
  DEPS
    ::simple_mul_bytecode_module_cc
    iree::base::api
    iree::base::logging
    iree::hal::api
    iree::hal::vulkan::api
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
    absl::core_headers
    dear_imgui::dear_imgui
    dear_imgui::impl_sdl
    dear_imgui::impl_vulkan
    SDL2-static
    Vulkan::Headers
    Vulkan::Vulkan
)
