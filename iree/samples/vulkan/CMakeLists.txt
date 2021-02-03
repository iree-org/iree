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

if(NOT "${IREE_TARGET_BACKEND_VULKAN-SPIRV}" OR
   NOT "${IREE_HAL_DRIVER_VULKAN}")
  return()
endif()

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

# This sample statically links against Vulkan.
# One way to achieve this is by installing the Vulkan SDK from
# https://vulkan.lunarg.com/.
# If Vulkan is not found, omit the target.
if(NOT TARGET iree::testing::vulkan::vulkan_gui_util)
  message(VERBOSE "Missing Vulkan dependencies, skipping vulkan_inference_gui sample")
  return()
endif()

iree_cc_binary(
  NAME
    vulkan_inference_gui
  SRCS
    "vulkan_inference_gui.cc"
  DEPS
    absl::base
    iree::base::internal::main
    iree::hal::vulkan::registration
    iree::modules::hal
    iree::samples::vulkan::simple_mul_bytecode_module_cc
    iree::testing::vulkan::vulkan_gui_util
    iree::vm
    iree::vm::bytecode_module
    iree::vm::cc
  LINKOPTS
    "${IREE_TARGET_GUI_LINKOPTS}"
)
