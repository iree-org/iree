# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV OR
   NOT IREE_HAL_DRIVER_VULKAN)
  return()
endif()

iree_bytecode_module(
  NAME
    simple_mul_bytecode_module
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_samples_vulkan_simple_mul_bytecode_module"
  FLAGS
    "--iree-input-type=mhlo"
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
    iree::base::internal::main
    iree::hal::vulkan::registration
    iree::modules::hal
    iree::samples::vulkan::simple_mul_bytecode_module_c
    iree::testing::vulkan::vulkan_gui_util
    iree::vm
    iree::vm::bytecode_module
    iree::vm::cc
  LINKOPTS
    "${IREE_TARGET_GUI_LINKOPTS}"
)
