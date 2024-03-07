# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_INPUT_STABLEHLO)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/input/StableHLO input/StableHLO)
endif()

if(IREE_INPUT_TORCH)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/input/Torch input/Torch)
endif()

if(IREE_INPUT_TOSA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/input/TOSA input/TOSA)
endif()

if(IREE_TARGET_BACKEND_CUDA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/CUDA target/CUDA)
endif()

if(IREE_TARGET_BACKEND_METAL_SPIRV)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/MetalSPIRV target/MetalSPIRV)
endif()

if(IREE_TARGET_BACKEND_ROCM)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/ROCM target/ROCM)
endif()

if(IREE_TARGET_BACKEND_VMVX)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/VMVX target/VMVX)
endif()

if(IREE_TARGET_BACKEND_WEBGPU_SPIRV)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/target/WebGPUSPIRV target/WebGPUSPIRV)
endif()
