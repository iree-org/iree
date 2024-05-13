# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if((NOT IREE_TARGET_BACKEND_ROCM) OR
   (NOT "rocm" IN_LIST IREE_EXTERNAL_HAL_DRIVERS))
  return()
endif()

if(NOT IREE_ROCM_PATH)
  message(WARNING "IREE_ROCM_PATH not specified; skipping custom_dispatch/hip/kernels sample")
  return()
endif()

# NOTE: this is not how one should actually build their HSACO files. Do not use
# this as an authoritative source for compilation settings or CMake goo. If you
# choose to go the route of custom CUDA kernels you must bring your own build
# infrastructure. This sample only demonstrates how to use compiled HSACO blobs
# inside of the IREE compiler and this is the minimum amount of hacking that
# could be done to do that.

# Builds a HSACO blob using the clang built by IREE from tip-of-tree LLVM.
function(hip_kernel_hsaco_clang _ARCH)
  set(_NAME iree_samples_custom_dispatch_hip_kernels_hsaco_${_ARCH})
  set(_HSACO_SRC_NAME "kernels.cu")
  get_filename_component(_HSACO_SRC_BASENAME ${_HSACO_SRC_NAME} NAME_WE CACHE)
  set(_HSACO_OBJ_NAME "${_HSACO_SRC_BASENAME}_${_ARCH}.co")
  add_custom_command(
    OUTPUT
      ${_HSACO_OBJ_NAME}
    DEPENDS
      ${_HSACO_SRC_NAME}
      ${IREE_CLANG_TARGET}
    COMMAND ${IREE_CLANG_BINARY}
      -x hip
      --offload-device-only
      --offload-arch=${_ARCH}
      --rocm-path=${IREE_ROCM_PATH}
      -fuse-cuid=none
      -O3
      ${CMAKE_CURRENT_SOURCE_DIR}/${_HSACO_SRC_NAME}
      -o ${CMAKE_CURRENT_BINARY_DIR}/${_HSACO_OBJ_NAME}
    VERBATIM
  )
  add_custom_target(${_NAME} DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/${_HSACO_OBJ_NAME}
  )
  add_dependencies(iree-sample-deps "${_NAME}")
endfunction()

# Build the kernels_*.co files for each architecture we target.
hip_kernel_hsaco_clang(gfx1100)

iree_lit_test_suite(
  NAME
    example
  SRCS
    "example.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=rocm"
    "hostonly"
)
