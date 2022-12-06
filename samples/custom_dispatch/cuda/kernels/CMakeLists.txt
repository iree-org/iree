# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_CUDA OR NOT IREE_HAL_DRIVER_CUDA)
  return()
endif()

# NOTE: this is not how one should actually build their PTX files. Do not use
# this as an authoritative source for compilation settings or CMake goo. If you
# choose to go the route of custom CUDA kernels you must bring your own build
# infrastructure. This sample only demonstrates how to use compiled PTX blobs
# inside of the IREE compiler and this is the minimum amount of hacking that
# could be done to do that.

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "IREE custom_dispatch/cuda/kernels ignored -- nvcc not found")
  return()
endif()
enable_language(CUDA)

function(cuda_kernel_ptx _ARCH)
  set(_NAME iree_samples_custom_dispatch_cuda_kernels_ptx_${_ARCH})
  set(_PTX_SRC_NAME "kernels.cu")
  get_filename_component(_PTX_SRC_BASENAME ${_PTX_SRC_NAME} NAME_WE CACHE)
  set(_PTX_OBJ_NAME "${_PTX_SRC_BASENAME}_sm_${_ARCH}")

  add_library(${_NAME}_obj OBJECT)
  target_sources(${_NAME}_obj PRIVATE ${_PTX_SRC_NAME})
  set_source_files_properties(${_PTX_SRC_NAME} PROPERTIES LANGUAGE CUDA)
  set_target_properties(${_NAME}_obj PROPERTIES
    LANGUAGE CUDA
    LINKER_LANGUAGE CUDA
    CUDA_PTX_COMPILATION ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${_ARCH}"
  )

  # This makes my eyes bleed. There is probably a much better way of doing this
  # and I wish the best of luck to those who try. From:
  # https://sourcegraph.com/github.com/NVIDIA/MDL-SDK/-/blob/cmake/utilities.cmake?L1266
  # This sample should probably just invoke nvcc directly.
  get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_GENERATOR_IS_MULTI_CONFIG)
    set(_PTX_CONFIG_FOLDER /$<CONFIG>)
    set(_CMAKEFILES_FOLDER "")
  else()
    set(_PTX_CONFIG_FOLDER "")
    set(_CMAKEFILES_FOLDER /CMakeFiles)
  endif()
  add_custom_command(
    OUTPUT ${_PTX_OBJ_NAME}.ptx
    DEPENDS $<TARGET_OBJECTS:${_NAME}_obj>
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}${_CMAKEFILES_FOLDER}/${_NAME}_obj.dir${_PTX_CONFIG_FOLDER}/${_PTX_SRC_BASENAME}.ptx
        ${CMAKE_CURRENT_BINARY_DIR}/${_PTX_OBJ_NAME}.ptx
  )
  add_custom_target(${_NAME} DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/${_PTX_OBJ_NAME}.ptx
  )
  add_dependencies(iree-sample-deps "${_NAME}")
endfunction()

# Build the kernels_*.ptx files for each architecture we target.
cuda_kernel_ptx(52)
cuda_kernel_ptx(80)

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
    "driver=cuda"
    "hostonly"
)
