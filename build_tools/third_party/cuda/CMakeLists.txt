# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Check for the cuda install and enable if possible. Pull relevant headers into the build
unset(CUDA_INCLUD_DIR CACHE)
unset(CUPTI_INCLUDE_DIR CACHE)

if(IREE_HAL_DRIVER_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building IREE with the CUDA driver requires a working CUDA install.")
  endif()

  find_package(CUDAToolkit) # Defines the CUDAToolkit_LIBRARY_ROOT variable

  # Locate the CUDA header dir
  find_path(
    CUDA_INCLUD_DIR
    NAMES
      "cuda.h"
    PATHS ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES include
  )

  # CUPTI is optional (used for tracing). If NOTFOUND, don't use.
  find_path(
    CUPTI_INCLUDE_DIR
    NAMES
      "cupti.h"
    PATHS ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES extras/CUPTI/include
  )
  if (NOT CUPTI_INCLUDE_DIR)
    unset(CUPTI_INCLUDE_DIR CACHE)
  endif()
endif() # //IREE_HAL_DRIVER_CUDA

external_cc_library(
  PACKAGE
    cuda_headers
  NAME
    cuda_headers
  ROOT
    NOT_USED
  INCLUDES
  ${CUDA_INCLUD_DIR}
  ${CUPTI_INCLUDE_DIR}
)
