# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CUDA_HEADERS_API_ROOT "${IREE_ROOT_DIR}/third_party/cuda/include")

unset(CUDA_INCLUDE_DIR CACHE)
unset(CUPTI_INCLUDE_DIR CACHE)

# Check for the CUDA install and enable if possible. Pull relevant headers into the build
if(IREE_HAL_DRIVER_CUDA AND IREE_ENABLE_RUNTIME_TRACING)
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building IREE with the CUDA driver requires a working CUDA install.")
  endif()

  find_package(CUDAToolkit) # Defines the CUDAToolkit_LIBRARY_ROOT variable

  find_path(
    CUDA_INCLUDE_DIR
    NAMES
      "cuda.h"
    PATHS ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES include
  )
  find_path(
    CUPTI_INCLUDE_DIR
    NAMES
      "cupti.h"
    PATHS ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES extras/CUPTI/include
  )
endif() # //IREE_HAL_DRIVER_CUDA AND IREE_ENABLE_RUNTIME_TRACING


if (CUDA_INCLUD_DIR AND CUPTI_INCLUDE_DIR)
  external_cc_library(
    PACKAGE
      cuda_headers
    NAME
      cuda_headers
    ROOT
      ${CUDA_HEADERS_API_ROOT}
    HDRS
      "cuda.h"
    DEFINES
      IREE_ENABLE_CUPTI
    INCLUDES
      ${CUDA_HEADERS_API_ROOT}
      ${CUDA_INCLUDE_DIR}
      ${CUPTI_INCLUDE_DIR}
  )
else()
  external_cc_library(
    PACKAGE
      cuda_headers
    NAME
      cuda_headers
    ROOT
      ${CUDA_HEADERS_API_ROOT}
    HDRS
      "cuda.h"
    INCLUDES
      ${CUDA_HEADERS_API_ROOT}
  )
endif() # //CUDA_INCLUD_DIR AND CUPTI_INCLUDE_DIR
