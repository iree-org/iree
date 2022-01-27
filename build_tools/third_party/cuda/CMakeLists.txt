# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CUDA_HEADERS_API_ROOT "${IREE_ROOT_DIR}/third_party/cuda/include")

if (TARGET CUDA::cupti)
  # Locate the CUDA and CUPTI headers.
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

  external_cc_library(
    PACKAGE
      cuda_headers
    NAME
      cuda_headers
    ROOT
      ${CUDA_HEADERS_API_ROOT}
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
    INCLUDES
      ${CUDA_HEADERS_API_ROOT}
  )

endif(TARGET CUDA::cupti)
