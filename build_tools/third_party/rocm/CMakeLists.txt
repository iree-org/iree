# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
if(NOT ${IREE_BUILD_EXPERIMENTAL_ROCM})
  return()
endif()

if(NOT ROCM_HEADERS_API_ROOT)
  set(ROCM_HEADERS_API_ROOT "/opt/rocm/include")
endif()

if (EXISTS ${ROCM_HEADERS_API_ROOT})
  message(STATUS "ROCm Header Path: ${ROCM_HEADERS_API_ROOT}")
else()
  message(SEND_ERROR "Could not locate ROCm: ${ROCM_HEADERS_API_ROOT}")
endif()

external_cc_library(
  PACKAGE
    rocm_headers
  NAME
    rocm_headers
  ROOT
    ${ROCM_HEADERS_API_ROOT}
  HDRS
    "hip/hip_runtime.h"
  INCLUDES
    ${ROCM_HEADERS_API_ROOT}
)

unset(ROCM_HEADERS_API_ROOT)