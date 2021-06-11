# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CUDA_HEADERS_API_ROOT "${IREE_ROOT_DIR}/third_party/cuda/include")

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

