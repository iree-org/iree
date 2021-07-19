# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(WEBGPU_HEADERS_ROOT "${IREE_ROOT_DIR}/third_party/webgpu-headers/")

external_cc_library(
  PACKAGE
    webgpu_headers
  NAME
    webgpu_headers
  ROOT
    ${WEBGPU_HEADERS_ROOT}
  HDRS
    "webgpu.h"
)
