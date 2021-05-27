# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(RENDERDOC_API_ROOT "${IREE_ROOT_DIR}/third_party/renderdoc_api/")

external_cc_library(
  PACKAGE
    renderdoc_api
  NAME
    renderdoc_app
  ROOT
    ${RENDERDOC_API_ROOT}
  HDRS
    "app/renderdoc_app.h"
)
