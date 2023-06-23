# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

external_cc_library(
  PACKAGE
    tracy_client
  NAME
    runtime
  ROOT
    "${IREE_ROOT_DIR}/third_party/tracy/public"
  HDRS
    "tracy/Tracy.hpp"
    "tracy/TracyC.h"
  INCLUDES
    "${IREE_ROOT_DIR}/third_party/tracy/public"
  DEPS
    ${CMAKE_DL_LIBS}
)
