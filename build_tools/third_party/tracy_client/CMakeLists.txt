# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The runtime tracing library will depend on this target unconditionally.
# It conditionally either includes the necessary headers/defines or no-ops.
if(IREE_ENABLE_RUNTIME_TRACING)
  external_cc_library(
    PACKAGE
      tracy_client
    NAME
      runtime_impl
    ROOT
      "${IREE_ROOT_DIR}/third_party/tracy/public"
    HDRS
      "tracy/Tracy.hpp"
      "tracy/TracyC.h"
    INCLUDES
      "${IREE_ROOT_DIR}/third_party/tracy/public"
    DEPS
      ${CMAKE_DL_LIBS}
    DEFINES
      "IREE_TRACING_MODE=2"
  )
else()
  external_cc_library(
    PACKAGE
      tracy_client
    NAME
      runtime_impl
    ROOT
      "${IREE_ROOT_DIR}/third_party/tracy/public"
  )
endif()
