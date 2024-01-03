# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Traced builds require the headers.
set(IREE_HDRS_ROOT_PATH "${IREE_ROOT_DIR}/third_party/tracy/public")

# Considered part of the runtime. See runtime/src/CMakeLists.txt.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEBundledLibraries)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET "Runtime")

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
  SYSTEM_INCLUDES
    "${IREE_ROOT_DIR}/third_party/tracy/public"
  DEPS
    ${CMAKE_DL_LIBS}
)
