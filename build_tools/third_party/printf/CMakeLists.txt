# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(PRINTF_ROOT "${IREE_ROOT_DIR}/third_party/printf/")

# We don't install any headers because their only use is via the
# iree/base/printf.h wrapper.
set(IREE_HDRS_ROOT_PATH OFF)
# Considered part of the runtime.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEBundledLibraries)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET Runtime)

external_cc_library(
  PACKAGE
    printf
  NAME
    printf
  ROOT
    ${PRINTF_ROOT}
  INCLUDES
    "${PRINTF_ROOT}/src"
  SRCS
    "src/printf/printf.c"
  HDRS
    "src/printf/printf.h"
  COPTS
    "-DPRINTF_SUPPORT_WRITEBACK_SPECIFIER=0"
  PUBLIC
)
