# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# We don't install any headers because their only use is in built code
# (not public).
set(IREE_HDRS_ROOT_PATH OFF)
# Considered part of the runtime. See runtime/src/CMakeLists.txt.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEDevLibraries)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET Runtime)

set(RCCL_SOURCE_DIR
  "${IREE_SOURCE_DIR}/third_party/rccl/"
)

external_cc_library(
  PACKAGE
    rccl
  NAME
    headers
  ROOT
    "${RCCL_SOURCE_DIR}"
  INCLUDES
    "${RCCL_SOURCE_DIR}"
  PUBLIC
)
