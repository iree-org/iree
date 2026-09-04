# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Shardy common utilities

add_mlir_library(ShardyCommon
  PARTIAL_SOURCES_INTENDED
  ${SHARDY_SOURCE_DIR}/shardy/common/logging.cc
  ${SHARDY_SOURCE_DIR}/shardy/common/file_utils.cc
  ${SHARDY_SOURCE_DIR}/shardy/common/save_module_op.cc

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

target_include_directories(ShardyCommon PUBLIC
  $<BUILD_INTERFACE:${SHARDY_SOURCE_DIR}>
  $<BUILD_INTERFACE:${SHARDY_BINARY_DIR}>
)
