# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Global targets defining usage requirements for various parts of the
# tree.

add_library(iree_defs_runtime INTERFACE)

target_include_directories(
  iree_defs_runtime INTERFACE
    ${IREE_SOURCE_DIR}/runtime/src
)