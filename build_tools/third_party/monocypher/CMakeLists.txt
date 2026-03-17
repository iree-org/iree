# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(MONOCYPHER_ROOT "${IREE_ROOT_DIR}/third_party/monocypher/")

external_cc_library(
  PACKAGE
    monocypher
  NAME
    monocypher
  ROOT
    ${MONOCYPHER_ROOT}
  INCLUDES
    "${MONOCYPHER_ROOT}/src"
  SRCS
    "src/monocypher.c"
  HDRS
    "src/monocypher.h"
  PUBLIC
)

iree_install_targets(
  TARGETS
    monocypher_monocypher
  COMPONENT
    IREEBundledLibraries
  EXPORT_SET
    Runtime
)
