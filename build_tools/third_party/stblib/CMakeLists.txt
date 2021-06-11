# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(STBLIB_ROOT "${IREE_ROOT_DIR}/third_party/stblib/")

iree_select_compiler_opts(_STBLIB_COPTS
  CLANG
    "-Wno-implicit-fallthrough"
)

external_cc_library(
  PACKAGE
    stblib
  NAME
    stb_image
  ROOT
    "${IREE_ROOT_DIR}/build_tools/third_party/stblib/"
  SRCS
    "stb_image.c"
  COPTS
    ${_STBLIB_COPTS}
  INCLUDES
    ${STBLIB_ROOT}
)
