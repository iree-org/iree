# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LIBURING_ROOT "${IREE_ROOT_DIR}/third_party/liburing/src/")

external_cc_library(
  PACKAGE
    liburing
  NAME
    liburing
  ROOT
    ${LIBURING_ROOT}
  INCLUDES
    ${LIBURING_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/default_config/src/include
  SRCS
    "queue.c"
    "register.c"
    "setup.c"
    "syscall.c"
    "syscall.h"
  HDRS
    "include/liburing.h"
    "include/liburing/barrier.h"
    "include/liburing/io_uring.h"
)
