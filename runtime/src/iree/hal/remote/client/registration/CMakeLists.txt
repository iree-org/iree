# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(NOT IREE_HAL_DRIVER_REMOTE)
  return()
endif()

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    iree::base
    iree::hal
    iree::hal::remote::client
  DEFINES
    "IREE_HAVE_HAL_REMOTE_CLIENT_DRIVER_MODULE=1"
  PUBLIC
)
