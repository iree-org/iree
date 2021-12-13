# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_HAL_DRIVER_DYLIB})

iree_cts_target(
  DRIVER_NAME
    dylib
  SRCS
    "driver_registration.cc"
  DEPS
    iree::hal::dylib::registration
)

endif()

if(${IREE_HAL_DRIVER_DYLIB_SYNC})

iree_cts_target(
  DRIVER_NAME
    dylib-sync
  SRCS
    "sync_driver_registration.cc"
  DEPS
    iree::hal::dylib::registration::sync
  EXCLUDED_TESTS
    # TODO(#4680): command buffer recording so that these can run on sync HAL
    "command_buffer"
    "event"
    "semaphore_submission"
)

endif()
