# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_HAL_DRIVER_DYLIB})

iree_hal_cts_test_suite(
  DRIVER_NAME
    dylib
  DRIVER_REGISTRATION_HDR
    "iree/hal/dylib/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_dylib_driver_module_register"
  DEPS
    iree::hal::dylib::registration
)

endif()

if(${IREE_HAL_DRIVER_DYLIB_SYNC})

iree_hal_cts_test_suite(
  DRIVER_NAME
    dylib-sync
  DRIVER_REGISTRATION_HDR
    "iree/hal/dylib/registration/driver_module_sync.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_dylib_sync_driver_module_register"
  DEPS
    iree::hal::dylib::registration::sync
  EXCLUDED_TESTS
    # TODO(#4680): command buffer recording so that these can run on sync HAL
    "command_buffer"
    "event"
    "semaphore_submission"
)

endif()
