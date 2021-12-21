# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_HAL_DRIVER_VMVX})

iree_hal_cts_test_suite(
  DRIVER_NAME
    vmvx
  DRIVER_REGISTRATION_HDR
    "iree/hal/vmvx/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_vmvx_driver_module_register"
  DEPS
    iree::hal::vmvx::registration
)

endif()

if(${IREE_HAL_DRIVER_VMVX_SYNC})

iree_hal_cts_test_suite(
  DRIVER_NAME
    vmvx-sync
  DRIVER_REGISTRATION_HDR
    "iree/hal/vmvx/registration/driver_module_sync.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_vmvx_sync_driver_module_register"
  DEPS
    iree::hal::vmvx::registration::sync
  EXCLUDED_TESTS
    # TODO(#4680): command buffer recording so that these can run on sync HAL
    "command_buffer"
    "event"
    "semaphore_submission"
)

endif()
