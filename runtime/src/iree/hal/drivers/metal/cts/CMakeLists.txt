# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: currently does not work on macos for me, don't have time to configure this right now.
# iree_hal_cts_test_suite(
#   DRIVER_NAME
#     metal
#   DRIVER_REGISTRATION_HDR
#     "runtime/src/iree/hal/drivers/metal/registration/driver_module.h"
#   DRIVER_REGISTRATION_FN
#     "iree_hal_metal_driver_module_register"
#   COMPILER_TARGET_BACKEND
#     "metal-spirv"
#   EXECUTABLE_FORMAT
#     "\"MTLE\""
#   DEPS
#     iree::hal::drivers::metal::registration
#   EXCLUDED_TESTS
#     # HAL event is unimplemented for Metal right now.
#     "event"
#     # Disabled until fixing semaphore tests.
#     # https://github.com/iree-org/iree/pull/17080#discussion_r1569867998
#     "semaphore_submission"
#     "semaphore"
#   LABELS
#     driver=metal
# )
