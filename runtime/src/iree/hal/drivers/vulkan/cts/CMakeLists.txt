# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unset(ADDITIONAL_TEST_ARGS)
if(ANDROID)
  # Disable this test due to flaky failures on Google Pixel 6 Pro.
  set(ADDITIONAL_TEST_ARGS "--gtest_filter=-*WaitForFiniteTime*")
endif()

iree_hal_cts_test_suite(
  DRIVER_NAME
    vulkan
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/vulkan/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_vulkan_driver_module_register"
  COMPILER_TARGET_BACKEND
    "vulkan-spirv"
  EXECUTABLE_FORMAT
    "\"vulkan-spirv-fb\""
  DEPS
    iree::hal::drivers::vulkan::registration
  EXCLUDED_TESTS
    "semaphore_submission"  # SubmitWithWait hangs (requires async)
  LABELS
    driver=vulkan
  ARGS
    ${ADDITIONAL_TEST_ARGS}
)
