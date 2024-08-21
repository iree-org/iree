# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: investigate why these tests fail on Vulkan.
unset(FILTER_TESTS)
string(APPEND FILTER_TESTS "SemaphoreTest.WaitThenFail:")
string(APPEND FILTER_TESTS "SemaphoreTest.FailThenWait:")
string(APPEND FILTER_TESTS "SemaphoreTest.MultiWaitThenFail:")
string(APPEND FILTER_TESTS "SemaphoreTest.DeviceMultiWaitThenFail:")
string(APPEND FILTER_TESTS "SemaphoreSubmissionTest.PropagateFailSignal:")

if(ANDROID)
  # Disable this test on Android due to flaky failures on Google Pixel 6 Pro
  # and Moto Edge X30.
  string(APPEND FILTER_TESTS "SemaphoreTest.WaitForFiniteTime:")
endif()
set(ADDITIONAL_TEST_ARGS
  "--gtest_filter=-${FILTER_TESTS}"
)

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
