# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unset(FILTER_TESTS)
string(APPEND FILTER_TESTS "SemaphoreTest.WaitThenFail:")
string(APPEND FILTER_TESTS "SemaphoreTest.FailThenWait:")
string(APPEND FILTER_TESTS "SemaphoreTest.MultiWaitThenFail:")
string(APPEND FILTER_TESTS "SemaphoreTest.DeviceMultiWaitThenFail:")
string(APPEND FILTER_TESTS "SemaphoreSubmissionTest.PropagateFailSignal:")
set(FILTER_TESTS_ARGS
  "--gtest_filter=-${FILTER_TESTS}"
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda
  VARIANT_SUFFIX
    graph
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/cuda/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_cuda_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--cuda_use_streams=false"
    ${FILTER_TESTS_ARGS}
  DEPS
    iree::hal::drivers::cuda::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=cuda
    requires-gpu-nvidia
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    cuda
  VARIANT_SUFFIX
    stream
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/cuda/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_cuda_driver_module_register"
  COMPILER_TARGET_BACKEND
    "cuda"
  EXECUTABLE_FORMAT
    "\"PTXE\""
  ARGS
    "--cuda_use_streams=true"
    ${FILTER_TESTS_ARGS}
  DEPS
    iree::hal::drivers::cuda::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
  LABELS
    driver=cuda
    requires-gpu-nvidia
)
