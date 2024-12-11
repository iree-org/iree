# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HIP_TEST_TARGET_CHIP MATCHES "^gfx")
  return()
endif()

unset(IREE_HIP_TEST_COMPILER_FLAGS)
list(APPEND IREE_HIP_TEST_COMPILER_FLAGS
  "--iree-hip-target=${IREE_HIP_TEST_TARGET_CHIP}"
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    stream
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=true"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
  LABELS
    driver=hip
    requires-gpu-amd
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    graph
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=false"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
  LABELS
    driver=hip
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    multi_queue_stream
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  DEVICE_CREATION_HDR
    "runtime/src/iree/hal/drivers/hip/cts/multi_queue_device_creation.h"
  DEFAULT_DEVICE_CREATION_FN
    "iree_hal_drivers_hip_cts_default_multi_queue_create"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=true"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
  LABELS
    driver=hip
    requires-gpu-amd
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    multi_queue_graph
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  DEVICE_CREATION_HDR
    "runtime/src/iree/hal/drivers/hip/cts/multi_queue_device_creation.h"
  DEFAULT_DEVICE_CREATION_FN
    "iree_hal_drivers_hip_cts_default_multi_queue_create"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=false"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
  LABELS
    driver=hip
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    multi_queue_stream_queue_1
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  DEVICE_CREATION_HDR
    "runtime/src/iree/hal/drivers/hip/cts/multi_queue_device_creation.h"
  DEFAULT_DEVICE_CREATION_FN
    "iree_hal_drivers_hip_cts_default_multi_queue_create"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  DEFAULT_SUBMIT_QUEUE_AFFINITY
    "(1 << 1)"
  ARGS
    "--hip_use_streams=true"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
  LABELS
    driver=hip
    requires-gpu-amd
)

iree_hal_cts_test_suite(
  DRIVER_NAME
    hip
  VARIANT_SUFFIX
    multi_queue_graph_queue_1
  DRIVER_REGISTRATION_HDR
    "runtime/src/iree/hal/drivers/hip/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hip_driver_module_register"
  DEVICE_CREATION_HDR
    "runtime/src/iree/hal/drivers/hip/cts/multi_queue_device_creation.h"
  DEFAULT_DEVICE_CREATION_FN
    "iree_hal_drivers_hip_cts_default_multi_queue_create"
  DEFAULT_SUBMIT_QUEUE_AFFINITY
    "(1 << 1)"
  COMPILER_TARGET_BACKEND
    "rocm"
  EXECUTABLE_FORMAT
    "\"HSACO\""
  COMPILER_FLAGS
    ${IREE_HIP_TEST_COMPILER_FLAGS}
  ARGS
    "--hip_use_streams=false"
  DEPS
    iree::hal::drivers::hip::registration
  EXCLUDED_TESTS
    # HAL event is unimplemented for now.
    "event"
    # These tests fail with:
    #     UNAVAILABLE; missing hipDrvGraphAddMemcpyNode symbol;
    #     cannot use graph-based command buffer
    "command_buffer_copy_buffer"
    "command_buffer_dispatch"
    "command_buffer_update_buffer"
    "file"
  LABELS
    driver=hip
)
