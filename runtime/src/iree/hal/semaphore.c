// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/semaphore.h"

#include <stddef.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(semaphore, method_name) \
  IREE_HAL_VTABLE_DISPATCH(semaphore, iree_hal_semaphore, method_name)

IREE_HAL_API_RETAIN_RELEASE(semaphore);

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_create(iree_hal_device_t* device, uint64_t initial_value,
                          iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, initial_value);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_semaphore)(
          device, initial_value, out_semaphore);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(semaphore, query)(semaphore, out_value);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, *out_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, new_value);
  iree_status_t status =
      _VTABLE_DISPATCH(semaphore, signal)(semaphore, new_value);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore,
                                             iree_status_t status) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, iree_status_code(status));
  _VTABLE_DISPATCH(semaphore, fail)(semaphore, status);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, value);
  iree_status_t status =
      _VTABLE_DISPATCH(semaphore, wait)(semaphore, value, timeout);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_semaphore_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void* params, void** inout_ptr) {
  iree_hal_semaphore_t* semaphore = (iree_hal_semaphore_t*)wait_source.self;
  const uint64_t target_value = wait_source.data;
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t* out_wait_status_code = (iree_status_code_t*)inout_ptr;
      uint64_t current_value = 0;
      iree_status_t status =
          iree_hal_semaphore_query(semaphore, &current_value);
      if (!iree_status_is_ok(status)) {
        *out_wait_status_code = iree_status_code(status);
        iree_status_ignore(status);
      } else {
        *out_wait_status_code = current_value < target_value
                                    ? IREE_STATUS_DEFERRED
                                    : IREE_STATUS_OK;
      }
      return iree_ok_status();
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      const iree_timeout_t timeout =
          ((const iree_wait_source_wait_params_t*)params)->timeout;
      return iree_hal_semaphore_wait(semaphore, target_value, timeout);
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT: {
      const iree_wait_primitive_type_t target_type =
          ((const iree_wait_source_export_params_t*)params)->target_type;
      // TODO(benvanik): support exporting semaphores to real wait handles.
      iree_wait_primitive_t* out_wait_primitive =
          (iree_wait_primitive_t*)inout_ptr;
      memset(out_wait_primitive, 0, sizeof(*out_wait_primitive));
      (void)target_type;
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "requested wait primitive type %d is unavailable",
                              (int)target_type);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented wait_source command");
  }
}

IREE_API_EXPORT iree_wait_source_t
iree_hal_semaphore_await(iree_hal_semaphore_t* semaphore, uint64_t value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  return (iree_wait_source_t){
      .self = semaphore,
      .data = value,
      .ctl = iree_hal_semaphore_wait_source_ctl,
  };
}
