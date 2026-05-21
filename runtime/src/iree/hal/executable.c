// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/executable.h"

#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

#define _VTABLE_DISPATCH(executable, method_name) \
  IREE_HAL_VTABLE_DISPATCH(executable, iree_hal_executable, method_name)

IREE_HAL_API_RETAIN_RELEASE(executable);

IREE_API_EXPORT iree_host_size_t
iree_hal_executable_function_count(iree_hal_executable_t* executable) {
  IREE_ASSERT_ARGUMENT(executable);
  return _VTABLE_DISPATCH(executable, function_count)(executable);
}

IREE_API_EXPORT iree_status_t iree_hal_executable_function_info(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_info);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, function_info)(
      executable, function, out_info);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_function_parameters(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_parameters);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, function_parameters)(
      executable, function, capacity, out_parameters);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_function_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_function);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, lookup_function_by_name)(
      executable, name, out_function);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_global_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;
  iree_status_t status = _VTABLE_DISPATCH(executable, lookup_global_by_name)(
      executable, name, queue_affinity, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
