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
iree_hal_executable_export_count(iree_hal_executable_t* executable) {
  IREE_ASSERT_ARGUMENT(executable);
  return _VTABLE_DISPATCH(executable, export_count)(executable);
}

IREE_API_EXPORT iree_status_t iree_hal_executable_export_info(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_info);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, export_info)(
      executable, export_ordinal, out_info);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_export_parameters(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_parameters);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, export_parameters)(
      executable, export_ordinal, capacity, out_parameters);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_export_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_export_ordinal);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(executable, lookup_export_by_name)(
      executable, name, out_export_ordinal);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
