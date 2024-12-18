// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/file.h"

#include <stddef.h>

#include "iree/hal/detail.h"
#include "iree/hal/device.h"

#define _VTABLE_DISPATCH(file, method_name) \
  IREE_HAL_VTABLE_DISPATCH(file, iree_hal_file, method_name)

IREE_HAL_API_RETAIN_RELEASE(file);

IREE_API_EXPORT iree_status_t iree_hal_file_import(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, import_file)(
          device, queue_affinity, access, handle, flags, out_file);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_memory_access_t
iree_hal_file_allowed_access(iree_hal_file_t* file) {
  IREE_ASSERT_ARGUMENT(file);
  return _VTABLE_DISPATCH(file, allowed_access)(file);
}

IREE_API_EXPORT uint64_t iree_hal_file_length(iree_hal_file_t* file) {
  IREE_ASSERT_ARGUMENT(file);
  return _VTABLE_DISPATCH(file, length)(file);
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_file_storage_buffer(
    iree_hal_file_t* file) {
  IREE_ASSERT_ARGUMENT(file);
  return _VTABLE_DISPATCH(file, storage_buffer)(file);
}

IREE_API_EXPORT bool iree_hal_file_supports_synchronous_io(
    iree_hal_file_t* file) {
  IREE_ASSERT_ARGUMENT(file);
  return _VTABLE_DISPATCH(file, supports_synchronous_io)(file);
}

IREE_API_EXPORT iree_status_t iree_hal_file_read(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(file);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, file_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)buffer_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);
  iree_status_t status = _VTABLE_DISPATCH(file, read)(file, file_offset, buffer,
                                                      buffer_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_file_write(
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(file);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, file_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)buffer_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);
  iree_status_t status = _VTABLE_DISPATCH(file, write)(
      file, file_offset, buffer, buffer_offset, length);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
