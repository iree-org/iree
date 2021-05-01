// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/allocator.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"

#define _VTABLE_DISPATCH(allocator, method_name) \
  IREE_HAL_VTABLE_DISPATCH(allocator, iree_hal_allocator, method_name)

IREE_HAL_API_RETAIN_RELEASE(allocator);

IREE_API_EXPORT iree_allocator_t
iree_hal_allocator_host_allocator(const iree_hal_allocator_t* allocator) {
  IREE_ASSERT_ARGUMENT(allocator);
  return _VTABLE_DISPATCH(allocator, host_allocator)(allocator);
}

IREE_API_EXPORT iree_hal_buffer_compatibility_t
iree_hal_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  IREE_ASSERT_ARGUMENT(allocator);
  return _VTABLE_DISPATCH(allocator, query_buffer_compatibility)(
      allocator, memory_type, allowed_usage, intended_usage, allocation_size);
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, allocate_buffer)(
      allocator, memory_type, allowed_usage, allocation_size, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_wrap_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, wrap_buffer)(
      allocator, memory_type, allowed_access, allowed_usage, data,
      data_allocator, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
