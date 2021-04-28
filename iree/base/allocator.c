// Copyright 2019 Google LLC
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

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_malloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (!allocator.alloc) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no alloc routine");
  }
  return allocator.alloc(allocator.self, IREE_ALLOCATION_MODE_ZERO_CONTENTS,
                         byte_length, out_ptr);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_allocator_realloc(
    iree_allocator_t allocator, iree_host_size_t byte_length, void** out_ptr) {
  if (!allocator.alloc) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no alloc routine");
  }
  return allocator.alloc(allocator.self,
                         IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING, byte_length,
                         out_ptr);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_clone(iree_allocator_t allocator,
                     iree_const_byte_span_t source_bytes, void** out_ptr) {
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, source_bytes.data_length, out_ptr));
  memcpy(*out_ptr, source_bytes.data, source_bytes.data_length);
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL
iree_allocator_free(iree_allocator_t allocator, void* ptr) {
  if (ptr && allocator.free) {
    allocator.free(allocator.self, ptr);
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_allocator_system_allocate(void* self, iree_allocation_mode_t mode,
                               iree_host_size_t byte_length, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocations must be >0 bytes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  void* existing_ptr = *out_ptr;
  void* ptr = NULL;
  if (existing_ptr && (mode & IREE_ALLOCATION_MODE_TRY_REUSE_EXISTING)) {
    ptr = realloc(existing_ptr, byte_length);
    if (ptr && (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS)) {
      memset(ptr, 0, byte_length);
    }
  } else {
    existing_ptr = NULL;
    if (mode & IREE_ALLOCATION_MODE_ZERO_CONTENTS) {
      ptr = calloc(1, byte_length);
    } else {
      ptr = malloc(byte_length);
    }
  }
  if (!ptr) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "system allocator failed the request");
  }

  if (existing_ptr) {
    IREE_TRACE_FREE(existing_ptr);
  }
  IREE_TRACE_ALLOC(ptr, byte_length);

  *out_ptr = ptr;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void IREE_API_CALL iree_allocator_system_free(void* self,
                                                              void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_FREE(ptr);
  if (ptr) {
    free(ptr);
  }
  IREE_TRACE_ZONE_END(z0);
}
