// Copyright 2021 Google LLC
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
// limitations ufnder the License.

#include "iree/hal/cuda/event_semaphore.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/status_util.h"

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_cuda_context_wrapper_t* context;
  uint64_t initial_value;
} iree_hal_cuda_semaphore_t;

extern const iree_hal_semaphore_vtable_t iree_hal_cuda_semaphore_vtable;

static iree_hal_cuda_semaphore_t* iree_hal_cuda_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_semaphore_vtable);
  return (iree_hal_cuda_semaphore_t*)base_value;
}

iree_status_t iree_hal_cuda_semaphore_create(
    iree_hal_cuda_context_wrapper_t* context, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_semaphore_vtable,
                                 &semaphore->resource);
    semaphore->context = context;
    semaphore->initial_value = initial_value;
    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_cuda_semaphore_t* semaphore =
      iree_hal_cuda_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  // TODO: Support semaphores completely.
  *out_value = 0;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "Not impemented on CUDA");
}

static iree_status_t iree_hal_cuda_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  // TODO: Support semaphores completely. Return OK currently as everything is
  // synchronized for each submit to allow things to run.
  return iree_ok_status();
}

static void iree_hal_cuda_semaphore_fail(iree_hal_semaphore_t* base_semaphore,
                                         iree_status_t status) {}

static iree_status_t iree_hal_cuda_semaphore_wait_with_deadline(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_time_t deadline_ns) {
  // TODO: Support semaphores completely. Return OK currently as everything is
  // synchronized for each submit to allow things to run.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_semaphore_wait_with_timeout(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_duration_t timeout_ns) {
  return iree_hal_cuda_semaphore_wait_with_deadline(
      base_semaphore, value, iree_relative_timeout_to_deadline_ns(timeout_ns));
}

const iree_hal_semaphore_vtable_t iree_hal_cuda_semaphore_vtable = {
    .destroy = iree_hal_cuda_semaphore_destroy,
    .query = iree_hal_cuda_semaphore_query,
    .signal = iree_hal_cuda_semaphore_signal,
    .fail = iree_hal_cuda_semaphore_fail,
    .wait_with_deadline = iree_hal_cuda_semaphore_wait_with_deadline,
    .wait_with_timeout = iree_hal_cuda_semaphore_wait_with_timeout,
};
