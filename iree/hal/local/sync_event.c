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
// limitations under the License.

#include "iree/hal/local/sync_event.h"

#include "iree/base/tracing.h"

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_sync_event_t;

static const iree_hal_event_vtable_t iree_hal_sync_event_vtable;

static iree_hal_sync_event_t* iree_hal_sync_event_cast(
    iree_hal_event_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_sync_event_vtable);
  return (iree_hal_sync_event_t*)base_value;
}

iree_status_t iree_hal_sync_event_create(iree_allocator_t host_allocator,
                                         iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_sync_event_t* event = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_sync_event_vtable, &event->resource);
    event->host_allocator = host_allocator;
    *out_event = (iree_hal_event_t*)event;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_sync_event_destroy(iree_hal_event_t* base_event) {
  iree_hal_sync_event_t* event = iree_hal_sync_event_cast(base_event);
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_event_vtable_t iree_hal_sync_event_vtable = {
    .destroy = iree_hal_sync_event_destroy,
};
