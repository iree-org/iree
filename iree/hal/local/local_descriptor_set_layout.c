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

#include "iree/hal/local/local_descriptor_set_layout.h"

#include "iree/base/tracing.h"

static const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_local_descriptor_set_layout_vtable;

iree_status_t iree_hal_local_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_ASSERT_ARGUMENT(!binding_count || bindings);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = NULL;
  if (binding_count > IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "binding count %zu over the limit of %d",
        binding_count, IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_descriptor_set_layout_t* layout = NULL;
  iree_host_size_t total_size =
      sizeof(*layout) + binding_count * sizeof(*layout->bindings);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_local_descriptor_set_layout_vtable,
                                 &layout->resource);
    layout->host_allocator = host_allocator;
    layout->usage_type = usage_type;
    layout->binding_count = binding_count;
    memcpy(layout->bindings, bindings,
           binding_count * sizeof(iree_hal_descriptor_set_layout_binding_t));
    *out_descriptor_set_layout = (iree_hal_descriptor_set_layout_t*)layout;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_descriptor_set_layout_destroy(
    iree_hal_descriptor_set_layout_t* base_layout) {
  iree_hal_local_descriptor_set_layout_t* layout =
      (iree_hal_local_descriptor_set_layout_t*)base_layout;
  iree_allocator_t host_allocator = layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, layout);

  IREE_TRACE_ZONE_END(z0);
}

iree_hal_local_descriptor_set_layout_t*
iree_hal_local_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value) {
  return (iree_hal_local_descriptor_set_layout_t*)base_value;
}

static const iree_hal_descriptor_set_layout_vtable_t
    iree_hal_local_descriptor_set_layout_vtable = {
        .destroy = iree_hal_local_descriptor_set_layout_destroy,
};
