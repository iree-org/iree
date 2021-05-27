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

#include "iree/hal/local/local_executable_layout.h"

#include "iree/base/tracing.h"
#include "iree/hal/local/local_descriptor_set_layout.h"

static const iree_hal_executable_layout_vtable_t
    iree_hal_local_executable_layout_vtable;

iree_hal_local_executable_layout_t* iree_hal_local_executable_layout_cast(
    iree_hal_executable_layout_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_local_executable_layout_vtable);
  return (iree_hal_local_executable_layout_t*)base_value;
}

iree_status_t iree_hal_local_executable_layout_create(
    iree_host_size_t push_constants, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_allocator_t host_allocator,
    iree_hal_executable_layout_t** out_executable_layout) {
  IREE_ASSERT_ARGUMENT(!set_layout_count || set_layouts);
  IREE_ASSERT_ARGUMENT(out_executable_layout);
  *out_executable_layout = NULL;
  if (set_layout_count > IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "set layout count %zu over the limit of %d",
                            set_layout_count,
                            IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT);
  }
  if (push_constants > IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant count %zu over the limit of %d",
                            push_constants,
                            IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size =
      sizeof(iree_hal_local_executable_layout_t) +
      set_layout_count * sizeof(iree_hal_descriptor_set_layout_t*);

  iree_hal_local_executable_layout_t* layout = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&layout);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_local_executable_layout_vtable,
                                 &layout->resource);
    layout->host_allocator = host_allocator;
    layout->push_constants = push_constants;
    layout->dynamic_binding_count = 0;
    layout->used_bindings = 0;
    layout->set_layout_count = set_layout_count;
    for (iree_host_size_t i = 0; i < set_layout_count; ++i) {
      layout->set_layouts[i] = set_layouts[i];
      iree_hal_descriptor_set_layout_retain(layout->set_layouts[i]);

      iree_hal_local_descriptor_set_layout_t* local_set_layout =
          iree_hal_local_descriptor_set_layout_cast(set_layouts[i]);
      for (iree_host_size_t j = 0; j < local_set_layout->binding_count; ++j) {
        const iree_hal_descriptor_set_layout_binding_t* binding =
            &local_set_layout->bindings[j];
        layout->used_bindings |=
            1ull << (i * IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT + j);
        switch (binding->type) {
          case IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
          case IREE_HAL_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            ++layout->dynamic_binding_count;
            break;
        }
      }
    }
    *out_executable_layout = (iree_hal_executable_layout_t*)layout;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_executable_layout_destroy(
    iree_hal_executable_layout_t* base_layout) {
  iree_hal_local_executable_layout_t* layout =
      iree_hal_local_executable_layout_cast(base_layout);
  iree_allocator_t host_allocator = layout->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < layout->set_layout_count; ++i) {
    iree_hal_descriptor_set_layout_release(layout->set_layouts[i]);
  }
  iree_allocator_free(host_allocator, layout);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_layout_vtable_t
    iree_hal_local_executable_layout_vtable = {
        .destroy = iree_hal_local_executable_layout_destroy,
};
