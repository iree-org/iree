// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_executable.h"

#include "iree/base/tracing.h"
#include "iree/hal/local/executable_environment.h"

void iree_hal_local_executable_initialize(
    const iree_hal_local_executable_vtable_t* vtable,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* source_executable_layouts,
    iree_hal_local_executable_layout_t** target_executable_layouts,
    iree_allocator_t host_allocator,
    iree_hal_local_executable_t* out_base_executable) {
  iree_hal_resource_initialize(vtable, &out_base_executable->resource);
  out_base_executable->host_allocator = host_allocator;

  out_base_executable->executable_layout_count = executable_layout_count;
  out_base_executable->executable_layouts = target_executable_layouts;
  for (iree_host_size_t i = 0; i < executable_layout_count; ++i) {
    target_executable_layouts[i] =
        (iree_hal_local_executable_layout_t*)source_executable_layouts[i];
    iree_hal_executable_layout_retain(source_executable_layouts[i]);
  }

  // Function attributes are optional and populated by the parent type.
  out_base_executable->dispatch_attrs = NULL;

  // Default environment with no imports assigned.
  iree_hal_executable_environment_initialize(host_allocator,
                                             &out_base_executable->environment);
}

void iree_hal_local_executable_deinitialize(
    iree_hal_local_executable_t* base_executable) {
  for (iree_host_size_t i = 0; i < base_executable->executable_layout_count;
       ++i) {
    iree_hal_executable_layout_release(
        (iree_hal_executable_layout_t*)base_executable->executable_layouts[i]);
  }
}

iree_hal_local_executable_t* iree_hal_local_executable_cast(
    iree_hal_executable_t* base_value) {
  return (iree_hal_local_executable_t*)base_value;
}

iree_status_t iree_hal_local_executable_issue_call(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id, iree_byte_span_t local_memory) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(dispatch_state);
  IREE_ASSERT_ARGUMENT(workgroup_id);
  return ((const iree_hal_local_executable_vtable_t*)
              executable->resource.vtable)
      ->issue_call(executable, ordinal, dispatch_state, workgroup_id,
                   local_memory);
}

iree_status_t iree_hal_local_executable_issue_dispatch_inline(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    iree_byte_span_t local_memory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO(benvanik): annotate with executable name to calculate total time.

  const iree_hal_vec3_t workgroup_count = dispatch_state->workgroup_count;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  char xyz_string[32];
  int xyz_string_length =
      snprintf(xyz_string, IREE_ARRAYSIZE(xyz_string), "%ux%ux%u",
               workgroup_count.x, workgroup_count.y, workgroup_count.z);
  IREE_TRACE_ZONE_APPEND_TEXT_STRING_VIEW(z0, xyz_string, xyz_string_length);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  iree_status_t status = iree_ok_status();

  iree_hal_vec3_t workgroup_id;
  for (workgroup_id.z = 0; workgroup_id.z < workgroup_count.z;
       ++workgroup_id.z) {
    for (workgroup_id.y = 0; workgroup_id.y < workgroup_count.y;
         ++workgroup_id.y) {
      for (workgroup_id.x = 0; workgroup_id.x < workgroup_count.x;
           ++workgroup_id.x) {
        status = iree_hal_local_executable_issue_call(
            executable, ordinal, dispatch_state, &workgroup_id, local_memory);
        if (!iree_status_is_ok(status)) break;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
