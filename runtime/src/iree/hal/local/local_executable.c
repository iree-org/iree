// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_executable.h"

#include "iree/hal/local/executable_environment.h"

void iree_hal_local_executable_initialize(
    const iree_hal_local_executable_vtable_t* vtable,
    iree_allocator_t host_allocator,
    iree_hal_local_executable_t* out_base_executable) {
  iree_hal_resource_initialize(vtable, &out_base_executable->resource);
  out_base_executable->host_allocator = host_allocator;

  // Function attributes are optional and populated by the parent type.
  out_base_executable->dispatch_attrs = NULL;

  // Default environment with no imports assigned.
  iree_hal_executable_environment_initialize(host_allocator,
                                             &out_base_executable->environment);
}

void iree_hal_local_executable_deinitialize(
    iree_hal_local_executable_t* base_executable) {}

iree_hal_local_executable_t* iree_hal_local_executable_cast(
    iree_hal_executable_t* base_value) {
  return (iree_hal_local_executable_t*)base_value;
}

iree_status_t iree_hal_local_executable_issue_call(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(dispatch_state);
  IREE_ASSERT_ARGUMENT(workgroup_state);
  return ((const iree_hal_local_executable_vtable_t*)
              executable->resource.vtable)
      ->issue_call(executable, ordinal, dispatch_state, workgroup_state,
                   worker_id);
}

iree_status_t iree_hal_local_executable_issue_dispatch_inline(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    uint32_t processor_id, iree_byte_span_t local_memory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO(benvanik): annotate with executable name to calculate total time.

  const uint32_t workgroup_count_x = dispatch_state->workgroup_count_x;
  const uint32_t workgroup_count_y = dispatch_state->workgroup_count_y;
  const uint32_t workgroup_count_z = dispatch_state->workgroup_count_z;

#if IREE_HAL_VERBOSE_TRACING_ENABLE
  // TODO(benvanik): tracing.h helper that speeds this up; too slow.
  IREE_TRACE({
    char xyz_string[32];
    int xyz_string_length =
        snprintf(xyz_string, IREE_ARRAYSIZE(xyz_string), "%ux%ux%u",
                 workgroup_count_x, workgroup_count_y, workgroup_count_z);
    IREE_TRACE_ZONE_APPEND_TEXT(z0, xyz_string, xyz_string_length);
  });
#endif  // IREE_HAL_VERBOSE_TRACING_ENABLE

  iree_status_t status = iree_ok_status();

  iree_alignas(64) iree_hal_executable_workgroup_state_v0_t workgroup_state = {
      .workgroup_id_x = 0,
      .workgroup_id_y = 0,
      .workgroup_id_z = 0,
      .processor_id = processor_id,
      .local_memory = local_memory.data,
      .local_memory_size = (size_t)local_memory.data_length,
  };
  for (uint32_t z = 0; z < workgroup_count_z; ++z) {
    workgroup_state.workgroup_id_z = z;
    for (uint32_t y = 0; y < workgroup_count_y; ++y) {
      workgroup_state.workgroup_id_y = y;
      for (uint32_t x = 0; x < workgroup_count_x; ++x) {
        workgroup_state.workgroup_id_x = x;
        status = iree_hal_local_executable_issue_call(
            executable, ordinal, dispatch_state, &workgroup_state,
            /*worker_id=*/0);
        if (!iree_status_is_ok(status)) break;
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
