// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/inline_dispatch.h"

#include "iree/hal/local/local_executable.h"

iree_status_t iree_hal_local_executable_dispatch_inline(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_t* bindings, iree_host_size_t binding_count,
    iree_hal_dispatch_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Stack-allocate mapping and pointer arrays for the bindings.
  iree_hal_buffer_mapping_t* mappings = NULL;
  void** binding_ptrs = NULL;
  size_t* binding_lengths = NULL;
  if (binding_count > 0) {
    mappings = (iree_hal_buffer_mapping_t*)iree_alloca(binding_count *
                                                       sizeof(*mappings));
    memset(mappings, 0, binding_count * sizeof(*mappings));
    binding_ptrs = (void**)iree_alloca(binding_count * sizeof(*binding_ptrs));
    binding_lengths =
        (size_t*)iree_alloca(binding_count * sizeof(*binding_lengths));
  }

  // Map all binding buffers.
  iree_status_t status = iree_ok_status();
  iree_host_size_t mapped_count = 0;
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_buffer_map_range(
        bindings[i].buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_ANY, bindings[i].offset, bindings[i].length,
        &mappings[i]);
    if (iree_status_is_ok(status)) {
      binding_ptrs[i] = mappings[i].contents.data;
      binding_lengths[i] = mappings[i].contents.data_length;
      ++mapped_count;
    }
  }

  // Execute all workgroups inline.
  if (iree_status_is_ok(status)) {
    iree_hal_executable_dispatch_state_v0_t dispatch_state = {
        .workgroup_size_x = config.workgroup_size[0],
        .workgroup_size_y = config.workgroup_size[1],
        .workgroup_size_z = (uint16_t)config.workgroup_size[2],
        .constant_count = (uint16_t)(constants.data_length / sizeof(uint32_t)),
        .workgroup_count_x = config.workgroup_count[0],
        .workgroup_count_y = config.workgroup_count[1],
        .workgroup_count_z = (uint16_t)config.workgroup_count[2],
        .max_concurrency = 1,
        .binding_count = (uint8_t)binding_count,
        .constants = (const uint32_t*)constants.data,
        .binding_ptrs = binding_ptrs,
        .binding_lengths = binding_lengths,
    };
    status = iree_hal_local_executable_issue_dispatch_inline(
        iree_hal_local_executable_cast(executable), export_ordinal,
        &dispatch_state, /*processor_id=*/0, iree_byte_span_empty());
  }

  // Unmap all binding buffers (even on failure — mappings hold refs).
  for (iree_host_size_t i = 0; i < mapped_count; ++i) {
    iree_status_ignore(iree_hal_buffer_unmap_range(&mappings[i]));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
