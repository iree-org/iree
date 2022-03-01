// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOCAL_EXECUTABLE_H_
#define IREE_HAL_LOCAL_LOCAL_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable_layout.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_local_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t executable_layout_count;
  iree_hal_local_executable_layout_t** executable_layouts;

  // Defines per-entry point how much workgroup local memory is required.
  // Contains entries with 0 to indicate no local memory is required or >0 in
  // units of IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE for the minimum amount
  // of memory required by the function.
  const iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs;

  // Execution environment.
  iree_hal_executable_environment_v0_t environment;
} iree_hal_local_executable_t;

typedef struct iree_hal_local_executable_vtable_t {
  iree_hal_executable_vtable_t base;

  iree_status_t(IREE_API_PTR* issue_call)(
      iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
      const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
      const iree_hal_vec3_t* workgroup_id, iree_byte_span_t local_memory);
} iree_hal_local_executable_vtable_t;

// Initializes the local executable base type.
//
// Callers must allocate memory for |target_executable_layouts| with at least
// `executable_layout_count * sizeof(*target_executable_layouts)` bytes.
void iree_hal_local_executable_initialize(
    const iree_hal_local_executable_vtable_t* vtable,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* source_executable_layouts,
    iree_hal_local_executable_layout_t** target_executable_layouts,
    iree_allocator_t host_allocator,
    iree_hal_local_executable_t* out_base_executable);

void iree_hal_local_executable_deinitialize(
    iree_hal_local_executable_t* base_executable);

iree_hal_local_executable_t* iree_hal_local_executable_cast(
    iree_hal_executable_t* base_value);

iree_status_t iree_hal_local_executable_issue_call(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id, iree_byte_span_t local_memory);

iree_status_t iree_hal_local_executable_issue_dispatch_inline(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    iree_byte_span_t local_memory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_EXECUTABLE_H_
