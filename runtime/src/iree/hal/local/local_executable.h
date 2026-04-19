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

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_local_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Per-entry point dispatch attributes (constant counts, binding counts,
  // local memory requirements). NULL if the executable has no dispatch attrs.
  const iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs;

  // Per-entry point native function pointers. NULL for backends that don't
  // support direct dispatch (e.g., VMVX which dispatches through the VM).
  // Enables recording-time function resolution for the block ISA command
  // buffer: the raw function pointer is baked into .text at recording time
  // so execution has zero indirection.
  const iree_hal_executable_dispatch_v0_t* dispatch_ptrs;

  // Process-local nonzero executable identifier used by profiling sessions.
  uint64_t profile_id;

  // Execution environment.
  iree_hal_executable_environment_v0_t environment;
} iree_hal_local_executable_t;

typedef struct iree_hal_local_executable_vtable_t {
  iree_hal_executable_vtable_t base;

  iree_status_t(IREE_API_PTR* issue_call)(
      iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
      const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
      const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
      uint32_t worker_id);
} iree_hal_local_executable_vtable_t;

// Initializes the local executable base type.
void iree_hal_local_executable_initialize(
    const iree_hal_local_executable_vtable_t* vtable,
    iree_allocator_t host_allocator,
    iree_hal_local_executable_t* out_base_executable);

void iree_hal_local_executable_deinitialize(
    iree_hal_local_executable_t* base_executable);

iree_hal_local_executable_t* iree_hal_local_executable_cast(
    iree_hal_executable_t* base_value);

// Returns the process-local nonzero profiling identifier for |executable|.
uint64_t iree_hal_local_executable_profile_id(
    const iree_hal_local_executable_t* executable);

iree_status_t iree_hal_local_executable_issue_call(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id);

iree_status_t iree_hal_local_executable_issue_dispatch_inline(
    iree_hal_local_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    uint32_t processor_id, iree_byte_span_t local_memory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_EXECUTABLE_H_
