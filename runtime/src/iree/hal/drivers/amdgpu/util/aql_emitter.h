// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AQL packet emission helpers. Pure functions that populate packet fields and
// return the header bits. They do NOT write the header — the caller commits
// it separately via iree_hal_amdgpu_aql_ring_commit(), which performs the
// atomic store-release that publishes the packet to the CP.
//
// This separation allows the caller to:
//   - Batch multiple packet commits before a single doorbell ring
//   - Control completion_signal assignment (epoch signal on last packet only)
//   - Populate kernarg memory between emission and commit
//
// All emitters zero reserved fields to prevent undefined behavior from stale
// ring data.
//
// Current host-queue policy sets the BARRIER bit on every packet so one AQL
// queue behaves as a single in-order dependency chain. That is intentionally
// stronger than HAL queue semantics, where user-visible ordering is expressed
// by semaphore signal->wait edges, not queue submission order. If packet
// barrier bits become conditional for independent HIP streams, the same-queue
// wait-elision/frontier logic in host_queue.c must be updated to materialize
// the required AQL dependency edges explicitly.
//
// Submission ordering contract (from kernarg_ring.h):
//   1. Reserve AQL ring slots (backpressure gate)
//   2. Allocate kernarg blocks (sizing invariant guarantees space)
//   3. Populate kernarg + packet fields (emit helpers)
//   4. Commit packet headers (atomic store-release)
//   5. Ring doorbell (once per batch)

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_

#include <string.h>

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Populates a kernel dispatch packet and returns the 16-bit AQL header.
// The grid dimensions (setup field) are returned via |out_setup| for the
// caller to pass to iree_hal_amdgpu_aql_ring_commit().
//
// Does NOT write the header word — the caller commits it after all packet
// fields and kernarg memory are fully populated.
static inline uint16_t iree_hal_amdgpu_aql_emit_dispatch(
    iree_hsa_kernel_dispatch_packet_t* packet, uint64_t kernel_object,
    const void* kernarg_address, const uint16_t workgroup_size[3],
    const uint32_t grid_size[3], uint32_t private_segment_size,
    uint32_t group_segment_size, iree_hsa_signal_t completion_signal,
    uint16_t* out_setup) {
  // Setup encodes the number of grid dimensions (always 3 for IREE).
  *out_setup = 3;

  packet->workgroup_size[0] = workgroup_size[0];
  packet->workgroup_size[1] = workgroup_size[1];
  packet->workgroup_size[2] = workgroup_size[2];
  packet->reserved0 = 0;
  packet->grid_size[0] = grid_size[0];
  packet->grid_size[1] = grid_size[1];
  packet->grid_size[2] = grid_size[2];
  packet->private_segment_size = private_segment_size;
  packet->group_segment_size = group_segment_size;
  packet->kernel_object = kernel_object;
  packet->kernarg_address = (void*)kernarg_address;
  packet->reserved2 = 0;
  packet->completion_signal = completion_signal;

  return (uint16_t)iree_hsa_make_packet_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      /*is_barrier=*/true, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_HSA_FENCE_SCOPE_SYSTEM);
}

// Populates an AMD barrier-value packet and returns the 16-bit AQL header.
// The setup field for barrier packets is always 0.
//
// The barrier halts the CP until:
//   (signal_load(dep_signal) & mask) CONDITION compare_value
//
// For cross-queue epoch waits the typical usage is:
//   dep_signal = source_queue->epoch.signal
//   condition  = IREE_HSA_SIGNAL_CONDITION_LT
//   compare_value = EPOCH_INITIAL_VALUE - target_epoch + 1
//   mask = ~0 (all bits)
static inline uint16_t iree_hal_amdgpu_aql_emit_barrier_value(
    iree_hsa_amd_barrier_value_packet_t* packet, iree_hsa_signal_t dep_signal,
    iree_hsa_signal_condition_t condition,
    iree_hsa_signal_value_t compare_value, iree_hsa_signal_value_t mask,
    iree_hsa_signal_t completion_signal) {
  // The vendor packet header has a secondary AmdFormat field.
  packet->header.AmdFormat = IREE_HSA_AMD_PACKET_TYPE_BARRIER_VALUE;
  packet->header.reserved = 0;
  packet->reserved0 = 0;
  packet->signal = dep_signal;
  packet->value = compare_value;
  packet->mask = mask;
  packet->cond = (iree_hsa_signal_condition32_t)condition;
  packet->reserved1 = 0;
  packet->reserved2 = 0;
  packet->reserved3 = 0;
  packet->completion_signal = completion_signal;

  // The primary header uses VENDOR_SPECIFIC packet type for AMD extensions.
  return (uint16_t)iree_hsa_make_packet_header(
      IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC,
      /*is_barrier=*/true, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_HSA_FENCE_SCOPE_SYSTEM);
}

// Populates a barrier-AND packet and returns the 16-bit AQL header.
// The barrier halts the CP until all non-null dependency signals reach 0.
// Up to 5 dependency signals are supported per packet.
static inline uint16_t iree_hal_amdgpu_aql_emit_barrier_and(
    iree_hsa_barrier_and_packet_t* packet, const iree_hsa_signal_t* dep_signals,
    uint32_t dep_count, iree_hsa_signal_t completion_signal) {
  packet->reserved0 = 0;
  packet->reserved1 = 0;
  // Fill dependency signals, nulling any unused slots.
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(packet->dep_signal); ++i) {
    packet->dep_signal[i] =
        i < dep_count ? dep_signals[i] : iree_hsa_signal_null();
  }
  packet->reserved2 = 0;
  packet->completion_signal = completion_signal;

  return (uint16_t)iree_hsa_make_packet_header(IREE_HSA_PACKET_TYPE_BARRIER_AND,
                                               /*is_barrier=*/true,
                                               IREE_HSA_FENCE_SCOPE_SYSTEM,
                                               IREE_HSA_FENCE_SCOPE_SYSTEM);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_
