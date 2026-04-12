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

#include <stdbool.h>
#include <string.h>

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Packet header controls shared by all AQL emitters.
//
// Direct host-queue submissions still use barrier+system-fence packets so one
// queue behaves as an in-order chain. Command-buffer replay can opt into
// non-barrier dispatch packets and only set the barrier bit at logical ordering
// boundaries.
typedef struct iree_hal_amdgpu_aql_packet_control_t {
  // True when the packet participates in AQL queue-order dependency chaining.
  bool has_barrier;
  // Acquire fence scope encoded in the packet header.
  iree_hsa_fence_scope_t acquire_fence_scope;
  // Release fence scope encoded in the packet header.
  iree_hsa_fence_scope_t release_fence_scope;
} iree_hal_amdgpu_aql_packet_control_t;

// Returns packet control for a barrier packet with caller-selected scopes.
static inline iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_packet_control_barrier(
    iree_hsa_fence_scope_t acquire_fence_scope,
    iree_hsa_fence_scope_t release_fence_scope) {
  iree_hal_amdgpu_aql_packet_control_t packet_control;
  packet_control.has_barrier = true;
  packet_control.acquire_fence_scope = acquire_fence_scope;
  packet_control.release_fence_scope = release_fence_scope;
  return packet_control;
}

// Returns the current host-queue packet policy: barrier + system-scope fences.
static inline iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_aql_packet_control_barrier_system(void) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_SYSTEM);
}

// Builds the 16-bit packet header from |packet_type| and |packet_control|.
static inline uint16_t iree_hal_amdgpu_aql_make_header(
    iree_hsa_packet_type_t packet_type,
    iree_hal_amdgpu_aql_packet_control_t packet_control) {
  return (uint16_t)iree_hsa_make_packet_header(
      packet_type, packet_control.has_barrier,
      packet_control.acquire_fence_scope, packet_control.release_fence_scope);
}

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
    uint32_t group_segment_size,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
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

  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                                         packet_control);
}

// Populates an AMD barrier-value packet and returns the 16-bit AQL header.
// The vendor packet's upper 16 commit bits carry AmdFormat/reserved instead of
// the normal dispatch setup field and are returned in |out_setup|.
//
// The barrier halts the CP until:
//   (signal_load(dep_signal) & mask) CONDITION compare_value
//
// For cross-queue epoch waits the typical usage is:
//   dep_signal = source_queue->epoch.signal
//   condition  = IREE_HSA_SIGNAL_CONDITION_LT
//   compare_value = EPOCH_INITIAL_VALUE - target_epoch + 1
//   mask = INT64_MAX (all non-sign bits)
static inline uint16_t iree_hal_amdgpu_aql_emit_barrier_value(
    iree_hsa_amd_barrier_value_packet_t* packet, iree_hsa_signal_t dep_signal,
    iree_hsa_signal_condition_t condition,
    iree_hsa_signal_value_t compare_value, iree_hsa_signal_value_t mask,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  // Keep the entire first dword (primary header + AmdFormat/reserved) untouched
  // until aql_ring_commit publishes it with release semantics.
  packet->reserved0 = 0;
  packet->signal = dep_signal;
  packet->value = compare_value;
  packet->mask = mask;
  packet->cond = (iree_hsa_signal_condition32_t)condition;
  packet->reserved1 = 0;
  packet->reserved2 = 0;
  packet->reserved3 = 0;
  packet->completion_signal = completion_signal;
  *out_setup = IREE_HSA_AMD_AQL_FORMAT_BARRIER_VALUE;

  // The primary header uses VENDOR_SPECIFIC packet type for AMD extensions.
  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC,
                                         packet_control);
}

// Populates a barrier-AND packet and returns the 16-bit AQL header.
// The barrier halts the CP until all non-null dependency signals reach 0.
// Up to 5 dependency signals are supported per packet.
static inline uint16_t iree_hal_amdgpu_aql_emit_barrier_and(
    iree_hsa_barrier_and_packet_t* packet, const iree_hsa_signal_t* dep_signals,
    uint32_t dep_count, iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal) {
  packet->reserved0 = 0;
  packet->reserved1 = 0;
  // Fill dependency signals, nulling any unused slots.
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(packet->dep_signal); ++i) {
    packet->dep_signal[i] =
        i < dep_count ? dep_signals[i] : iree_hsa_signal_null();
  }
  packet->reserved2 = 0;
  packet->completion_signal = completion_signal;

  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_BARRIER_AND,
                                         packet_control);
}

// Populates a no-op packet in |packet| and returns its 16-bit AQL header.
// Zero-dependency BARRIER_AND is the canonical "consume one slot, do no work"
// packet. Callers choose whether that no-op packet carries a barrier edge and
// what fence scopes it should use.
static inline uint16_t iree_hal_amdgpu_aql_emit_nop(
    iree_hsa_barrier_and_packet_t* packet,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal) {
  return iree_hal_amdgpu_aql_emit_barrier_and(packet, /*dep_signals=*/NULL,
                                              /*dep_count=*/0, packet_control,
                                              completion_signal);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AQL_EMITTER_H_
