// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_WAITS_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_WAITS_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stack-allocable frontier storage sized to the host queue frontier capacity.
typedef iree_hal_amdgpu_host_queue_frontier_t iree_hal_amdgpu_fixed_frontier_t;

static inline iree_async_frontier_t* iree_hal_amdgpu_fixed_frontier_as_frontier(
    iree_hal_amdgpu_fixed_frontier_t* storage) {
  return iree_async_fixed_frontier_as_frontier(storage);
}

//===----------------------------------------------------------------------===//
// Wait resolution
//===----------------------------------------------------------------------===//

// A single device-side wait barrier emitted for each undominated local queue
// axis that the current submission must wait on before its own packets run.
typedef struct iree_hal_amdgpu_wait_barrier_t {
  // Producer queue axis to wait on.
  iree_async_axis_t axis;
  // Producer queue epoch signal consumed by the barrier packet.
  hsa_signal_t epoch_signal;
  // Producer queue epoch that must be complete before the barrier releases.
  uint64_t target_epoch;
} iree_hal_amdgpu_wait_barrier_t;

// Result of resolving a wait_semaphore_list. Either all waits are resolved
// with |barriers[0..barrier_count]|, or software deferral is required.
typedef struct iree_hal_amdgpu_wait_resolution_t {
  // Number of valid device-side barriers in |barriers|.
  uint8_t barrier_count;
  // True if at least one wait requires software deferral.
  bool needs_deferral;
  // Padding reserved to keep the fence scopes aligned.
  uint8_t reserved[2];
  // Number of wait semaphore edges represented by this resolution.
  uint32_t wait_count;
  // Queue profiling flags describing how this resolution was reached.
  iree_hal_profile_queue_event_flags_t profile_event_flags;
  // Acquire scope required on the final operation packet for waits resolved
  // without dedicated wait-barrier packets.
  iree_hsa_fence_scope_t inline_acquire_scope;
  // Acquire scope required on dedicated wait-barrier packets.
  iree_hsa_fence_scope_t barrier_acquire_scope;
  // Device-side wait barriers sorted by ascending producer axis.
  iree_hal_amdgpu_wait_barrier_t
      barriers[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
} iree_hal_amdgpu_wait_resolution_t;

// Resolves a wait_semaphore_list into device-side barriers or software
// deferral. Caller must hold submission_mutex.
void iree_hal_amdgpu_host_queue_resolve_waits(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_wait_resolution_t* out_resolution);

// Writes one device-side wait barrier packet body and returns the header/setup
// bits that will publish it. Caller must commit the packet header after this
// returns. Caller must hold submission_mutex.
uint16_t iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_barrier_t* barrier, uint64_t packet_id,
    hsa_signal_t completion_signal, iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope, iree_hal_amdgpu_aql_packet_t* packet,
    uint16_t* out_setup);

// Emits device-side wait barrier packets for a resolved wait list. Caller must
// have reserved |resolution->barrier_count| consecutive AQL slots starting at
// |first_packet_id|. Caller must hold submission_mutex.
void iree_hal_amdgpu_host_queue_emit_barriers(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    uint64_t first_packet_id);

// Merges the resolved wait barrier axes into the queue's accumulated frontier
// after successful submission publication. Caller must hold submission_mutex.
void iree_hal_amdgpu_host_queue_merge_barrier_axes(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution);

// Returns the queue-order frontier to use for pool acquire_reservation() after
// accounting for dependency barriers in |resolution|.
const iree_async_frontier_t* iree_hal_amdgpu_host_queue_pool_requester_frontier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_fixed_frontier_t* storage);

// Imports a pool-owned death frontier into the queue's AQL dependency list.
// Entries already dominated by |requester_frontier| are skipped; remaining
// local-queue axes become device-side wait barriers in |resolution|.
bool iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_async_frontier_t* requester_frontier,
    const iree_async_frontier_t* wait_frontier,
    iree_hal_amdgpu_wait_resolution_t* resolution);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_WAITS_H_
