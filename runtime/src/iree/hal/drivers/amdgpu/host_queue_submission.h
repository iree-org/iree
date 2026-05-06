// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_SUBMISSION_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_SUBMISSION_H_

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_host_queue_profile_event_info_t
    iree_hal_amdgpu_host_queue_profile_event_info_t;

typedef void(IREE_API_PTR* iree_hal_amdgpu_host_queue_post_commit_fn_t)(
    void* user_data, const iree_async_frontier_t* queue_frontier,
    uint64_t submission_id);

// Optional callback invoked after queue frontier state has advanced and before
// the completion packet is published.
typedef struct iree_hal_amdgpu_host_queue_post_commit_callback_t {
  // Function invoked with the queue frontier and submission id visible after
  // commit.
  iree_hal_amdgpu_host_queue_post_commit_fn_t fn;

  // Opaque user data passed to |fn|.
  void* user_data;
} iree_hal_amdgpu_host_queue_post_commit_callback_t;

// Returns a null post-commit callback.
static inline iree_hal_amdgpu_host_queue_post_commit_callback_t
iree_hal_amdgpu_host_queue_post_commit_callback_null(void) {
  iree_hal_amdgpu_host_queue_post_commit_callback_t callback = {
      .fn = NULL,
      .user_data = NULL,
  };
  return callback;
}

// Flags controlling submission helper ownership transfers.
typedef uint32_t iree_hal_amdgpu_host_queue_submission_flags_t;
enum iree_hal_amdgpu_host_queue_submission_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE = 0u,
  // Retains signal semaphores and operation resources into the reclaim entry.
  // When omitted, the helper transfers one existing retain for each resource
  // from the caller into the reclaim entry on success.
  IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES = 1u << 0,
};

// One in-flight kernel-shaped packet submission assembled under
// submission_mutex. Owns the generic notification/reclaim, AQL reservation, and
// kernarg reservation state shared by direct queue dispatches and
// command-buffer replay.
typedef struct iree_hal_amdgpu_host_queue_kernel_submission_t {
  // Reclaim entry reserved from the notification ring for this submission.
  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry;
  // Reclaim resource slots owned by |reclaim_entry|.
  iree_hal_resource_t** reclaim_resources;
  // Queue-owned kernarg reservation for this submission.
  struct {
    // First kernarg block reserved for this submission, or NULL when unused.
    iree_hal_amdgpu_kernarg_block_t* blocks;
    // Kernarg ring write position to reclaim after this submission completes.
    uint64_t write_position;
  } kernargs;
  // First AQL packet id reserved for this submission.
  uint64_t first_packet_id;
  // Queue-control upload reservation for this submission.
  struct {
    // Upload ring write position to reclaim after this submission completes.
    uint64_t write_position;
  } queue_upload;
  // Number of AQL packets reserved starting at |first_packet_id|.
  uint32_t packet_count;
  // Number of valid entries in |reclaim_resources|.
  uint16_t reclaim_resource_count;
  // Optional action executed before user signals are published when this
  // submission completes.
  iree_hal_amdgpu_reclaim_action_t pre_signal_action;
} iree_hal_amdgpu_host_queue_kernel_submission_t;

// One in-flight barrier-shaped packet submission assembled under
// submission_mutex. Owns the generic notification/reclaim and AQL reservation
// state for submissions that complete with a barrier or no-op packet and do not
// require queue-owned kernarg storage.
typedef struct iree_hal_amdgpu_host_queue_barrier_submission_t {
  // Reclaim entry reserved from the notification ring for this submission.
  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry;
  // Reclaim resource slots owned by |reclaim_entry|.
  iree_hal_resource_t** reclaim_resources;
  // Queue device profile event reservation for this submission.
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events;
  // First AQL packet id reserved for this submission.
  uint64_t first_packet_id;
  // Number of AQL packets reserved starting at |first_packet_id|.
  uint32_t packet_count;
  // Number of valid entries in |reclaim_resources|.
  uint16_t reclaim_resource_count;
} iree_hal_amdgpu_host_queue_barrier_submission_t;

// One in-flight single-dispatch submission assembled under submission_mutex.
// Operation implementations populate the dispatch packet and kernargs directly
// while generic ownership and publication stay in |kernel|.
typedef struct iree_hal_amdgpu_host_queue_dispatch_submission_t {
  // Generic kernel-shaped submission state.
  iree_hal_amdgpu_host_queue_kernel_submission_t kernel;
  // Queue device profile event reservation for this submission.
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events;
  // Packet id of |dispatch_slot|.
  uint64_t dispatch_packet_id;
  // Uncommitted dispatch payload AQL slot.
  iree_hal_amdgpu_aql_packet_t* dispatch_slot;
  // Optional trailing harvest slot used when |dispatch_slot| completes with a
  // profiling-owned signal instead of the queue epoch signal.
  iree_hal_amdgpu_aql_packet_t* profile_harvest_slot;
  // Queue-owned kernarg blocks reserved for |profile_harvest_slot|.
  iree_hal_amdgpu_kernarg_block_t* profile_harvest_kernarg_blocks;
  // Dispatch profile event reservation harvested by |profile_harvest_slot|.
  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events;
  // Number of counter sets captured around |dispatch_slot|.
  uint32_t profile_counter_set_count;
  // Number of executable trace start packets before |dispatch_slot|.
  uint32_t profile_trace_start_packet_count;
  // Completion signal to write into |dispatch_slot|.
  iree_hsa_signal_t dispatch_completion_signal;
  // Setup bits published with |dispatch_slot|'s final header.
  uint16_t dispatch_setup;
  // Setup bits published with |profile_harvest_slot|'s final header.
  uint16_t profile_harvest_setup;
  // Minimum acquire fence scope required by operation-local data visibility.
  iree_hsa_fence_scope_t minimum_acquire_scope;
  // Minimum release fence scope required by operation-local data visibility.
  iree_hsa_fence_scope_t minimum_release_scope;
} iree_hal_amdgpu_host_queue_dispatch_submission_t;

// One in-flight PM4-IB payload submission assembled under submission_mutex.
// Operation implementations append payload packets through |pm4_ib_builder|
// while generic ownership and publication stay in |kernel|.
typedef struct iree_hal_amdgpu_host_queue_pm4_ib_submission_t {
  // Generic payload-shaped submission state.
  iree_hal_amdgpu_host_queue_kernel_submission_t kernel;
  // Queue device profile event reservation for this submission.
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events;
  // Uncommitted PM4-IB payload AQL slot.
  iree_hal_amdgpu_aql_packet_t* pm4_ib_packet_slot;
  // Queue-owned PM4 IB storage referenced by |pm4_ib_packet_slot|.
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot;
  // Bounded builder for appending PM4 packets to |pm4_ib_slot|.
  iree_hal_amdgpu_pm4_ib_builder_t pm4_ib_builder;
} iree_hal_amdgpu_host_queue_pm4_ib_submission_t;

// Returns the number of retained resources required for a submission with
// |signal_semaphore_count| user-visible signal semaphores and
// |operation_resource_count| additional operation-owned resources.
iree_status_t iree_hal_amdgpu_host_queue_count_reclaim_resources(
    iree_host_size_t signal_semaphore_count,
    iree_host_size_t operation_resource_count,
    uint16_t* out_reclaim_resource_count);

// Attempts to begin one kernel-shaped packet submission without waiting for
// ring capacity. If temporary AQL/notification capacity is unavailable then
// |out_ready| is set to false, no queue state is mutated, and OK is returned.
// Any non-OK status is a real structural failure rather than retry state.
//
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count, uint32_t payload_packet_count,
    uint32_t kernarg_block_count, bool* out_ready,
    iree_hal_amdgpu_host_queue_kernel_submission_t* out_submission);

// Publishes host-populated queue-owned kernargs before committing packet
// headers that reference them. Callers must have already written all kernarg
// bytes for |submission|.
static inline void iree_hal_amdgpu_host_queue_publish_submission_kernargs(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_host_queue_kernel_submission_t* submission) {
  if (submission->kernargs.blocks) {
    iree_hal_amdgpu_kernarg_ring_publish_host_writes(&queue->kernarg_ring);
  }
}

// Returns the acquire scope required for device execution to observe
// host-populated queue-owned kernargs. Device-local rings need SYSTEM acquire
// here because publish_submission_kernargs() only drains host writes before
// packet publication; shader-visible memory may otherwise retain stale
// contents across ring-slot reuse.
static inline iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hsa_fence_scope_t minimum_acquire_scope) {
  if (iree_hal_amdgpu_kernarg_ring_requires_host_write_publication(
          &queue->kernarg_ring)) {
    return iree_hal_amdgpu_host_queue_max_fence_scope(
        minimum_acquire_scope, IREE_HSA_FENCE_SCOPE_SYSTEM);
  }
  return minimum_acquire_scope;
}

// Attempts to begin one barrier-shaped packet submission without waiting for
// ring capacity. If temporary AQL/notification capacity is unavailable then
// |out_ready| is set to false, no queue state is mutated, and OK is returned.
// Any non-OK status is a structural failure rather than retry state.
//
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    bool* out_ready,
    iree_hal_amdgpu_host_queue_barrier_submission_t* out_submission);

// Publishes a barrier-shaped packet submission and returns its queue submission
// epoch. Caller must hold submission_mutex.
uint64_t iree_hal_amdgpu_host_queue_finish_barrier_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    iree_hal_amdgpu_host_queue_post_commit_callback_t post_commit_callback,
    iree_hal_resource_set_t* resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_barrier_submission_t* submission);

// Emits no-op packets for a barrier-shaped submission whose AQL slots were
// reserved but whose payload could not be published. User signal semaphores are
// not signaled.
void iree_hal_amdgpu_host_queue_fail_barrier_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_barrier_submission_t* submission);

// Publishes the software side of a kernel-shaped packet submission: transfers
// operation resources and an optional resource set to the reclaim entry,
// advances queue/frontier state, and records user-visible signal metadata.
// Payload packet headers remain uncommitted when this returns. Caller must hold
// submission_mutex.
uint64_t iree_hal_amdgpu_host_queue_finish_kernel_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_resource_set_t** inout_resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_kernel_submission_t* submission);

// Emits reclaim-only no-op packets for a kernel-shaped submission whose AQL and
// kernarg slots were reserved but whose payload could not be published. User
// signal semaphores are not signaled; only queue-private reclaim can advance.
void iree_hal_amdgpu_host_queue_fail_kernel_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_kernel_submission_t* submission);

// Publishes wait-barrier packets for a successful kernel-shaped submission.
// Caller must have already populated payload packet bodies but not committed
// payload packet headers.
void iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_amdgpu_host_queue_kernel_submission_t* submission);

// Commits a non-final queue-device start timestamp packet. The packet must be
// reserved in |submission| and precede the queue operation payload.
void iree_hal_amdgpu_host_queue_commit_queue_device_start_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution, uint64_t packet_id,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event);

// Commits the final queue-device end timestamp packet and queue completion.
// The packet must be reserved in |submission| and follow the queue operation
// payload.
void iree_hal_amdgpu_host_queue_commit_queue_device_end_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list, uint64_t packet_id,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event);

// Attempts to begin one kernel-dispatch submission without waiting for ring
// capacity. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count, uint32_t kernarg_block_count,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    bool* out_ready,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* out_submission);

// Attempts to begin one PM4-IB payload submission without waiting for ring
// capacity. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    bool* out_ready,
    iree_hal_amdgpu_host_queue_pm4_ib_submission_t* out_submission);

// Emits reclaim-only no-op packets for a PM4-IB submission whose AQL slots were
// reserved but whose payload could not be published. User signal semaphores are
// not signaled; only queue-private reclaim can advance.
void iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_pm4_ib_submission_t* submission);

// Writes one final dispatch packet body into an AQL slot in forward field
// order and returns the setup bits that must be published with the header.
uint16_t iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
    iree_hsa_kernel_dispatch_packet_t* IREE_RESTRICT dispatch_packet,
    const iree_hsa_kernel_dispatch_packet_t* IREE_RESTRICT
        dispatch_packet_template,
    void* kernarg_address, iree_hsa_signal_t completion_signal);

// Finishes a submission by transferring retained resources to the reclaim
// entry, publishing queue/semaphore frontier state, committing the final
// dispatch header, ringing the doorbell, and returning the assigned queue
// submission epoch. Caller must hold submission_mutex.
uint64_t iree_hal_amdgpu_host_queue_finish_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* submission);

// Finishes a PM4-IB payload submission by transferring retained resources to
// the reclaim entry, publishing queue/semaphore frontier state, committing the
// final PM4-IB packet header, ringing the doorbell, and writing the assigned
// queue submission epoch to |out_submission_epoch|. Caller must hold
// submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_pm4_ib_submission_t* submission,
    uint64_t* out_submission_epoch);

// Emits one kernel-dispatch submission using an already-prepared packet shape
// and kernargs blob. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_dispatch_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hsa_kernel_dispatch_packet_t* dispatch_packet_template,
    const void* kernargs, iree_host_size_t kernarg_length,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready, uint64_t* out_submission_id);

// Attempts to submit a barrier-only operation without waiting for temporary
// ring capacity. On not-ready, no queue state or ownership is mutated.
// Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_try_submit_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    iree_hal_amdgpu_host_queue_post_commit_callback_t post_commit_callback,
    iree_hal_resource_set_t* resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready, uint64_t* out_submission_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_SUBMISSION_H_
