// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_amdgpu_aql_block_processor_profile_flags_t;
enum iree_hal_amdgpu_aql_block_processor_profile_flag_bits_t {
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_NONE = 0u,
  // This block reserves dispatch timestamp events.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_DISPATCH_PACKETS = 1u << 0,
  // This block reserves a whole-block queue-device timestamp event.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_QUEUE_DEVICE_EVENT = 1u << 1,
};

typedef uint8_t iree_hal_amdgpu_aql_block_processor_profile_terminator_t;
enum iree_hal_amdgpu_aql_block_processor_profile_terminator_e {
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_TERMINATOR_NONE = 0u,
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_TERMINATOR_RETURN = 1u,
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_TERMINATOR_BRANCH = 2u,
};

// Selected command-buffer dispatch for host profile packet augmentation.
typedef struct iree_hal_amdgpu_aql_block_processor_profile_dispatch_t {
  // Retained dispatch summary selected by the active capture filter.
  const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary;
} iree_hal_amdgpu_aql_block_processor_profile_dispatch_t;

// Selected command-buffer dispatches in command order.
typedef struct iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t {
  // Selected dispatch entries allocated from caller-owned scratch storage.
  const iree_hal_amdgpu_aql_block_processor_profile_dispatch_t* values;
  // Number of selected dispatch entries.
  uint32_t count;
} iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t;

// Host-side processor for one profiled AQL command-buffer block.
typedef struct iree_hal_amdgpu_aql_block_processor_profile_t {
  // Host queue owning the AQL ring and profiling slot storage.
  iree_hal_amdgpu_host_queue_t* queue;
  // Command buffer being replayed.
  iree_hal_command_buffer_t* command_buffer;
  // Recorded command-buffer block being emitted.
  const iree_hal_amdgpu_command_buffer_block_header_t* block;
  // Submission-level packet control inputs.
  struct {
    // Wait resolution prefixing this block submission.
    const iree_hal_amdgpu_wait_resolution_t* resolution;
    // Final signal list for this block when it is terminal.
    iree_hal_semaphore_list_t signal_semaphore_list;
  } submission;
  // Queue-execute binding state consumed by dynamic block operands.
  struct {
    // Binding table supplied to queue_execute.
    iree_hal_buffer_binding_table_t table;
    // Pre-resolved binding pointers indexed by queue_execute binding table
    // slot.
    const uint64_t* ptrs;
  } bindings;
  // Reserved packet span populated by profiled replay.
  struct {
    // First reserved AQL payload packet id after wait/profile prefix packets.
    uint64_t first_payload_id;
    // Logical packet index of |first_payload_id| within the full submission.
    uint32_t index_base;
    // Number of reserved payload packet slots available to the processor.
    uint32_t count;
    // Header words produced by profiled replay and published by the caller.
    uint16_t* headers;
    // Setup words produced with |headers|.
    uint16_t* setups;
  } packets;
  // Reserved queue-owned kernarg storage consumed by profiled replay.
  struct {
    // First reserved kernarg block.
    iree_hal_amdgpu_kernarg_block_t* blocks;
    // Number of reserved kernarg blocks.
    uint32_t count;
  } kernargs;
  // Queue-owned payload visibility requirements.
  struct {
    // Minimum acquire scope for replayed payload packets in this block.
    iree_hsa_fence_scope_t acquire_scope;
    // Number of leading recorded payload packets requiring |acquire_scope|.
    uint32_t acquire_packet_count;
  } payload;
  // Host profile sidecars consumed by profiled replay.
  struct {
    // Selected dispatches that receive profile packet augmentation.
    iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t dispatches;
    // Dispatch timestamp event reservation for profiled dispatches.
    iree_hal_amdgpu_profile_dispatch_event_reservation_t dispatch_events;
    // Harvest sources written when dispatch timestamp profiling is active.
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* harvest_sources;
    // Stable command buffer id used in emitted profile records.
    uint64_t command_buffer_id;
    // Number of counter sets emitted around each profiled dispatch.
    uint32_t counter_set_count;
    // Number of executable trace packets emitted across this block.
    uint32_t trace_packet_count;
  } profile;
  // Flags from iree_hal_amdgpu_aql_block_processor_profile_flag_bits_t.
  iree_hal_amdgpu_aql_block_processor_profile_flags_t flags;
} iree_hal_amdgpu_aql_block_processor_profile_t;

// Result of invoking the profiled processor on one block.
typedef struct iree_hal_amdgpu_aql_block_processor_profile_result_t {
  // Packet accounting reported by the processor.
  struct {
    // Number of recorded block AQL packets consumed.
    uint32_t recorded;
    // Number of reserved AQL packets populated.
    uint32_t emitted;
  } packets;
  // Kernarg accounting reported by the processor.
  struct {
    // Number of reserved kernarg blocks consumed.
    uint32_t consumed;
  } kernargs;
  // Profile accounting reported by the processor.
  struct {
    // Number of dispatch profile events emitted.
    uint32_t events;
  } profile;
  // Terminator kind reached by this invocation.
  iree_hal_amdgpu_aql_block_processor_profile_terminator_t terminator;
  // Branch target block ordinal when |terminator| is BRANCH.
  uint32_t target_block_ordinal;
} iree_hal_amdgpu_aql_block_processor_profile_result_t;

// Initializes |out_processor| with borrowed submission storage.
void iree_hal_amdgpu_aql_block_processor_profile_initialize(
    const iree_hal_amdgpu_aql_block_processor_profile_t* params,
    iree_hal_amdgpu_aql_block_processor_profile_t* out_processor);

// Deinitializes |processor|. This currently releases no resources.
void iree_hal_amdgpu_aql_block_processor_profile_deinitialize(
    iree_hal_amdgpu_aql_block_processor_profile_t* processor);

// Invokes |processor| and populates reserved packet/kernarg/profile storage.
iree_status_t iree_hal_amdgpu_aql_block_processor_profile_invoke(
    const iree_hal_amdgpu_aql_block_processor_profile_t* processor,
    iree_hal_amdgpu_aql_block_processor_profile_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_H_
