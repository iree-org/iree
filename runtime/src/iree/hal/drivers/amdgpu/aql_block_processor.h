// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/kernarg_ring.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_amdgpu_aql_block_processor_flags_t;
enum iree_hal_amdgpu_aql_block_processor_flag_bits_t {
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE = 0u,
  // The final recorded payload packet carries terminal signal release scope.
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET = 1u << 0,
};

typedef uint8_t iree_hal_amdgpu_aql_block_processor_terminator_t;
enum iree_hal_amdgpu_aql_block_processor_terminator_e {
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_NONE = 0u,
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN = 1u,
  IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_BRANCH = 2u,
};

// Host-side base processor for one AQL command-buffer block.
typedef struct iree_hal_amdgpu_aql_block_processor_t {
  // Device helper kernel context used for transfer and patch dispatches.
  const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context;
  // Command buffer that owns static buffers and retained rodata.
  iree_hal_command_buffer_t* command_buffer;
  // Queue-execute binding state consumed by dynamic block operands.
  struct {
    // Binding table supplied to queue_execute.
    iree_hal_buffer_binding_table_t table;
    // Pre-resolved binding pointers indexed by queue_execute binding table
    // slot.
    const uint64_t* ptrs;
  } bindings;
  // Reserved packet span populated by the processor.
  struct {
    // AQL ring containing the reserved packet span.
    iree_hal_amdgpu_aql_ring_t* ring;
    // First reserved packet id in |ring|.
    uint64_t first_id;
    // Logical packet index of |first_id| within the full submission.
    uint32_t index_base;
    // Number of reserved packet slots available to the processor.
    uint32_t count;
    // Header words produced by the processor and published by the caller.
    uint16_t* headers;
    // Setup words produced with |headers|.
    uint16_t* setups;
  } packets;
  // Reserved queue-owned kernarg storage consumed by the processor.
  struct {
    // First reserved kernarg block.
    iree_hal_amdgpu_kernarg_block_t* blocks;
    // Number of reserved kernarg blocks.
    uint32_t count;
  } kernargs;
  // Submission overlay precomputed by host queue policy.
  struct {
    // Number of wait-barrier prefix packets before the payload span.
    uint32_t wait_barrier_count;
    // Acquire scope required by inline wait payloads on logical packet zero.
    iree_hsa_fence_scope_t inline_acquire_scope;
    // Release scope required when the payload span carries terminal signals.
    iree_hsa_fence_scope_t signal_release_scope;
  } submission;
  // Queue-owned payload visibility requirements.
  struct {
    // Minimum acquire scope required for queue-owned kernarg visibility.
    iree_hsa_fence_scope_t acquire_scope;
  } payload;
  // Flags from iree_hal_amdgpu_aql_block_processor_flag_bits_t.
  iree_hal_amdgpu_aql_block_processor_flags_t flags;
} iree_hal_amdgpu_aql_block_processor_t;

// Result of invoking the processor on one block.
typedef struct iree_hal_amdgpu_aql_block_processor_result_t {
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
  // Terminator kind reached by this invocation.
  iree_hal_amdgpu_aql_block_processor_terminator_t terminator;
  // Branch target block ordinal when |terminator| is BRANCH.
  uint32_t target_block_ordinal;
} iree_hal_amdgpu_aql_block_processor_result_t;

// Initializes |out_processor| with borrowed submission storage.
void iree_hal_amdgpu_aql_block_processor_initialize(
    const iree_hal_amdgpu_aql_block_processor_t* params,
    iree_hal_amdgpu_aql_block_processor_t* out_processor);

// Deinitializes |processor|. This currently releases no resources.
void iree_hal_amdgpu_aql_block_processor_deinitialize(
    iree_hal_amdgpu_aql_block_processor_t* processor);

// Invokes |processor| on |block| and populates reserved packet/kernarg storage.
iree_status_t iree_hal_amdgpu_aql_block_processor_invoke(
    const iree_hal_amdgpu_aql_block_processor_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_aql_block_processor_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_H_
