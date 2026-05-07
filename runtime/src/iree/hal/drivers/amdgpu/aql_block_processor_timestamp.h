// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_TIMESTAMP_H_
#define IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_TIMESTAMP_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/timestamp.h"
#include "iree/hal/drivers/amdgpu/aql_block_processor.h"
#include "iree/hal/drivers/amdgpu/device/timestamp.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Dispatch timestamp sidecar for one already-recorded dispatch packet.
typedef struct iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t {
  // Precomputed ordinals in the payload and timestamp record streams.
  struct {
    // Payload packet ordinal whose completion signal must be timestamped.
    uint32_t packet_ordinal;
    // Dispatch timestamp record ordinal written into the record header.
    uint32_t record_ordinal;
  } ordinals;
  // Correlation metadata copied into the fixed timestamp record.
  struct {
    // Producer-defined command-buffer identifier, or 0 for direct dispatch.
    uint64_t command_buffer_id;
    // Producer-defined executable identifier, or 0 when unavailable.
    uint64_t executable_id;
    // Command-buffer block ordinal containing this dispatch.
    uint32_t block_ordinal;
    // Program-global command index of this dispatch.
    uint32_t command_index;
    // Executable export ordinal dispatched.
    uint32_t export_ordinal;
    // Flags from iree_hal_amdgpu_dispatch_timestamp_record_flag_bits_t.
    iree_hal_amdgpu_dispatch_timestamp_record_flags_t flags;
  } metadata;
  // Caller-owned storage patched or populated by the timestamp processor.
  struct {
    // Raw completion signal that receives CP dispatch timestamps.
    iree_amd_signal_t* completion_signal;
    // Fixed binary dispatch timestamp record populated by the processor.
    iree_hal_amdgpu_dispatch_timestamp_record_t* record;
  } target;
} iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t;

// Dispatch timestamp sidecars in command order.
typedef struct iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_t {
  // Dispatch timestamp sidecars selected for this block.
  const iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t* values;
  // Number of entries in |values|.
  uint32_t count;
} iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_t;

// Parameters for materializing dispatch timestamp sidecars from retained
// command-buffer dispatch summaries.
typedef struct
    iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_params_t {
  // Retained command-buffer dispatch summaries in command order.
  struct {
    // First retained dispatch summary in a linked block-local list.
    const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* first;
    // Number of retained dispatch summaries expected in |first|.
    uint32_t count;
  } summaries;
  // Timestamp metadata shared by every materialized dispatch sidecar.
  struct {
    // Producer-defined command-buffer identifier used for correlation.
    uint64_t command_buffer_id;
    // Command-buffer block ordinal containing these dispatches.
    uint32_t block_ordinal;
    // First dispatch timestamp record ordinal assigned to this list.
    uint32_t first_record_ordinal;
  } metadata;
  // Caller-owned storage receiving sidecar records and timestamp targets.
  struct {
    // Sidecar array with capacity for |summaries.count| entries.
    iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t* dispatches;
    // Raw completion signal targets with capacity for |summaries.count|
    // entries.
    iree_amd_signal_t* completion_signals;
    // Fixed timestamp records with capacity for |summaries.count| entries.
    iree_hal_amdgpu_dispatch_timestamp_record_t* records;
  } storage;
} iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_params_t;

// Opt-in timestamp processor for one AQL command-buffer block.
typedef struct iree_hal_amdgpu_aql_block_processor_timestamp_t {
  // Base payload processor configuration. Its packet span excludes timestamp
  // prefix, suffix, and harvest packets.
  iree_hal_amdgpu_aql_block_processor_t base;
  // Optional command-buffer/block timestamp record and PM4 packets.
  struct {
    // Correlation metadata copied into the fixed timestamp record.
    struct {
      // Command-buffer timestamp record ordinal written into the record header.
      uint32_t record_ordinal;
      // Producer-defined command-buffer identifier used for correlation.
      uint64_t command_buffer_id;
      // Command-buffer block ordinal, or UINT32_MAX for whole-execute records.
      uint32_t block_ordinal;
    } metadata;
    // Caller-owned storage receiving command-buffer timestamp data.
    struct {
      // Fixed binary command-buffer timestamp record, or NULL when disabled.
      iree_hal_amdgpu_command_buffer_timestamp_record_t* record;
    } target;
    // PM4 packet sequence used for command-buffer timestamp ranges.
    iree_hal_amdgpu_pm4_timestamp_strategy_t pm4_timestamp_strategy;
    // PM4 timestamp packets owned by the enclosing submission.
    struct {
      // Start timestamp packet emitted before the payload span.
      struct {
        // AQL packet receiving the start-timestamp PM4 IB envelope.
        iree_hal_amdgpu_aql_packet_t* packet;
        // PM4 IB slot referenced by |packet|.
        iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot;
        // Packet control used when publishing |packet|.
        iree_hal_amdgpu_aql_packet_control_t control;
      } start;
      // End timestamp packet emitted after the payload and harvest spans.
      struct {
        // AQL packet receiving the end-timestamp PM4 IB envelope.
        iree_hal_amdgpu_aql_packet_t* packet;
        // PM4 IB slot referenced by |packet|.
        iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot;
        // Packet control used when publishing |packet|.
        iree_hal_amdgpu_aql_packet_control_t control;
        // Optional completion signal decremented when |packet| completes.
        iree_hsa_signal_t completion_signal;
      } end;
    } packets;
  } command_buffer;
  // Optional dispatch timestamp records patched into payload packets.
  iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_t dispatches;
  // Optional dispatch-timestamp harvest packet and kernargs.
  struct {
    // Builtin harvest kernel arguments.
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args;
    // AQL dispatch packet emitted after payload dispatches complete.
    iree_hal_amdgpu_aql_packet_t* packet;
    // Queue-owned kernarg storage for the harvest dispatch.
    void* kernarg_ptr;
    // Packet control used when publishing |packet|.
    iree_hal_amdgpu_aql_packet_control_t packet_control;
    // Optional completion signal decremented when |packet| completes.
    iree_hsa_signal_t completion_signal;
  } harvest;
} iree_hal_amdgpu_aql_block_processor_timestamp_t;

// Result of invoking the timestamp processor on one block.
typedef struct iree_hal_amdgpu_aql_block_processor_timestamp_result_t {
  // Result reported by the embedded base payload processor.
  iree_hal_amdgpu_aql_block_processor_result_t base;
  // Command-buffer timestamp packet metadata produced when enabled.
  struct {
    // Start timestamp packet commit metadata.
    struct {
      // AQL packet header for the start timestamp packet.
      uint16_t header;
      // AQL packet setup word for the start timestamp packet.
      uint16_t setup;
    } start;
    // End timestamp packet commit metadata.
    struct {
      // AQL packet header for the end timestamp packet.
      uint16_t header;
      // AQL packet setup word for the end timestamp packet.
      uint16_t setup;
    } end;
  } command_buffer;
  // Dispatch timestamp accounting produced when enabled.
  struct {
    // Number of dispatch timestamp records initialized and harvest sources set.
    uint32_t count;
  } dispatches;
  // Dispatch timestamp harvest packet metadata produced when enabled.
  struct {
    // AQL packet header for the harvest dispatch.
    uint16_t header;
    // AQL packet setup word for the harvest dispatch.
    uint16_t setup;
  } harvest;
} iree_hal_amdgpu_aql_block_processor_timestamp_result_t;

// Initializes |out_processor| with borrowed submission storage.
void iree_hal_amdgpu_aql_block_processor_timestamp_initialize(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* params,
    iree_hal_amdgpu_aql_block_processor_timestamp_t* out_processor);

// Deinitializes |processor|. This currently releases no resources.
void iree_hal_amdgpu_aql_block_processor_timestamp_deinitialize(
    iree_hal_amdgpu_aql_block_processor_timestamp_t* processor);

// Materializes dispatch timestamp sidecars from retained command-buffer
// dispatch summaries and caller-owned target storage.
iree_status_t
iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_initialize(
    const iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_params_t*
        params,
    iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_t*
        out_dispatches);

// Invokes |processor| on |block| and populates payload packets plus any
// timestamp sidecars selected by the caller.
iree_status_t iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_aql_block_processor_timestamp_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_AQL_BLOCK_PROCESSOR_TIMESTAMP_H_
