// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_block_processor_timestamp.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

static bool iree_hal_amdgpu_aql_block_processor_timestamp_has_command_buffer(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor) {
  return processor->command_buffer.target.record != NULL;
}

static bool
iree_hal_amdgpu_aql_block_processor_timestamp_has_command_buffer_storage(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor) {
  return processor->command_buffer.target.record ||
         processor->command_buffer.packets.start.packet ||
         processor->command_buffer.packets.start.pm4_ib_slot ||
         processor->command_buffer.packets.end.packet ||
         processor->command_buffer.packets.end.pm4_ib_slot;
}

static bool iree_hal_amdgpu_aql_block_processor_timestamp_has_dispatches(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor) {
  return processor->dispatches.count != 0;
}

static iree_hal_amdgpu_dispatch_timestamp_record_flags_t
iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_flags(
    const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary) {
  iree_hal_amdgpu_dispatch_timestamp_record_flags_t flags =
      IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_NONE;
  if (iree_any_bit_set(
          summary->metadata.dispatch_flags,
          IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS)) {
    flags |= IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_INDIRECT_PARAMETERS;
  }
  return flags;
}

static iree_hsa_signal_t
iree_hal_amdgpu_aql_block_processor_timestamp_completion_signal(
    const iree_amd_signal_t* signal) {
  return (iree_hsa_signal_t){.handle = (uint64_t)(uintptr_t)signal};
}

static void iree_hal_amdgpu_aql_block_processor_timestamp_initialize_header(
    uint32_t record_length, iree_hal_amdgpu_timestamp_record_type_t type,
    uint32_t record_ordinal,
    iree_hal_amdgpu_timestamp_record_header_t* out_header) {
  out_header->record_length = record_length;
  out_header->version = IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_0;
  out_header->type = type;
  out_header->record_ordinal = record_ordinal;
  out_header->reserved0 = 0;
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_timestamp_validate_command_buffer(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor) {
  if (!iree_hal_amdgpu_aql_block_processor_timestamp_has_command_buffer(
          processor) &&
      iree_hal_amdgpu_aql_block_processor_timestamp_has_command_buffer_storage(
          processor)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer timestamp mode requires a timestamp record target");
  }
  if (!iree_hal_amdgpu_aql_block_processor_timestamp_has_command_buffer(
          processor)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!processor->command_buffer.packets.start.packet ||
                    !processor->command_buffer.packets.start.pm4_ib_slot ||
                    !processor->command_buffer.packets.end.packet ||
                    !processor->command_buffer.packets.end.pm4_ib_slot)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer timestamp mode requires start and end PM4 packet "
        "storage");
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
          processor->command_buffer.pm4_timestamp_strategy))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command-buffer timestamp mode requires a PM4 timestamp range "
        "strategy");
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_block_processor_timestamp_emit_command_buffer(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor,
    iree_hal_amdgpu_aql_block_processor_timestamp_result_t* out_result) {
  iree_hal_amdgpu_command_buffer_timestamp_record_t* record =
      processor->command_buffer.target.record;
  memset(record, 0, sizeof(*record));
  iree_hal_amdgpu_aql_block_processor_timestamp_initialize_header(
      sizeof(*record), IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_COMMAND_BUFFER,
      processor->command_buffer.metadata.record_ordinal, &record->header);
  record->command_buffer_id =
      processor->command_buffer.metadata.command_buffer_id;
  record->block_ordinal = processor->command_buffer.metadata.block_ordinal;

  out_result->command_buffer.start.header =
      iree_hal_amdgpu_aql_emit_timestamp_start(
          &processor->command_buffer.packets.start.packet->pm4_ib,
          processor->command_buffer.packets.start.pm4_ib_slot,
          processor->command_buffer.packets.start.control,
          processor->command_buffer.pm4_timestamp_strategy,
          &record->ticks.start_tick, &out_result->command_buffer.start.setup);
  out_result->command_buffer.end.header =
      iree_hal_amdgpu_aql_emit_timestamp_end(
          &processor->command_buffer.packets.end.packet->pm4_ib,
          processor->command_buffer.packets.end.pm4_ib_slot,
          processor->command_buffer.packets.end.control,
          processor->command_buffer.pm4_timestamp_strategy,
          processor->command_buffer.packets.end.completion_signal,
          &record->ticks.end_tick, &out_result->command_buffer.end.setup);
}

static iree_status_t
iree_hal_amdgpu_aql_block_processor_timestamp_validate_dispatches(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor) {
  if (!iree_hal_amdgpu_aql_block_processor_timestamp_has_dispatches(
          processor)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!processor->dispatches.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch timestamp mode requires dispatch sidecar records");
  }
  if (IREE_UNLIKELY(!processor->harvest.kernel_args ||
                    !processor->harvest.packet ||
                    !processor->harvest.kernarg_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch timestamp mode requires a harvest kernel, packet, and "
        "kernargs");
  }
  for (uint32_t i = 0; i < processor->dispatches.count; ++i) {
    const iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t* dispatch =
        &processor->dispatches.values[i];
    if (IREE_UNLIKELY(dispatch->ordinals.packet_ordinal >=
                      processor->base.packets.count)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch timestamp packet ordinal %" PRIu32
                              " exceeds emitted payload packet count %" PRIu32,
                              dispatch->ordinals.packet_ordinal,
                              processor->base.packets.count);
    }
    if (IREE_UNLIKELY(!dispatch->target.completion_signal ||
                      !dispatch->target.record)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch timestamp mode requires a completion signal and record");
    }
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_aql_block_processor_timestamp_emit_dispatches(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor,
    iree_hal_amdgpu_aql_block_processor_timestamp_result_t* out_result) {
  iree_hal_amdgpu_dispatch_timestamp_harvest_source_t* sources =
      iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
          processor->harvest.kernel_args, processor->dispatches.count,
          &processor->harvest.packet->dispatch, processor->harvest.kernarg_ptr);
  for (uint32_t i = 0; i < processor->dispatches.count; ++i) {
    const iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t* dispatch =
        &processor->dispatches.values[i];
    iree_hal_amdgpu_dispatch_timestamp_record_t* record =
        dispatch->target.record;
    memset(record, 0, sizeof(*record));
    iree_hal_amdgpu_aql_block_processor_timestamp_initialize_header(
        sizeof(*record), IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_DISPATCH,
        dispatch->ordinals.record_ordinal, &record->header);
    record->command_buffer_id = dispatch->metadata.command_buffer_id;
    record->executable_id = dispatch->metadata.executable_id;
    record->block_ordinal = dispatch->metadata.block_ordinal;
    record->command_index = dispatch->metadata.command_index;
    record->export_ordinal = dispatch->metadata.export_ordinal;
    record->flags = dispatch->metadata.flags;

    iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
        processor->base.packets.ring,
        processor->base.packets.first_id + dispatch->ordinals.packet_ordinal);
    packet->dispatch.completion_signal =
        iree_hal_amdgpu_aql_block_processor_timestamp_completion_signal(
            dispatch->target.completion_signal);
    sources[i].completion_signal = dispatch->target.completion_signal;
    sources[i].ticks = &record->ticks;
  }

  processor->harvest.packet->dispatch.completion_signal =
      processor->harvest.completion_signal;
  out_result->dispatches.count = processor->dispatches.count;
  out_result->harvest.header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH, processor->harvest.packet_control);
  out_result->harvest.setup = processor->harvest.packet->dispatch.setup;
}

void iree_hal_amdgpu_aql_block_processor_timestamp_initialize(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* params,
    iree_hal_amdgpu_aql_block_processor_timestamp_t* out_processor) {
  *out_processor = *params;
}

void iree_hal_amdgpu_aql_block_processor_timestamp_deinitialize(
    iree_hal_amdgpu_aql_block_processor_timestamp_t* processor) {
  memset(processor, 0, sizeof(*processor));
}

iree_status_t
iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_initialize(
    const iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_params_t*
        params,
    iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_t*
        out_dispatches) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_dispatches);
  memset(out_dispatches, 0, sizeof(*out_dispatches));
  if (params->summaries.count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!params->summaries.first || !params->storage.dispatches ||
                    !params->storage.completion_signals ||
                    !params->storage.records)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch timestamp list requires summaries and target storage");
  }
  if (IREE_UNLIKELY(params->metadata.first_record_ordinal >
                    UINT32_MAX - (params->summaries.count - 1u))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch timestamp record ordinal range overflows uint32_t");
  }

  const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary =
      params->summaries.first;
  for (uint32_t summary_ordinal = 0; summary_ordinal < params->summaries.count;
       ++summary_ordinal) {
    if (IREE_UNLIKELY(!summary)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "retained dispatch summary list ended after %" PRIu32 " of %" PRIu32
          " entries",
          summary_ordinal, params->summaries.count);
    }

    iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t* dispatch =
        &params->storage.dispatches[summary_ordinal];
    memset(dispatch, 0, sizeof(*dispatch));
    dispatch->ordinals.packet_ordinal = summary->packets.dispatch_ordinal;
    dispatch->ordinals.record_ordinal =
        params->metadata.first_record_ordinal + summary_ordinal;
    dispatch->metadata.command_buffer_id = params->metadata.command_buffer_id;
    dispatch->metadata.executable_id = summary->metadata.executable_id;
    dispatch->metadata.block_ordinal = params->metadata.block_ordinal;
    dispatch->metadata.command_index = summary->metadata.command_index;
    dispatch->metadata.export_ordinal = summary->metadata.export_ordinal;
    dispatch->metadata.flags =
        iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_flags(summary);
    dispatch->target.completion_signal =
        &params->storage.completion_signals[summary_ordinal];
    dispatch->target.record = &params->storage.records[summary_ordinal];
    summary = summary->next;
  }

  out_dispatches->values = params->storage.dispatches;
  out_dispatches->count = params->summaries.count;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
    const iree_hal_amdgpu_aql_block_processor_timestamp_t* processor,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_aql_block_processor_timestamp_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_timestamp_validate_command_buffer(
          processor);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_aql_block_processor_timestamp_validate_dispatches(
        processor);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_aql_block_processor_invoke(&processor->base, block,
                                                        &out_result->base);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_aql_block_processor_timestamp_has_command_buffer(
          processor)) {
    iree_hal_amdgpu_aql_block_processor_timestamp_emit_command_buffer(
        processor, out_result);
  }
  if (iree_status_is_ok(status) &&
      iree_hal_amdgpu_aql_block_processor_timestamp_has_dispatches(processor)) {
    iree_hal_amdgpu_aql_block_processor_timestamp_emit_dispatches(processor,
                                                                  out_result);
  }
  return status;
}
