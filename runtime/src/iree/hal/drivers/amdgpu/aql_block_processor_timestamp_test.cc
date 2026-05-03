// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_block_processor_timestamp.h"

#include <cstring>

#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct DirectDispatchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Custom-direct dispatch command under test.
  iree_hal_amdgpu_command_buffer_dispatch_command_t dispatch_command;
  // Inline custom-direct kernarg tail copied into queue-owned kernargs.
  uint64_t tail[2];
  // Return terminator following the dispatch command and inline tail.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
};

static void InitializeBlockHeader(
    uint32_t block_length, uint32_t command_length, uint16_t command_count,
    uint32_t aql_packet_count, uint32_t kernarg_length,
    iree_hal_amdgpu_command_buffer_block_header_t* out_header) {
  std::memset(out_header, 0, sizeof(*out_header));
  out_header->magic = IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_MAGIC;
  out_header->version = IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_VERSION_0;
  out_header->header_length = sizeof(*out_header);
  out_header->block_length = block_length;
  out_header->command_offset = sizeof(*out_header);
  out_header->command_length = command_length;
  out_header->command_count = command_count;
  out_header->aql_packet_count = aql_packet_count;
  out_header->kernarg_length = kernarg_length;
  out_header->initial_barrier_packet_count = aql_packet_count;
  out_header->binding_source_offset = block_length;
  out_header->rodata_offset = block_length;
}

static void InitializeReturnCommand(
    iree_hal_amdgpu_command_buffer_return_command_t* out_command) {
  std::memset(out_command, 0, sizeof(*out_command));
  out_command->header.opcode = IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN;
  out_command->header.length_qwords =
      sizeof(*out_command) / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
}

static DirectDispatchBlock MakeDirectDispatchBlock() {
  DirectDispatchBlock block;
  const uint32_t dispatch_command_length =
      sizeof(block.dispatch_command) + sizeof(block.tail);
  const uint32_t command_length =
      dispatch_command_length + sizeof(block.return_command);
  InitializeBlockHeader(sizeof(block), command_length, /*command_count=*/2,
                        /*aql_packet_count=*/1,
                        /*kernarg_length=*/sizeof(block.tail), &block.header);

  std::memset(&block.dispatch_command, 0, sizeof(block.dispatch_command));
  block.dispatch_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH;
  block.dispatch_command.header.flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS;
  block.dispatch_command.header.command_index = 12;
  block.dispatch_command.header.length_qwords =
      dispatch_command_length / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  block.dispatch_command.kernel_object = 0x123456789ABCDEF0ull;
  block.dispatch_command.payload_reference = sizeof(block.dispatch_command);
  block.dispatch_command.kernarg_length_qwords =
      sizeof(block.tail) / sizeof(uint64_t);
  block.dispatch_command.payload.tail_length_qwords =
      sizeof(block.tail) / sizeof(uint64_t);
  block.dispatch_command.kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
  block.dispatch_command.implicit_args_offset_qwords = UINT16_MAX;
  block.dispatch_command.setup = 3;
  block.dispatch_command.workgroup_size[0] = 4;
  block.dispatch_command.workgroup_size[1] = 2;
  block.dispatch_command.workgroup_size[2] = 1;
  block.dispatch_command.grid_size[0] = 64;
  block.dispatch_command.grid_size[1] = 8;
  block.dispatch_command.grid_size[2] = 1;
  block.tail[0] = 0x0A0B0C0D0E0F1011ull;
  block.tail[1] = 0x1213141516171819ull;

  block.header.dispatch_count = 1;
  block.header.terminator_opcode = IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN;
  InitializeReturnCommand(&block.return_command);
  return block;
}

static iree_hal_amdgpu_device_kernel_args_t MakeHarvestKernelArgs() {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = 0x12345678ull;
  kernel_args.setup = 2;
  kernel_args.workgroup_size[0] = 32;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.kernarg_alignment = 16;
  return kernel_args;
}

static iree_hal_amdgpu_aql_block_processor_t MakeBaseProcessor(
    iree_hal_amdgpu_aql_ring_t* ring, uint16_t* packet_headers,
    uint16_t* packet_setups, iree_hal_amdgpu_kernarg_block_t* kernarg_blocks) {
  iree_hal_amdgpu_aql_block_processor_t processor = {};
  processor.packets.ring = ring;
  processor.packets.first_id = 4;
  processor.packets.index_base = 0;
  processor.packets.count = 1;
  processor.packets.headers = packet_headers;
  processor.packets.setups = packet_setups;
  processor.kernargs.blocks = kernarg_blocks;
  processor.kernargs.count = 1;
  processor.submission.signal_release_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  processor.payload.acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  processor.flags =
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET;
  return processor;
}

static uint16_t AqlHeaderField(uint16_t header, uint32_t bit_offset,
                               uint32_t bit_width) {
  return (header >> bit_offset) & ((1u << bit_width) - 1u);
}

static iree_hsa_packet_type_t AqlHeaderType(uint16_t header) {
  return (iree_hsa_packet_type_t)AqlHeaderField(
      header, IREE_HSA_PACKET_HEADER_TYPE, IREE_HSA_PACKET_HEADER_WIDTH_TYPE);
}

TEST(AqlBlockProcessorTimestampTest,
     CommandBufferTimestampInitializesRecordAndPackets) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  iree_hal_amdgpu_pm4_ib_slot_t pm4_ib_slots[8] = {};
  uint16_t packet_headers[1] = {};
  uint16_t packet_setups[1] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[1] = {};
  iree_hal_amdgpu_command_buffer_timestamp_record_t record = {};

  iree_hal_amdgpu_aql_block_processor_timestamp_t processor = {};
  processor.base =
      MakeBaseProcessor(&ring, packet_headers, packet_setups, kernarg_blocks);
  processor.command_buffer.metadata.record_ordinal = 7;
  processor.command_buffer.metadata.command_buffer_id = 0xCAFEull;
  processor.command_buffer.metadata.block_ordinal = 3;
  processor.command_buffer.target.record = &record;
  processor.command_buffer.pm4_timestamp_strategy =
      IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM;
  processor.command_buffer.packets.start.packet = &packets[2];
  processor.command_buffer.packets.start.pm4_ib_slot = &pm4_ib_slots[2];
  processor.command_buffer.packets.start.control =
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                 IREE_HSA_FENCE_SCOPE_NONE);
  processor.command_buffer.packets.end.packet = &packets[6];
  processor.command_buffer.packets.end.pm4_ib_slot = &pm4_ib_slots[6];
  processor.command_buffer.packets.end.control =
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_NONE,
                                                 IREE_HSA_FENCE_SCOPE_SYSTEM);
  processor.command_buffer.packets.end.completion_signal.handle = 0x1234;

  iree_hal_amdgpu_aql_block_processor_timestamp_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.base.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(record.header.record_length, sizeof(record));
  EXPECT_EQ(record.header.version, IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_0);
  EXPECT_EQ(record.header.type,
            IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_COMMAND_BUFFER);
  EXPECT_EQ(record.header.record_ordinal, 7u);
  EXPECT_EQ(record.command_buffer_id, 0xCAFEull);
  EXPECT_EQ(record.block_ordinal, 3u);
  EXPECT_EQ(record.ticks.start_tick, 0u);
  EXPECT_EQ(record.ticks.end_tick, 0u);

  EXPECT_EQ(AqlHeaderType(result.command_buffer.start.header),
            IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC);
  EXPECT_EQ(result.command_buffer.start.setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packets[2].pm4_ib.completion_signal.handle, 0u);
  EXPECT_EQ(AqlHeaderType(result.command_buffer.end.header),
            IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC);
  EXPECT_EQ(result.command_buffer.end.setup, IREE_HSA_AMD_AQL_FORMAT_PM4_IB);
  EXPECT_EQ(packets[6].pm4_ib.completion_signal.handle, 0x1234u);
}

TEST(AqlBlockProcessorTimestampTest,
     RejectsMissingCommandBufferTimestampStrategy) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  iree_hal_amdgpu_pm4_ib_slot_t pm4_ib_slots[8] = {};
  uint16_t packet_headers[1] = {};
  uint16_t packet_setups[1] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[1] = {};
  iree_hal_amdgpu_command_buffer_timestamp_record_t record = {};

  iree_hal_amdgpu_aql_block_processor_timestamp_t processor = {};
  processor.base =
      MakeBaseProcessor(&ring, packet_headers, packet_setups, kernarg_blocks);
  processor.command_buffer.target.record = &record;
  processor.command_buffer.packets.start.packet = &packets[2];
  processor.command_buffer.packets.start.pm4_ib_slot = &pm4_ib_slots[2];
  processor.command_buffer.packets.end.packet = &packets[6];
  processor.command_buffer.packets.end.pm4_ib_slot = &pm4_ib_slots[6];

  iree_hal_amdgpu_aql_block_processor_timestamp_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
                            &processor, &block.header, &result));
}

TEST(AqlBlockProcessorTimestampTest,
     DispatchTimestampPatchesCompletionSignalAndHarvestSource) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[1] = {};
  uint16_t packet_setups[1] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[2] = {};
  iree_amd_signal_t completion_signal = {};
  iree_hal_amdgpu_dispatch_timestamp_record_t record = {};
  iree_hal_amdgpu_device_kernel_args_t harvest_kernel_args =
      MakeHarvestKernelArgs();

  iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t dispatch = {};
  dispatch.ordinals.packet_ordinal = 0;
  dispatch.ordinals.record_ordinal = 5;
  dispatch.metadata.command_buffer_id = 0xBEEF;
  dispatch.metadata.executable_id = 0xFEED;
  dispatch.metadata.block_ordinal = 9;
  dispatch.metadata.command_index = block.dispatch_command.header.command_index;
  dispatch.metadata.export_ordinal = 4;
  dispatch.target.completion_signal = &completion_signal;
  dispatch.target.record = &record;

  iree_hal_amdgpu_aql_block_processor_timestamp_t processor = {};
  processor.base =
      MakeBaseProcessor(&ring, packet_headers, packet_setups, kernarg_blocks);
  processor.dispatches.values = &dispatch;
  processor.dispatches.count = 1;
  processor.harvest.kernel_args = &harvest_kernel_args;
  processor.harvest.packet = &packets[5];
  processor.harvest.kernarg_ptr = kernarg_blocks[1].data;
  processor.harvest.packet_control = iree_hal_amdgpu_aql_packet_control_barrier(
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_SYSTEM);
  processor.harvest.completion_signal.handle = 0x1234;

  iree_hal_amdgpu_aql_block_processor_timestamp_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.base.packets.emitted, 1u);
  EXPECT_EQ(result.dispatches.count, 1u);
  EXPECT_EQ(packets[4].dispatch.completion_signal.handle,
            (uint64_t)(uintptr_t)&completion_signal);
  EXPECT_EQ(record.header.record_length, sizeof(record));
  EXPECT_EQ(record.header.version, IREE_HAL_AMDGPU_TIMESTAMP_RECORD_VERSION_0);
  EXPECT_EQ(record.header.type, IREE_HAL_AMDGPU_TIMESTAMP_RECORD_TYPE_DISPATCH);
  EXPECT_EQ(record.header.record_ordinal, 5u);
  EXPECT_EQ(record.command_buffer_id, 0xBEEFull);
  EXPECT_EQ(record.executable_id, 0xFEEDull);
  EXPECT_EQ(record.block_ordinal, 9u);
  EXPECT_EQ(record.command_index, block.dispatch_command.header.command_index);
  EXPECT_EQ(record.export_ordinal, 4u);
  EXPECT_EQ(record.flags, IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_NONE);
  EXPECT_EQ(record.ticks.start_tick, 0u);
  EXPECT_EQ(record.ticks.end_tick, 0u);

  const auto* harvest_args = reinterpret_cast<
      const iree_hal_amdgpu_dispatch_timestamp_harvest_args_t*>(
      kernarg_blocks[1].data);
  ASSERT_EQ(harvest_args->source_count, 1u);
  const iree_hal_amdgpu_dispatch_timestamp_harvest_source_t* source =
      harvest_args->sources;
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source[0].completion_signal, &completion_signal);
  EXPECT_EQ(source[0].ticks, &record.ticks);
  EXPECT_EQ(packets[5].dispatch.completion_signal.handle, 0x1234u);
  EXPECT_EQ(AqlHeaderType(result.harvest.header),
            IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH);
  EXPECT_EQ(result.harvest.setup, harvest_kernel_args.setup);
}

TEST(AqlBlockProcessorTimestampTest,
     DispatchListInitializesSidecarsFromSummaries) {
  iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t summaries[2] = {};
  summaries[0].next = &summaries[1];
  summaries[0].packets.first_ordinal = 0;
  summaries[0].packets.dispatch_ordinal = 0;
  summaries[0].metadata.executable_id = 0xA0;
  summaries[0].metadata.command_index = 12;
  summaries[0].metadata.export_ordinal = 2;
  summaries[0].metadata.dispatch_flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_NONE;
  summaries[1].packets.first_ordinal = 1;
  summaries[1].packets.dispatch_ordinal = 2;
  summaries[1].metadata.executable_id = 0xB0;
  summaries[1].metadata.command_index = 13;
  summaries[1].metadata.export_ordinal = 4;
  summaries[1].metadata.dispatch_flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS;

  iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t dispatches[2] = {};
  iree_amd_signal_t completion_signals[2] = {};
  iree_hal_amdgpu_dispatch_timestamp_record_t records[2] = {};
  const iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_params_t
      params = {
          .summaries =
              {
                  .first = summaries,
                  .count = 2,
              },
          .metadata =
              {
                  .command_buffer_id = 0xCAFE,
                  .block_ordinal = 5,
                  .first_record_ordinal = 9,
              },
          .storage =
              {
                  .dispatches = dispatches,
                  .completion_signals = completion_signals,
                  .records = records,
              },
      };

  iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_t list;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_list_initialize(
          &params, &list));

  ASSERT_EQ(list.values, dispatches);
  ASSERT_EQ(list.count, 2u);
  EXPECT_EQ(dispatches[0].ordinals.packet_ordinal, 0u);
  EXPECT_EQ(dispatches[0].ordinals.record_ordinal, 9u);
  EXPECT_EQ(dispatches[0].metadata.command_buffer_id, 0xCAFEull);
  EXPECT_EQ(dispatches[0].metadata.executable_id, 0xA0ull);
  EXPECT_EQ(dispatches[0].metadata.block_ordinal, 5u);
  EXPECT_EQ(dispatches[0].metadata.command_index, 12u);
  EXPECT_EQ(dispatches[0].metadata.export_ordinal, 2u);
  EXPECT_EQ(dispatches[0].metadata.flags,
            IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_NONE);
  EXPECT_EQ(dispatches[0].target.completion_signal, &completion_signals[0]);
  EXPECT_EQ(dispatches[0].target.record, &records[0]);

  EXPECT_EQ(dispatches[1].ordinals.packet_ordinal, 2u);
  EXPECT_EQ(dispatches[1].ordinals.record_ordinal, 10u);
  EXPECT_EQ(dispatches[1].metadata.command_buffer_id, 0xCAFEull);
  EXPECT_EQ(dispatches[1].metadata.executable_id, 0xB0ull);
  EXPECT_EQ(dispatches[1].metadata.block_ordinal, 5u);
  EXPECT_EQ(dispatches[1].metadata.command_index, 13u);
  EXPECT_EQ(dispatches[1].metadata.export_ordinal, 4u);
  EXPECT_EQ(dispatches[1].metadata.flags,
            IREE_HAL_AMDGPU_DISPATCH_TIMESTAMP_RECORD_FLAG_INDIRECT_PARAMETERS);
  EXPECT_EQ(dispatches[1].target.completion_signal, &completion_signals[1]);
  EXPECT_EQ(dispatches[1].target.record, &records[1]);
}

TEST(AqlBlockProcessorTimestampTest, RejectsPartialCommandBufferTimestampPlan) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  iree_hal_amdgpu_pm4_ib_slot_t pm4_ib_slots[8] = {};
  uint16_t packet_headers[1] = {};
  uint16_t packet_setups[1] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[1] = {};

  iree_hal_amdgpu_aql_block_processor_timestamp_t processor = {};
  processor.base =
      MakeBaseProcessor(&ring, packet_headers, packet_setups, kernarg_blocks);
  processor.command_buffer.packets.start.packet = &packets[2];
  processor.command_buffer.packets.start.pm4_ib_slot = &pm4_ib_slots[2];

  iree_hal_amdgpu_aql_block_processor_timestamp_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
                            &processor, &block.header, &result));
  EXPECT_EQ(packets[4].dispatch.kernel_object, 0u);
}

TEST(AqlBlockProcessorTimestampTest, RejectsOutOfRangeDispatchPacketOrdinal) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[1] = {};
  uint16_t packet_setups[1] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[2] = {};
  iree_amd_signal_t completion_signal = {};
  iree_hal_amdgpu_dispatch_timestamp_record_t record = {};
  iree_hal_amdgpu_device_kernel_args_t harvest_kernel_args =
      MakeHarvestKernelArgs();

  iree_hal_amdgpu_aql_block_processor_timestamp_dispatch_t dispatch = {};
  dispatch.ordinals.packet_ordinal = 1;
  dispatch.target.completion_signal = &completion_signal;
  dispatch.target.record = &record;

  iree_hal_amdgpu_aql_block_processor_timestamp_t processor = {};
  processor.base =
      MakeBaseProcessor(&ring, packet_headers, packet_setups, kernarg_blocks);
  processor.dispatches.values = &dispatch;
  processor.dispatches.count = 1;
  processor.harvest.kernel_args = &harvest_kernel_args;
  processor.harvest.packet = &packets[5];
  processor.harvest.kernarg_ptr = kernarg_blocks[1].data;

  iree_hal_amdgpu_aql_block_processor_timestamp_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_aql_block_processor_timestamp_invoke(
                            &processor, &block.header, &result));
}

}  // namespace
}  // namespace iree::hal::amdgpu
