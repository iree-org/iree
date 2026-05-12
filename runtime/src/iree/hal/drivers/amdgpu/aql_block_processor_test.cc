// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_block_processor.h"

#include <array>
#include <cstring>
#include <memory>

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct ReturnBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Single return terminator command.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
};

struct BranchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Single branch terminator command.
  iree_hal_amdgpu_command_buffer_branch_command_t branch_command;
};

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

struct IndirectDispatchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Custom indirect dispatch command under test.
  iree_hal_amdgpu_command_buffer_dispatch_command_t dispatch_command;
  // Return terminator following the dispatch command.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
  // Indirect parameter source referenced by the dispatch command.
  iree_hal_amdgpu_command_buffer_binding_source_t indirect_params_source;
};

template <uint32_t DispatchCount>
struct DispatchBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Direct dispatch commands recorded in this block.
  iree_hal_amdgpu_command_buffer_dispatch_command_t
      dispatch_commands[DispatchCount];
  // Return terminator following the dispatch commands.
  iree_hal_amdgpu_command_buffer_return_command_t return_command;
};

struct MalformedBlock {
  // Block header at the ABI-defined block base.
  iree_hal_amdgpu_command_buffer_block_header_t header;
  // Non-terminating barrier command used to exercise validation.
  iree_hal_amdgpu_command_buffer_barrier_command_t barrier_command;
};

struct PacketHeaderSummary {
  // Counts accumulated across emitted packet headers.
  struct {
    // Number of emitted packet headers summarized.
    uint32_t total;
    // Number of emitted packet headers carrying the AQL barrier bit.
    uint32_t barrier;
    // Number of emitted packet headers with SYSTEM acquire scope.
    uint32_t system_acquire;
    // Number of emitted packet headers with SYSTEM release scope.
    uint32_t system_release;
  } counts;
  // Boundary packet headers from the emitted span.
  struct {
    // First emitted packet header, or zero for empty spans.
    uint16_t first;
    // Last emitted packet header, or zero for empty spans.
    uint16_t last;
  } headers;
};

struct CommandBufferDeleter {
  void operator()(iree_hal_command_buffer_t* command_buffer) const {
    iree_hal_command_buffer_release(command_buffer);
  }
};

using CommandBufferPtr =
    std::unique_ptr<iree_hal_command_buffer_t, CommandBufferDeleter>;

struct BufferDeleter {
  void operator()(iree_hal_buffer_t* buffer) const {
    iree_hal_buffer_release(buffer);
  }
};

using BufferPtr = std::unique_ptr<iree_hal_buffer_t, BufferDeleter>;

constexpr uint64_t kFillBlockX16KernelObject = 0xF160u;
constexpr uint64_t kCopyBlockX16KernelObject = 0xC160u;
constexpr uint64_t kPatchIndirectParamsKernelObject = 0x1D1EC7u;

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

static void SetReturnTerminator(
    iree_hal_amdgpu_command_buffer_block_header_t* block_header) {
  block_header->terminator_opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN;
  block_header->terminator_target_block_ordinal = 0;
}

static void SetBranchTerminator(
    uint32_t target_block_ordinal,
    iree_hal_amdgpu_command_buffer_block_header_t* block_header) {
  block_header->terminator_opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH;
  block_header->terminator_target_block_ordinal = target_block_ordinal;
}

static uint8_t CommandFlags(uint8_t flags, iree_hsa_fence_scope_t acquire_scope,
                            iree_hsa_fence_scope_t release_scope) {
  return iree_hal_amdgpu_command_buffer_command_flags_set_fence_scopes(
      flags, (uint8_t)acquire_scope, (uint8_t)release_scope);
}

static void InitializeDirectDispatchCommand(
    uint32_t command_index, uint8_t command_flags,
    iree_hal_amdgpu_command_buffer_dispatch_command_t* out_command) {
  std::memset(out_command, 0, sizeof(*out_command));
  out_command->header.opcode = IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH;
  out_command->header.flags = command_flags;
  out_command->header.length_qwords =
      sizeof(*out_command) / IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  out_command->header.command_index = command_index;
  out_command->kernel_object = 0xABCDEF0000000000ull + command_index;
  out_command->payload_reference = sizeof(*out_command);
  out_command->kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
  out_command->implicit_args_offset_qwords = UINT16_MAX;
  out_command->setup = 3;
  out_command->workgroup_size[0] = 1;
  out_command->workgroup_size[1] = 1;
  out_command->workgroup_size[2] = 1;
  out_command->grid_size[0] = 1;
  out_command->grid_size[1] = 1;
  out_command->grid_size[2] = 1;
}

static iree_hal_amdgpu_device_kernel_args_t MakeKernelArgs(
    uint64_t kernel_object, uint16_t setup, uint16_t workgroup_size_x,
    uint32_t private_segment_size, uint32_t group_segment_size) {
  iree_hal_amdgpu_device_kernel_args_t kernel_args = {};
  kernel_args.kernel_object = kernel_object;
  kernel_args.kernarg_size = 32;
  kernel_args.kernarg_alignment = 8;
  kernel_args.setup = setup;
  kernel_args.workgroup_size[0] = workgroup_size_x;
  kernel_args.workgroup_size[1] = 1;
  kernel_args.workgroup_size[2] = 1;
  kernel_args.private_segment_size = private_segment_size;
  kernel_args.group_segment_size = group_segment_size;
  return kernel_args;
}

static iree_hal_amdgpu_device_kernels_t MakeTransferKernels() {
  iree_hal_amdgpu_device_kernels_t kernels = {};
  kernels.iree_hal_amdgpu_device_buffer_fill_block_x16 =
      MakeKernelArgs(kFillBlockX16KernelObject, 5, 32, 8, 12);
  kernels.iree_hal_amdgpu_device_buffer_copy_block_x16 =
      MakeKernelArgs(kCopyBlockX16KernelObject, 10, 32, 13, 17);
  kernels.iree_hal_amdgpu_device_dispatch_patch_indirect_params =
      MakeKernelArgs(kPatchIndirectParamsKernelObject, 12, 1, 3, 7);
  return kernels;
}

static iree_hal_amdgpu_device_buffer_transfer_context_t MakeTransferContext(
    const iree_hal_amdgpu_device_kernels_t* kernels) {
  iree_hal_amdgpu_device_buffer_transfer_context_t context = {};
  iree_hal_amdgpu_device_buffer_transfer_context_initialize(
      kernels, /*compute_unit_count=*/4, /*wavefront_size=*/64, &context);
  return context;
}

static void NoOpBufferRelease(void* user_data, iree_hal_buffer_t* buffer) {
  (void)user_data;
  (void)buffer;
}

static iree_hal_buffer_binding_t MakeBinding(iree_hal_buffer_t* buffer,
                                             iree_device_size_t offset,
                                             iree_device_size_t length) {
  iree_hal_buffer_binding_t binding = {};
  binding.buffer = buffer;
  binding.offset = offset;
  binding.length = length;
  return binding;
}

static ReturnBlock MakeReturnBlock() {
  ReturnBlock block;
  InitializeBlockHeader(sizeof(block), sizeof(block.return_command),
                        /*command_count=*/1, /*aql_packet_count=*/0,
                        /*kernarg_length=*/0, &block.header);
  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);
  return block;
}

static BranchBlock MakeBranchBlock(uint32_t target_block_ordinal) {
  BranchBlock block;
  InitializeBlockHeader(sizeof(block), sizeof(block.branch_command),
                        /*command_count=*/1, /*aql_packet_count=*/0,
                        /*kernarg_length=*/0, &block.header);
  std::memset(&block.branch_command, 0, sizeof(block.branch_command));
  block.branch_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH;
  block.branch_command.header.length_qwords =
      sizeof(block.branch_command) /
      IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  block.branch_command.target_block_ordinal = target_block_ordinal;
  SetBranchTerminator(target_block_ordinal, &block.header);
  return block;
}

static MalformedBlock MakeUnterminatedBlock() {
  MalformedBlock block;
  InitializeBlockHeader(sizeof(block), sizeof(block.barrier_command),
                        /*command_count=*/1, /*aql_packet_count=*/0,
                        /*kernarg_length=*/0, &block.header);
  std::memset(&block.barrier_command, 0, sizeof(block.barrier_command));
  block.barrier_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER;
  block.barrier_command.header.length_qwords =
      sizeof(block.barrier_command) /
      IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  return block;
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
  block.dispatch_command.private_segment_size = 128;
  block.dispatch_command.group_segment_size = 256;
  block.tail[0] = 0x0A0B0C0D0E0F1011ull;
  block.tail[1] = 0x1213141516171819ull;

  block.header.dispatch_count = 1;
  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);
  return block;
}

static IndirectDispatchBlock MakeIndirectDispatchBlock(
    const uint32_t* workgroup_count,
    uint8_t command_flags =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS) {
  IndirectDispatchBlock block;
  const uint32_t command_length =
      sizeof(block.dispatch_command) + sizeof(block.return_command);
  InitializeBlockHeader(sizeof(block), command_length, /*command_count=*/2,
                        /*aql_packet_count=*/2,
                        /*kernarg_length=*/
                        2 * sizeof(iree_hal_amdgpu_kernarg_block_t),
                        &block.header);
  block.header.dispatch_count = 1;
  block.header.indirect_dispatch_count = 1;
  block.header.binding_source_count = 1;
  block.header.binding_source_offset =
      offsetof(IndirectDispatchBlock, indirect_params_source);

  std::memset(&block.dispatch_command, 0, sizeof(block.dispatch_command));
  block.dispatch_command.header.opcode =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH;
  block.dispatch_command.header.flags = command_flags;
  block.dispatch_command.header.length_qwords =
      sizeof(block.dispatch_command) /
      IREE_HAL_AMDGPU_COMMAND_BUFFER_RECORD_ALIGNMENT;
  block.dispatch_command.kernel_object = 0x123456789ABCDEF0ull;
  block.dispatch_command.binding_source_offset =
      block.header.binding_source_offset;
  block.dispatch_command.dispatch_flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_INDIRECT_PARAMETERS;
  block.dispatch_command.kernarg_strategy =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
  block.dispatch_command.implicit_args_offset_qwords = UINT16_MAX;
  block.dispatch_command.setup = 3;
  block.dispatch_command.workgroup_size[0] = 4;
  block.dispatch_command.workgroup_size[1] = 2;
  block.dispatch_command.workgroup_size[2] = 1;
  block.dispatch_command.private_segment_size = 128;
  block.dispatch_command.group_segment_size = 256;

  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);

  std::memset(&block.indirect_params_source, 0,
              sizeof(block.indirect_params_source));
  block.indirect_params_source.flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS;
  block.indirect_params_source.offset_or_pointer =
      (uint64_t)(uintptr_t)workgroup_count;
  return block;
}

template <uint32_t DispatchCount>
static DispatchBlock<DispatchCount> MakeDispatchBlock(
    const uint8_t (&dispatch_command_flags)[DispatchCount]) {
  DispatchBlock<DispatchCount> block;
  const uint32_t command_length =
      DispatchCount * sizeof(block.dispatch_commands[0]) +
      sizeof(block.return_command);
  InitializeBlockHeader(sizeof(block), command_length,
                        /*command_count=*/DispatchCount + 1,
                        /*aql_packet_count=*/DispatchCount,
                        /*kernarg_length=*/DispatchCount *
                            sizeof(iree_hal_amdgpu_kernarg_block_t),
                        &block.header);
  block.header.dispatch_count = DispatchCount;
  for (uint32_t i = 0; i < DispatchCount; ++i) {
    InitializeDirectDispatchCommand(i, dispatch_command_flags[i],
                                    &block.dispatch_commands[i]);
  }
  InitializeReturnCommand(&block.return_command);
  SetReturnTerminator(&block.header);
  return block;
}

static iree_hal_amdgpu_aql_block_processor_t MakeProcessor(
    iree_hal_amdgpu_aql_ring_t* ring, uint32_t packet_count,
    uint16_t* packet_headers, uint16_t* packet_setups,
    iree_hal_amdgpu_kernarg_block_t* kernarg_blocks,
    uint32_t kernarg_block_count,
    iree_hal_amdgpu_aql_block_processor_flags_t flags =
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE,
    iree_hsa_fence_scope_t inline_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE,
    iree_hsa_fence_scope_t signal_release_scope = IREE_HSA_FENCE_SCOPE_SYSTEM,
    iree_hsa_fence_scope_t payload_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context =
        nullptr,
    iree_hal_command_buffer_t* command_buffer = nullptr,
    iree_hal_buffer_binding_table_t binding_table = {0, nullptr}) {
  iree_hal_amdgpu_aql_block_processor_t processor = {};
  processor.transfer_context = transfer_context;
  processor.command_buffer = command_buffer;
  processor.bindings.table = binding_table;
  processor.packets.ring = ring;
  processor.packets.first_id = 4;
  processor.packets.index_base = 0;
  processor.packets.count = packet_count;
  processor.packets.headers = packet_headers;
  processor.packets.setups = packet_setups;
  processor.kernargs.blocks = kernarg_blocks;
  processor.kernargs.count = kernarg_block_count;
  processor.submission.inline_acquire_scope = inline_acquire_scope;
  processor.submission.signal_release_scope = signal_release_scope;
  processor.payload.acquire_scope = payload_acquire_scope;
  processor.flags = flags;
  return processor;
}

static uint32_t KernargBlockCount(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  return (uint32_t)iree_host_size_ceil_div(
      block->kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
}

static uint16_t AqlHeaderField(uint16_t header, uint32_t bit_offset,
                               uint32_t bit_width) {
  return (header >> bit_offset) & ((1u << bit_width) - 1u);
}

static bool AqlHeaderHasBarrier(uint16_t header) {
  return AqlHeaderField(header, IREE_HSA_PACKET_HEADER_BARRIER,
                        IREE_HSA_PACKET_HEADER_WIDTH_BARRIER) != 0;
}

static iree_hsa_fence_scope_t AqlHeaderAcquireScope(uint16_t header) {
  return (iree_hsa_fence_scope_t)AqlHeaderField(
      header, IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
      IREE_HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE);
}

static iree_hsa_fence_scope_t AqlHeaderReleaseScope(uint16_t header) {
  return (iree_hsa_fence_scope_t)AqlHeaderField(
      header, IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
      IREE_HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE);
}

static PacketHeaderSummary SummarizePacketHeaders(
    const uint16_t* packet_headers, uint32_t packet_count) {
  PacketHeaderSummary summary = {};
  for (uint32_t i = 0; i < packet_count; ++i) {
    const uint16_t header = packet_headers[i];
    if (summary.counts.total == 0) summary.headers.first = header;
    summary.headers.last = header;
    ++summary.counts.total;
    if (AqlHeaderHasBarrier(header)) ++summary.counts.barrier;
    if (AqlHeaderAcquireScope(header) == IREE_HSA_FENCE_SCOPE_SYSTEM) {
      ++summary.counts.system_acquire;
    }
    if (AqlHeaderReleaseScope(header) == IREE_HSA_FENCE_SCOPE_SYSTEM) {
      ++summary.counts.system_release;
    }
  }
  return summary;
}

template <uint32_t DispatchCount>
static iree_status_t InvokeAndSummarizeDispatchBlock(
    const DispatchBlock<DispatchCount>& block,
    iree_hal_amdgpu_aql_block_processor_flags_t flags,
    iree_hsa_fence_scope_t inline_acquire_scope,
    iree_hsa_fence_scope_t signal_release_scope,
    iree_hsa_fence_scope_t payload_acquire_scope,
    PacketHeaderSummary* out_summary) {
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[DispatchCount] = {};
  uint16_t packet_setups[DispatchCount] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[DispatchCount] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/DispatchCount, packet_headers, packet_setups,
      kernarg_blocks, /*kernarg_block_count=*/DispatchCount, flags,
      inline_acquire_scope, signal_release_scope, payload_acquire_scope);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));
  *out_summary = SummarizePacketHeaders(packet_headers, DispatchCount);
  return iree_ok_status();
}

class AqlBlockProcessorRecordedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("aql_block_processor_test"),
        iree_allocator_system(), iree_allocator_system(), &device_allocator_));
    iree_hal_amdgpu_profile_metadata_initialize(iree_allocator_system(),
                                                &profile_metadata_);
    IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_block_pool_initialize(
        block_size_, iree_allocator_system(), &block_pool_));
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_hal_amdgpu_profile_metadata_deinitialize(&profile_metadata_);
    iree_hal_allocator_release(device_allocator_);
  }

  CommandBufferPtr CreateCommandBuffer(iree_host_size_t binding_capacity) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_amdgpu_aql_command_buffer_create(
        device_allocator_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, /*device_ordinal=*/0,
        iree_hal_amdgpu_aql_prepublished_kernarg_storage_disabled(),
        &profile_metadata_, &block_pool_, &block_pool_, iree_allocator_system(),
        &command_buffer));
    return CommandBufferPtr(command_buffer);
  }

  BufferPtr CreateBuffer(void* storage, iree_device_size_t length) {
    iree_hal_buffer_release_callback_t release_callback = {};
    release_callback.fn = NoOpBufferRelease;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_amdgpu_buffer_create(
        /*libhsa=*/nullptr, iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH, length,
        length, storage, release_callback, iree_allocator_system(), &buffer));
    return BufferPtr(buffer);
  }

 private:
  // Test allocator borrowed by command buffers for validation.
  iree_hal_allocator_t* device_allocator_ = nullptr;
  // Fixed block size used by recorded command-buffer tests.
  iree_host_size_t block_size_ = 4096;
  // Program and resource-set block pool borrowed by test command buffers.
  iree_arena_block_pool_t block_pool_;
  // Profile metadata registry borrowed by test command buffers.
  iree_hal_amdgpu_profile_metadata_registry_t profile_metadata_;
};

TEST(AqlBlockProcessorTest, ReturnTerminatorProducesNoPayload) {
  ReturnBlock block = MakeReturnBlock();
  iree_hal_amdgpu_aql_block_processor_t processor =
      MakeProcessor(/*ring=*/nullptr, /*packet_count=*/0,
                    /*packet_headers=*/nullptr, /*packet_setups=*/nullptr,
                    /*kernarg_blocks=*/nullptr, /*kernarg_block_count=*/0);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(block.header.terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(result.packets.recorded, 0u);
  EXPECT_EQ(result.packets.emitted, 0u);
  EXPECT_EQ(result.kernargs.consumed, 0u);
}

TEST(AqlBlockProcessorTest, BranchTerminatorReportsTargetBlock) {
  BranchBlock block = MakeBranchBlock(/*target_block_ordinal=*/7);
  iree_hal_amdgpu_aql_block_processor_t processor =
      MakeProcessor(/*ring=*/nullptr, /*packet_count=*/0,
                    /*packet_headers=*/nullptr, /*packet_setups=*/nullptr,
                    /*kernarg_blocks=*/nullptr, /*kernarg_block_count=*/0);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_BRANCH);
  EXPECT_EQ(result.target_block_ordinal, 7u);
  EXPECT_EQ(block.header.terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH);
  EXPECT_EQ(block.header.terminator_target_block_ordinal,
            result.target_block_ordinal);
  EXPECT_EQ(result.packets.recorded, 0u);
  EXPECT_EQ(result.packets.emitted, 0u);
  EXPECT_EQ(result.kernargs.consumed, 0u);
}

TEST(AqlBlockProcessorTest, UnterminatedBlockFails) {
  MalformedBlock block = MakeUnterminatedBlock();
  iree_hal_amdgpu_aql_block_processor_t processor =
      MakeProcessor(/*ring=*/nullptr, /*packet_count=*/0,
                    /*packet_headers=*/nullptr, /*packet_setups=*/nullptr,
                    /*kernarg_blocks=*/nullptr, /*kernarg_block_count=*/0);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_aql_block_processor_invoke(
                            &processor, &block.header, &result));
}

TEST(AqlBlockProcessorTest, DirectDispatchPopulatesPacketAndKernarg) {
  DirectDispatchBlock block = MakeDirectDispatchBlock();
  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = 7;
  uint16_t packet_headers[1] = {0xCDCD};
  uint16_t packet_setups[1] = {0xCDCD};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[1] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/1, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/1,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(result.packets.recorded, 1u);
  EXPECT_EQ(result.packets.emitted, 1u);
  EXPECT_EQ(result.kernargs.consumed, 1u);

  const iree_hal_amdgpu_aql_packet_t& packet = packets[4];
  EXPECT_EQ(packet.dispatch.setup, block.dispatch_command.setup);
  EXPECT_EQ(packet.dispatch.workgroup_size[0],
            block.dispatch_command.workgroup_size[0]);
  EXPECT_EQ(packet.dispatch.grid_size[0], block.dispatch_command.grid_size[0]);
  EXPECT_EQ(packet.dispatch.private_segment_size,
            block.dispatch_command.private_segment_size);
  EXPECT_EQ(packet.dispatch.group_segment_size,
            block.dispatch_command.group_segment_size);
  EXPECT_EQ(packet.dispatch.kernel_object,
            block.dispatch_command.kernel_object);
  EXPECT_EQ(packet.dispatch.kernarg_address, kernarg_blocks[0].data);
  EXPECT_EQ(packet.dispatch.completion_signal.handle, 0u);
  EXPECT_EQ(packet_setups[0], block.dispatch_command.setup);
  EXPECT_EQ(std::memcmp(kernarg_blocks[0].data, block.tail, sizeof(block.tail)),
            0);

  EXPECT_EQ(packet_headers[0],
            iree_hsa_make_packet_header(IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                                        /*is_barrier=*/true,
                                        IREE_HSA_FENCE_SCOPE_SYSTEM,
                                        IREE_HSA_FENCE_SCOPE_SYSTEM));
}

TEST(AqlBlockProcessorTest,
     BuilderProducedSplitBlocksInvokeAsBranchThenReturn) {
  iree_arena_block_pool_t block_pool;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_block_pool_initialize(
      /*block_size=*/256, iree_allocator_system(), &block_pool));

  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(&block_pool, &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  for (uint32_t i = 0; i < 4; ++i) {
    iree_hal_amdgpu_command_buffer_command_header_t* command = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
        &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
        IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
        sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
        /*binding_source_count=*/0, /*aql_packet_count=*/1,
        sizeof(iree_hal_amdgpu_kernarg_block_t), &command,
        /*out_binding_sources=*/nullptr));
    auto* dispatch_command =
        reinterpret_cast<iree_hal_amdgpu_command_buffer_dispatch_command_t*>(
            command);
    dispatch_command->kernel_object = 0xABCDEF0000000000ull + i;
    dispatch_command->payload_reference = sizeof(*dispatch_command);
    dispatch_command->kernarg_strategy =
        IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STRATEGY_CUSTOM_DIRECT;
    dispatch_command->implicit_args_offset_qwords = UINT16_MAX;
    dispatch_command->setup = 3;
    dispatch_command->workgroup_size[0] = 1;
    dispatch_command->workgroup_size[1] = 1;
    dispatch_command->workgroup_size[2] = 1;
    dispatch_command->grid_size[0] = 1;
    dispatch_command->grid_size[1] = 1;
    dispatch_command->grid_size[2] = 1;
  }

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  ASSERT_GE(program.block_count, 2u);
  const iree_hal_amdgpu_command_buffer_block_header_t* first_block =
      program.first_block;
  const iree_hal_amdgpu_command_buffer_block_header_t* second_block =
      iree_hal_amdgpu_aql_program_block_next(&block_pool, first_block);
  ASSERT_NE(second_block, nullptr);

  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;

  uint16_t first_packet_headers[4] = {};
  uint16_t first_packet_setups[4] = {};
  iree_hal_amdgpu_kernarg_block_t first_kernarg_blocks[4] = {};
  iree_hal_amdgpu_aql_block_processor_t first_processor = MakeProcessor(
      &ring, first_block->aql_packet_count, first_packet_headers,
      first_packet_setups, first_kernarg_blocks, KernargBlockCount(first_block),
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_amdgpu_aql_block_processor_result_t first_result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &first_processor, first_block, &first_result));
  EXPECT_EQ(first_result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_BRANCH);
  EXPECT_EQ(first_result.target_block_ordinal, 1u);
  EXPECT_EQ(first_result.packets.emitted, first_block->aql_packet_count);
  EXPECT_EQ(first_result.kernargs.consumed, KernargBlockCount(first_block));
  EXPECT_EQ(first_block->terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH);

  uint16_t second_packet_headers[4] = {};
  uint16_t second_packet_setups[4] = {};
  iree_hal_amdgpu_kernarg_block_t second_kernarg_blocks[4] = {};
  iree_hal_amdgpu_aql_block_processor_t second_processor = MakeProcessor(
      &ring, second_block->aql_packet_count, second_packet_headers,
      second_packet_setups, second_kernarg_blocks,
      KernargBlockCount(second_block),
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_amdgpu_aql_block_processor_result_t second_result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &second_processor, second_block, &second_result));
  EXPECT_EQ(second_result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(second_result.packets.emitted, second_block->aql_packet_count);
  EXPECT_EQ(second_result.kernargs.consumed, KernargBlockCount(second_block));
  EXPECT_EQ(second_block->terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);

  iree_hal_amdgpu_aql_program_release(&program);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AqlBlockProcessorTest, IndirectDispatchEmitsPatchThenUnpublishedDispatch) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeTransferKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t transfer_context =
      MakeTransferContext(&kernels);
  const uint32_t workgroup_count[3] = {7, 5, 3};
  IndirectDispatchBlock block = MakeIndirectDispatchBlock(workgroup_count);

  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[2] = {};
  uint16_t packet_setups[2] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[2] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/2, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/2,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_HSA_FENCE_SCOPE_NONE, &transfer_context);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(result.packets.recorded, 2u);
  EXPECT_EQ(result.packets.emitted, 2u);
  EXPECT_EQ(result.kernargs.consumed, 2u);

  const iree_hal_amdgpu_aql_packet_t& patch_packet = packets[4];
  const iree_hal_amdgpu_aql_packet_t& dispatch_packet = packets[5];
  EXPECT_EQ(patch_packet.dispatch.kernel_object,
            kPatchIndirectParamsKernelObject);
  EXPECT_EQ(dispatch_packet.dispatch.kernel_object,
            block.dispatch_command.kernel_object);
  EXPECT_EQ(dispatch_packet.dispatch.kernarg_address, kernarg_blocks[1].data);
  EXPECT_EQ(packet_headers[1], IREE_HSA_PACKET_TYPE_INVALID);

  const auto* patch_args = reinterpret_cast<
      const iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t*>(
      kernarg_blocks[0].data);
  EXPECT_EQ(patch_args->workgroup_count, workgroup_count);
  EXPECT_EQ(patch_args->dispatch_packet, &packets[5].dispatch);
  EXPECT_EQ(patch_args->implicit_args, nullptr);
  EXPECT_EQ(patch_args->dispatch_header_setup,
            (uint32_t)iree_hal_amdgpu_aql_make_header(
                IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                iree_hal_amdgpu_aql_packet_control(
                    /*has_barrier=*/true, IREE_HSA_FENCE_SCOPE_NONE,
                    IREE_HSA_FENCE_SCOPE_SYSTEM)) |
                ((uint32_t)packet_setups[1] << 16));
  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[0]));
  EXPECT_EQ(AqlHeaderAcquireScope(packet_headers[0]),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(AqlHeaderReleaseScope(packet_headers[0]),
            IREE_HSA_FENCE_SCOPE_NONE);
}

TEST(AqlBlockProcessorTest, IndirectDispatchSplitsCommandBarrierScopes) {
  const iree_hal_amdgpu_device_kernels_t kernels = MakeTransferKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t transfer_context =
      MakeTransferContext(&kernels);
  const uint32_t workgroup_count[3] = {7, 5, 3};
  IndirectDispatchBlock block = MakeIndirectDispatchBlock(
      workgroup_count,
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS |
              IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
          IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_AGENT));

  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[2] = {};
  uint16_t packet_setups[2] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[2] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/2, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/2, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, &transfer_context);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));

  ASSERT_EQ(result.packets.recorded, 2u);
  ASSERT_EQ(result.packets.emitted, 2u);

  const auto* patch_args = reinterpret_cast<
      const iree_hal_amdgpu_device_dispatch_patch_indirect_params_args_t*>(
      kernarg_blocks[0].data);
  EXPECT_EQ(patch_args->dispatch_header_setup,
            (uint32_t)iree_hal_amdgpu_aql_make_header(
                IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
                iree_hal_amdgpu_aql_packet_control(
                    /*has_barrier=*/false, IREE_HSA_FENCE_SCOPE_NONE,
                    IREE_HSA_FENCE_SCOPE_AGENT)) |
                ((uint32_t)packet_setups[1] << 16));

  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[0]));
  EXPECT_EQ(AqlHeaderAcquireScope(packet_headers[0]),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(packet_headers[0]),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(packet_headers[1], IREE_HSA_PACKET_TYPE_INVALID);
}

TEST_F(AqlBlockProcessorRecordedTest, RecordedTransfersEmitBlitPackets) {
  alignas(16) uint8_t fill_target_storage[1024] = {};
  alignas(16) uint8_t copy_source_storage[1024] = {};
  alignas(16) uint8_t copy_target_storage[1024] = {};
  BufferPtr fill_target_buffer =
      CreateBuffer(fill_target_storage, sizeof(fill_target_storage));
  BufferPtr copy_source_buffer =
      CreateBuffer(copy_source_storage, sizeof(copy_source_storage));
  BufferPtr copy_target_buffer =
      CreateBuffer(copy_target_storage, sizeof(copy_target_storage));
  ASSERT_NE(fill_target_buffer, nullptr);
  ASSERT_NE(copy_source_buffer, nullptr);
  ASSERT_NE(copy_target_buffer, nullptr);

  CommandBufferPtr command_buffer = CreateCommandBuffer(/*binding_capacity=*/3);
  ASSERT_NE(command_buffer, nullptr);
  const uint32_t fill_pattern = 0xAABBCCDDu;
  const uint8_t update_source[16] = {
      0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
      0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
  };

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/32,
                                        /*length=*/512),
      &fill_pattern, sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer.get(),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/0,
                                        /*length=*/512),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/2, /*offset=*/64,
                                        /*length=*/512),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer.get(), update_source, /*source_offset=*/0,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/2, /*offset=*/128,
                                        sizeof(update_source)),
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer.get());
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->block_count, 1u);
  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program->first_block;
  ASSERT_EQ(block->aql_packet_count, 3u);
  ASSERT_EQ(KernargBlockCount(block), 3u);

  const iree_hal_amdgpu_device_kernels_t kernels = MakeTransferKernels();
  const iree_hal_amdgpu_device_buffer_transfer_context_t transfer_context =
      MakeTransferContext(&kernels);

  const std::array<iree_hal_buffer_binding_t, 3> bindings = {{
      MakeBinding(fill_target_buffer.get(), /*offset=*/0,
                  sizeof(fill_target_storage)),
      MakeBinding(copy_source_buffer.get(), /*offset=*/0,
                  sizeof(copy_source_storage)),
      MakeBinding(copy_target_buffer.get(), /*offset=*/0,
                  sizeof(copy_target_storage)),
  }};
  const iree_hal_buffer_binding_table_t binding_table = {bindings.size(),
                                                         bindings.data()};

  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[3] = {};
  uint16_t packet_setups[3] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[3] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/3, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/3,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_HSA_FENCE_SCOPE_NONE, &transfer_context, command_buffer.get(),
      binding_table);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_aql_block_processor_invoke(&processor, block, &result));

  EXPECT_EQ(result.terminator,
            IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_TERMINATOR_RETURN);
  EXPECT_EQ(result.packets.recorded, 3u);
  EXPECT_EQ(result.packets.emitted, 3u);
  EXPECT_EQ(result.kernargs.consumed, 3u);

  const iree_hal_amdgpu_aql_packet_t& fill_packet = packets[4];
  const iree_hal_amdgpu_aql_packet_t& copy_packet = packets[5];
  const iree_hal_amdgpu_aql_packet_t& update_packet = packets[6];
  EXPECT_EQ(fill_packet.dispatch.kernel_object, kFillBlockX16KernelObject);
  EXPECT_EQ(copy_packet.dispatch.kernel_object, kCopyBlockX16KernelObject);
  EXPECT_EQ(update_packet.dispatch.kernel_object, kCopyBlockX16KernelObject);
  EXPECT_EQ(fill_packet.dispatch.kernarg_address, kernarg_blocks[0].data);
  EXPECT_EQ(copy_packet.dispatch.kernarg_address, kernarg_blocks[1].data);
  EXPECT_EQ(update_packet.dispatch.kernarg_address, kernarg_blocks[2].data);

  const auto* fill_args =
      reinterpret_cast<const iree_hal_amdgpu_device_buffer_fill_kernargs_t*>(
          kernarg_blocks[0].data);
  EXPECT_EQ(fill_args->target_ptr,
            static_cast<void*>(fill_target_storage + 32));
  EXPECT_EQ(fill_args->element_length, 32u);
  EXPECT_EQ(fill_args->pattern, 0xAABBCCDDAABBCCDDull);

  const auto* copy_args =
      reinterpret_cast<const iree_hal_amdgpu_device_buffer_copy_kernargs_t*>(
          kernarg_blocks[1].data);
  EXPECT_EQ(copy_args->source_ptr,
            static_cast<const void*>(copy_source_storage));
  EXPECT_EQ(copy_args->target_ptr,
            static_cast<void*>(copy_target_storage + 64));
  EXPECT_EQ(copy_args->element_length, 32u);

  const auto* update_args =
      reinterpret_cast<const iree_hal_amdgpu_device_buffer_copy_kernargs_t*>(
          kernarg_blocks[2].data);
  EXPECT_EQ(update_args->target_ptr,
            static_cast<void*>(copy_target_storage + 128));
  EXPECT_EQ(update_args->element_length, 1u);
  ASSERT_NE(update_args->source_ptr, nullptr);
  EXPECT_EQ(std::memcmp(update_args->source_ptr, update_source,
                        sizeof(update_source)),
            0);

  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[2]));
  EXPECT_EQ(AqlHeaderReleaseScope(packet_headers[2]),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
}

TEST(AqlBlockProcessorTest,
     PacketHeadersOmitInteriorBarriersWithoutExecutionBarrier) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 1u);
  EXPECT_FALSE(AqlHeaderHasBarrier(summary.headers.first));
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.last));
}

TEST(AqlBlockProcessorTest, PacketHeadersBarrierFirstPayloadForInlineWait) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_SYSTEM, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 2u);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.first));
  EXPECT_EQ(AqlHeaderAcquireScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.last));
}

TEST(AqlBlockProcessorTest, PacketHeadersPreserveExplicitMemoryBarrierScopes) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_SYSTEM),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS |
              IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
          IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_AGENT),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 1u);
  EXPECT_FALSE(AqlHeaderHasBarrier(summary.headers.first));
  EXPECT_EQ(AqlHeaderAcquireScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.first),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_TRUE(AqlHeaderHasBarrier(summary.headers.last));
  EXPECT_EQ(AqlHeaderAcquireScope(summary.headers.last),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(AqlHeaderReleaseScope(summary.headers.last),
            IREE_HSA_FENCE_SCOPE_AGENT);
}

TEST(AqlBlockProcessorTest, PacketHeadersHonorExplicitExecutionBarrier) {
  const uint8_t dispatch_command_flags[3] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS |
              IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
          IREE_HSA_FENCE_SCOPE_AGENT, IREE_HSA_FENCE_SCOPE_AGENT),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<3> block = MakeDispatchBlock(dispatch_command_flags);

  alignas(64) iree_hal_amdgpu_aql_packet_t packets[8] = {};
  iree_hal_amdgpu_aql_ring_t ring = {};
  ring.base = packets;
  ring.mask = IREE_ARRAYSIZE(packets) - 1u;
  uint16_t packet_headers[3] = {};
  uint16_t packet_setups[3] = {};
  iree_hal_amdgpu_kernarg_block_t kernarg_blocks[3] = {};
  iree_hal_amdgpu_aql_block_processor_t processor = MakeProcessor(
      &ring, /*packet_count=*/3, packet_headers, packet_setups, kernarg_blocks,
      /*kernarg_block_count=*/3,
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_amdgpu_aql_block_processor_result_t result;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_block_processor_invoke(
      &processor, &block.header, &result));
  PacketHeaderSummary summary = SummarizePacketHeaders(packet_headers, 3);

  EXPECT_EQ(summary.counts.total, 3u);
  EXPECT_EQ(summary.counts.barrier, 2u);
  EXPECT_FALSE(AqlHeaderHasBarrier(packet_headers[0]));
  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[1]));
  EXPECT_TRUE(AqlHeaderHasBarrier(packet_headers[2]));
}

TEST(AqlBlockProcessorTest,
     PacketHeadersApplySystemAcquireOnlyToFirstDynamicKernargPacket) {
  const uint8_t dispatch_command_flags[2] = {
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
      CommandFlags(
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS,
          IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
  };
  DispatchBlock<2> block = MakeDispatchBlock(dispatch_command_flags);

  PacketHeaderSummary summary = {};
  IREE_ASSERT_OK(InvokeAndSummarizeDispatchBlock(
      block, IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET,
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_SYSTEM, &summary));

  EXPECT_EQ(summary.counts.total, 2u);
  EXPECT_EQ(summary.counts.barrier, 2u);
  EXPECT_EQ(summary.counts.system_acquire, 1u);
  EXPECT_EQ(summary.counts.system_release, 0u);
}

}  // namespace
}  // namespace iree::hal::amdgpu
