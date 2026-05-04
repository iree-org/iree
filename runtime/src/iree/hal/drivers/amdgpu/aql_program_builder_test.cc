// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_program_builder.h"

#include <cstddef>
#include <cstdint>

#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/aql_program_validation.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

class AqlProgramBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_block_pool_initialize(
        block_size_, iree_allocator_system(), &block_pool_));
  }

  void TearDown() override { iree_arena_block_pool_deinitialize(&block_pool_); }

  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

 private:
  iree_host_size_t block_size_ = 256;
  iree_arena_block_pool_t block_pool_;
};

static const iree_hal_amdgpu_command_buffer_command_header_t* FirstCommand(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  return iree_hal_amdgpu_command_buffer_block_commands_const(block);
}

static const iree_hal_amdgpu_command_buffer_command_header_t* LastCommand(
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      FirstCommand(block);
  for (uint16_t i = 1; i < block->command_count; ++i) {
    command = iree_hal_amdgpu_command_buffer_command_next_const(command);
  }
  return command;
}

static iree_status_t AppendBarriers(
    iree_hal_amdgpu_aql_program_builder_t* builder, int count) {
  for (int i = 0; i < count; ++i) {
    iree_hal_amdgpu_command_buffer_command_header_t* barrier = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_program_builder_append_command(
        builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER,
        IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
        sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t),
        /*binding_source_count=*/0, /*aql_packet_count=*/0,
        /*kernarg_length=*/0, &barrier, /*out_binding_sources=*/nullptr));
  }
  return iree_ok_status();
}

TEST(CommandBufferAbiTest, CoreRecordSizes) {
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_block_header_t), 64u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_command_header_t), 8u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_binding_source_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t), 80u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_fill_command_t), 40u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_copy_command_t), 48u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_update_command_t), 40u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_branch_command_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_cond_branch_command_t), 24u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_return_command_t), 8u);
}

TEST(CommandBufferAbiTest, BlockPoolRejectsNonPowerOfTwoBlockSize) {
  iree_arena_block_pool_t block_pool;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_aql_program_block_pool_initialize(
          /*block_size=*/384, iree_allocator_system(), &block_pool));
}

TEST(CommandBufferAbiTest, BuilderRejectsOversizedUsableBlockSize) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(
      (iree_host_size_t)UINT32_MAX + 1 + sizeof(iree_arena_block_t),
      iree_allocator_system(), &block_pool);

  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(&block_pool, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_aql_program_builder_begin(&builder));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST_F(AqlProgramBuilderTest, EmptyProgramRecordsReturnBlock) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  ASSERT_NE(program.first_block, nullptr);
  EXPECT_EQ(program.block_count, 1u);
  EXPECT_EQ(program.command_count, 1u);
  EXPECT_EQ(program.max_block_aql_packet_count, 0u);
  EXPECT_EQ(program.max_block_kernarg_length, 0u);

  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program.first_block;
  EXPECT_EQ(block->magic, IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_MAGIC);
  EXPECT_EQ(block->version, IREE_HAL_AMDGPU_COMMAND_BUFFER_BLOCK_VERSION_0);
  EXPECT_EQ(block->header_length, 64u);
  EXPECT_EQ(block->block_ordinal, 0u);
  EXPECT_EQ(block->block_length, block_pool()->usable_block_size);
  EXPECT_EQ(block->command_offset, 64u);
  EXPECT_EQ(block->command_count, 1u);
  EXPECT_EQ(block->binding_source_count, 0u);
  EXPECT_EQ(block->aql_packet_count, 0u);
  EXPECT_EQ(block->kernarg_length, 0u);
  EXPECT_EQ(block->terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(block->terminator_target_block_ordinal, 0u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      FirstCommand(block);
  EXPECT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_length(command),
            sizeof(iree_hal_amdgpu_command_buffer_return_command_t));

  IREE_EXPECT_OK(iree_hal_amdgpu_aql_program_validate_metadata_only(&program));

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, AppendsCommandAndBindingSources) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* command = nullptr;
  iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      /*binding_source_count=*/2, /*aql_packet_count=*/1,
      /*kernarg_length=*/128, &command, &binding_sources));

  ASSERT_NE(command, nullptr);
  ASSERT_NE(binding_sources, nullptr);
  EXPECT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  EXPECT_TRUE(iree_all_bits_set(
      command->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER |
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_USES_QUEUE_KERNARGS));
  EXPECT_EQ(command->command_index, 0u);

  binding_sources[0].flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC;
  binding_sources[0].slot = 3;
  binding_sources[0].offset_or_pointer = 64;
  binding_sources[1].flags =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE;
  binding_sources[1].slot = 0;
  binding_sources[1].offset_or_pointer = 0x12345678u;

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program.first_block;
  EXPECT_EQ(block->command_count, 2u);
  EXPECT_EQ(block->binding_source_count, 2u);
  EXPECT_EQ(block->dispatch_count, 1u);
  EXPECT_EQ(block->indirect_dispatch_count, 0u);
  EXPECT_EQ(block->profile_marker_count, 0u);
  EXPECT_EQ(block->aql_packet_count, 1u);
  EXPECT_EQ(block->kernarg_length, 128u);
  EXPECT_EQ(block->terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(block->terminator_target_block_ordinal, 0u);
  EXPECT_EQ(program.max_block_aql_packet_count, 1u);
  EXPECT_EQ(program.max_block_kernarg_length, 128u);

  const iree_hal_amdgpu_command_buffer_binding_source_t* block_binding_sources =
      iree_hal_amdgpu_command_buffer_block_binding_sources_const(block);
  EXPECT_EQ(block_binding_sources[0].flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC);
  EXPECT_EQ(block_binding_sources[0].slot, 3u);
  EXPECT_EQ(block_binding_sources[0].offset_or_pointer, 64u);
  EXPECT_EQ(block_binding_sources[1].flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_NONE);
  EXPECT_EQ(block_binding_sources[1].slot, 0u);
  EXPECT_EQ(block_binding_sources[1].offset_or_pointer, 0x12345678u);

  const iree_hal_amdgpu_command_buffer_command_header_t* return_command =
      LastCommand(block);
  EXPECT_EQ(return_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, PatchesBarrierScopesAtRecordingTime) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* first_dispatch = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/0, &first_dispatch,
      /*out_binding_sources=*/nullptr));

  iree_hal_amdgpu_command_buffer_command_header_t* barrier = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/0,
      /*kernarg_length=*/0, &barrier, /*out_binding_sources=*/nullptr));
  iree_hal_amdgpu_aql_program_builder_set_pending_barrier_scopes(
      &builder, IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_AGENT);

  iree_hal_amdgpu_command_buffer_command_header_t* second_dispatch = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/0, &second_dispatch,
      /*out_binding_sources=*/nullptr));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  EXPECT_FALSE(iree_any_bit_set(
      first_dispatch->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER));
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
                first_dispatch->flags),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_flags_release_scope(
                first_dispatch->flags),
            IREE_HSA_FENCE_SCOPE_AGENT);

  EXPECT_TRUE(iree_any_bit_set(
      second_dispatch->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER));
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
                second_dispatch->flags),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_flags_release_scope(
                second_dispatch->flags),
            IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, ForcedBarrierKeepsPendingAcquireScope) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* barrier = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_barrier_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/0,
      /*kernarg_length=*/0, &barrier, /*out_binding_sources=*/nullptr));
  iree_hal_amdgpu_aql_program_builder_set_pending_barrier_scopes(
      &builder, IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_amdgpu_command_buffer_command_header_t* dispatch = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/0, &dispatch, /*out_binding_sources=*/nullptr));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  EXPECT_TRUE(iree_any_bit_set(
      dispatch->flags,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER));
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_flags_acquire_scope(
                dispatch->flags),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_flags_release_scope(
                dispatch->flags),
            IREE_HSA_FENCE_SCOPE_AGENT);

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, SplitsBlocksWithBranchTerminator) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  for (int i = 0; i < 4; ++i) {
    iree_hal_amdgpu_command_buffer_command_header_t* command = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
        &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
        IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
        sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
        /*binding_source_count=*/0, /*aql_packet_count=*/1,
        /*kernarg_length=*/32, &command, /*out_binding_sources=*/nullptr));
    EXPECT_NE(command, nullptr);
  }

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  ASSERT_GE(program.block_count, 2u);
  const iree_hal_amdgpu_command_buffer_block_header_t* first_block =
      program.first_block;
  const iree_hal_amdgpu_command_buffer_command_header_t* first_terminator =
      LastCommand(first_block);
  ASSERT_EQ(first_terminator->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH);
  EXPECT_EQ(first_block->terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH);
  EXPECT_EQ(first_block->terminator_target_block_ordinal, 1u);
  const auto* branch =
      reinterpret_cast<const iree_hal_amdgpu_command_buffer_branch_command_t*>(
          first_terminator);
  EXPECT_EQ(branch->target_block_ordinal, 1u);

  const iree_hal_amdgpu_command_buffer_block_header_t* second_block =
      iree_hal_amdgpu_aql_program_block_next(block_pool(), first_block);
  ASSERT_NE(second_block, nullptr);
  EXPECT_EQ(second_block->block_ordinal, 1u);
  EXPECT_EQ(second_block->terminator_opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(second_block->terminator_target_block_ordinal, 0u);

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, ValidatesSplitMetadataOnlyProgram) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  IREE_ASSERT_OK(AppendBarriers(&builder, 12));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  ASSERT_GE(program.block_count, 2u);
  ASSERT_EQ(program.max_block_aql_packet_count, 0u);
  IREE_EXPECT_OK(iree_hal_amdgpu_aql_program_validate_metadata_only(&program));

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, MetadataOnlyValidationRejectsPayloadBlocks) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* dispatch = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/1,
      /*kernarg_length=*/0, &dispatch, /*out_binding_sources=*/nullptr));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_aql_program_validate_metadata_only(&program));

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, MetadataOnlyValidationRejectsProfileMarkers) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* profile_marker = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
      sizeof(iree_hal_amdgpu_command_buffer_command_header_t),
      /*binding_source_count=*/0, /*aql_packet_count=*/0,
      /*kernarg_length=*/0, &profile_marker,
      /*out_binding_sources=*/nullptr));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amdgpu_aql_program_validate_metadata_only(&program));

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest,
       MetadataOnlyValidationRejectsTerminatorMetadataMismatch) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  IREE_ASSERT_OK(AppendBarriers(&builder, 12));

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  ASSERT_GE(program.block_count, 2u);
  program.first_block->terminator_target_block_ordinal = 2;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_aql_program_validate_metadata_only(&program));

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, RejectsOversizedCommand) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* command = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amdgpu_aql_program_builder_append_command(
          &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
          IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_NONE,
          block_pool()->usable_block_size, /*binding_source_count=*/0,
          /*aql_packet_count=*/1, /*kernarg_length=*/0, &command,
          /*out_binding_sources=*/nullptr));

  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace iree::hal::amdgpu
