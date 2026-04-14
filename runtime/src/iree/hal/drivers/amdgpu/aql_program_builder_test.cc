// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_program_builder.h"

#include <cstddef>
#include <cstdint>

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

TEST(CommandBufferAbiTest, CoreRecordSizes) {
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_block_header_t), 64u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_command_header_t), 16u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_fixup_t), 24u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t), 48u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_fill_command_t), 48u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_copy_command_t), 56u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_branch_command_t), 24u);
  EXPECT_EQ(sizeof(iree_hal_amdgpu_command_buffer_return_command_t), 16u);
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
  EXPECT_EQ(block->fixup_count, 0u);
  EXPECT_EQ(block->aql_packet_count, 0u);
  EXPECT_EQ(block->kernarg_length, 0u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      FirstCommand(block);
  EXPECT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
  EXPECT_EQ(iree_hal_amdgpu_command_buffer_command_length(command),
            sizeof(iree_hal_amdgpu_command_buffer_return_command_t));

  iree_hal_amdgpu_aql_program_release(&program);
}

TEST_F(AqlProgramBuilderTest, AppendsCommandAndFixups) {
  iree_hal_amdgpu_aql_program_builder_t builder;
  iree_hal_amdgpu_aql_program_builder_initialize(block_pool(), &builder);
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_begin(&builder));

  iree_hal_amdgpu_command_buffer_command_header_t* command = nullptr;
  iree_hal_amdgpu_command_buffer_fixup_t* fixups = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_append_command(
      &builder, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH,
      IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER,
      sizeof(iree_hal_amdgpu_command_buffer_dispatch_command_t),
      /*fixup_count=*/2, /*aql_packet_count=*/1, /*kernarg_length=*/128,
      &command, &fixups));

  ASSERT_NE(command, nullptr);
  ASSERT_NE(fixups, nullptr);
  EXPECT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  EXPECT_EQ(command->flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER);
  EXPECT_EQ(command->command_index, 0u);
  EXPECT_EQ(command->fixup_count, 2u);

  fixups[0].kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_DYNAMIC_BINDING;
  fixups[0].ordinal = 3;
  fixups[0].source_offset = 64;
  fixups[0].patch_offset = 8;
  fixups[1].kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_STATIC_BINDING;
  fixups[1].ordinal = 4;
  fixups[1].source_offset = 128;
  fixups[1].patch_offset = 16;

  iree_hal_amdgpu_aql_program_t program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_builder_end(&builder, &program));
  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);

  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program.first_block;
  EXPECT_EQ(block->command_count, 2u);
  EXPECT_EQ(block->fixup_count, 2u);
  EXPECT_EQ(block->aql_packet_count, 1u);
  EXPECT_EQ(block->kernarg_length, 128u);
  EXPECT_EQ(program.max_block_aql_packet_count, 1u);
  EXPECT_EQ(program.max_block_kernarg_length, 128u);

  const iree_hal_amdgpu_command_buffer_fixup_t* block_fixups =
      reinterpret_cast<const iree_hal_amdgpu_command_buffer_fixup_t*>(
          reinterpret_cast<const uint8_t*>(block) + command->fixup_offset);
  EXPECT_EQ(block_fixups[0].kind,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_DYNAMIC_BINDING);
  EXPECT_EQ(block_fixups[0].ordinal, 3u);
  EXPECT_EQ(block_fixups[0].source_offset, 64u);
  EXPECT_EQ(block_fixups[0].patch_offset, 8u);
  EXPECT_EQ(block_fixups[1].kind,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_FIXUP_KIND_STATIC_BINDING);
  EXPECT_EQ(block_fixups[1].ordinal, 4u);
  EXPECT_EQ(block_fixups[1].source_offset, 128u);
  EXPECT_EQ(block_fixups[1].patch_offset, 16u);

  const iree_hal_amdgpu_command_buffer_command_header_t* return_command =
      LastCommand(block);
  EXPECT_EQ(return_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);

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
        /*fixup_count=*/0, /*aql_packet_count=*/1, /*kernarg_length=*/32,
        &command, /*out_fixups=*/nullptr));
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
  const auto* branch =
      reinterpret_cast<const iree_hal_amdgpu_command_buffer_branch_command_t*>(
          first_terminator);
  EXPECT_EQ(branch->target_block_ordinal, 1u);

  const iree_hal_amdgpu_command_buffer_block_header_t* second_block =
      iree_hal_amdgpu_aql_program_block_next(block_pool(), first_block);
  ASSERT_NE(second_block, nullptr);
  EXPECT_EQ(second_block->block_ordinal, 1u);

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
          block_pool()->usable_block_size, /*fixup_count=*/0,
          /*aql_packet_count=*/1, /*kernarg_length=*/0, &command,
          /*out_fixups=*/nullptr));

  iree_hal_amdgpu_aql_program_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace iree::hal::amdgpu
