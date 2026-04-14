// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

#include <array>
#include <cstring>
#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct CommandBufferDeleter {
  void operator()(iree_hal_command_buffer_t* command_buffer) const {
    iree_hal_command_buffer_release(command_buffer);
  }
};

using CommandBufferPtr =
    std::unique_ptr<iree_hal_command_buffer_t, CommandBufferDeleter>;

class AqlCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdgpu_aql_program_block_pool_initialize(
        block_size_, iree_allocator_system(), &block_pool_));
  }

  void TearDown() override { iree_arena_block_pool_deinitialize(&block_pool_); }

  CommandBufferPtr CreateCommandBuffer(iree_host_size_t binding_capacity = 0) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_amdgpu_aql_command_buffer_create(
        /*device_allocator=*/nullptr, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &block_pool_, iree_allocator_system(),
        &command_buffer));
    return CommandBufferPtr(command_buffer);
  }

 private:
  iree_host_size_t block_size_ = 256;
  iree_arena_block_pool_t block_pool_;
};

TEST_F(AqlCommandBufferTest, UnrecordedCommandBufferHasNoProgram) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  EXPECT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer.get()));
  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer.get());
  EXPECT_EQ(program->first_block, nullptr);
}

TEST_F(AqlCommandBufferTest, EmptyRecordingHasReturnTerminator) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer.get());
  ASSERT_NE(program->first_block, nullptr);
  EXPECT_EQ(program->block_count, 1u);
  EXPECT_EQ(program->command_count, 1u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  EXPECT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
}

TEST_F(AqlCommandBufferTest, BarrierOnlyRecordingHasBarrierAndReturn) {
  CommandBufferPtr command_buffer = CreateCommandBuffer();
  ASSERT_NE(command_buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer.get(), IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer.get());
  ASSERT_NE(program->first_block, nullptr);
  EXPECT_EQ(program->block_count, 1u);
  EXPECT_EQ(program->command_count, 2u);

  const iree_hal_amdgpu_command_buffer_command_header_t* barrier_command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  EXPECT_EQ(barrier_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER);
  const iree_hal_amdgpu_command_buffer_command_header_t* return_command =
      iree_hal_amdgpu_command_buffer_command_next_const(barrier_command);
  EXPECT_EQ(return_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN);
}

TEST_F(AqlCommandBufferTest, UpdatePayloadsUseStableRodataOrdinals) {
  CommandBufferPtr command_buffer = CreateCommandBuffer(/*binding_capacity=*/1);
  ASSERT_NE(command_buffer, nullptr);

  std::array<uint8_t, 300> source_bytes0;
  for (size_t i = 0; i < source_bytes0.size(); ++i) {
    source_bytes0[i] = (uint8_t)i;
  }
  std::array<uint8_t, 19> source_bytes1;
  for (size_t i = 0; i < source_bytes1.size(); ++i) {
    source_bytes1[i] = (uint8_t)(0xE0u + i);
  }

  iree_hal_buffer_ref_t target_ref0 = {0};
  target_ref0.buffer_slot = 0;
  target_ref0.offset = 0;
  target_ref0.length = source_bytes0.size();
  iree_hal_buffer_ref_t target_ref1 = {0};
  target_ref1.buffer_slot = 0;
  target_ref1.offset = source_bytes0.size();
  target_ref1.length = 7;

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer.get()));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer.get(), source_bytes0.data(), /*source_offset=*/0,
      target_ref0, IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer.get(), source_bytes1.data(), /*source_offset=*/5,
      target_ref1, IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer.get()));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer.get());
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->command_count, 3u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command0 =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(command0->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE);
  const iree_hal_amdgpu_command_buffer_update_command_t* update0 =
      (const iree_hal_amdgpu_command_buffer_update_command_t*)command0;
  const uint8_t* rodata0 = iree_hal_amdgpu_aql_command_buffer_rodata(
      command_buffer.get(), update0->rodata_ordinal, update0->length);
  ASSERT_NE(rodata0, nullptr);
  EXPECT_EQ(0,
            std::memcmp(rodata0, source_bytes0.data(), source_bytes0.size()));

  const iree_hal_amdgpu_command_buffer_command_header_t* command1 =
      iree_hal_amdgpu_command_buffer_command_next_const(command0);
  ASSERT_EQ(command1->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE);
  const iree_hal_amdgpu_command_buffer_update_command_t* update1 =
      (const iree_hal_amdgpu_command_buffer_update_command_t*)command1;
  const uint8_t* rodata1 = iree_hal_amdgpu_aql_command_buffer_rodata(
      command_buffer.get(), update1->rodata_ordinal, update1->length);
  ASSERT_NE(rodata1, nullptr);
  EXPECT_EQ(0, std::memcmp(rodata1, source_bytes1.data() + 5,
                           (size_t)target_ref1.length));
}

}  // namespace
}  // namespace iree::hal::amdgpu
