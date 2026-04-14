// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"

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

  CommandBufferPtr CreateCommandBuffer() {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_EXPECT_OK(iree_hal_amdgpu_aql_command_buffer_create(
        /*device_allocator=*/nullptr, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &block_pool_, iree_allocator_system(),
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

}  // namespace
}  // namespace iree::hal::amdgpu
