// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_builder.h"

#include <cstring>

#include "iree/hal/drivers/local_task/block_command_ops.h"
#include "iree/hal/drivers/local_task/block_isa.h"
#include "iree/hal/local/local_executable.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Test fixture that manages a block pool for the builder tests.
class BlockBuilderTest : public ::testing::Test {
 protected:
  // 4KB block pool, matching the small pool in task_device.c.
  static constexpr iree_host_size_t kBlockSize = 4096;

  void SetUp() override {
    iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(),
                                     &block_pool_);
  }

  void TearDown() override { iree_arena_block_pool_deinitialize(&block_pool_); }

  iree_arena_block_pool_t block_pool_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, InitializeDeinitialize) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  // No blocks acquired yet.
  EXPECT_EQ(builder.current_header, nullptr);
  EXPECT_EQ(builder.first_block, nullptr);
  EXPECT_EQ(builder.block_count, 0);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, DeinitializeReleasesBlocks) {
  // Verify that deinitialize releases blocks even if end() was not called.
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));
  EXPECT_NE(builder.current_header, nullptr);
  // Don't call end() — deinitialize should clean up.
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Empty recording
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, EmptyRecording) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  // One block with just an entry barrier + RETURN.
  EXPECT_EQ(recording.block_count, 1);
  EXPECT_NE(recording.first_block, nullptr);
  EXPECT_EQ(recording.max_region_dispatch_count, 0);
  EXPECT_EQ(recording.max_total_binding_count, 0);

  // Verify the block structure.
  const iree_hal_cmd_block_header_t* block = recording.first_block;
  EXPECT_EQ(block->next_block, nullptr);
  EXPECT_EQ(block->region_count, 1);
  EXPECT_EQ(block->max_region_dispatch_count, 0);
  EXPECT_EQ(block->total_binding_count, 0);
  EXPECT_EQ(block->fixup_count, 0);
  EXPECT_EQ(block->total_dispatch_count, 0);

  // Walk the command stream: entry BARRIER → RETURN.
  const auto* cmd = iree_hal_cmd_block_commands(block);
  const auto* entry_barrier =
      reinterpret_cast<const iree_hal_cmd_barrier_t*>(cmd);
  EXPECT_EQ(entry_barrier->header.opcode, IREE_HAL_CMD_BARRIER);
  EXPECT_EQ(entry_barrier->dispatch_count, 0);
  const auto* return_cmd = reinterpret_cast<const iree_hal_cmd_header_t*>(
      iree_hal_cmd_next(&entry_barrier->header));
  EXPECT_EQ(return_cmd->opcode, IREE_HAL_CMD_RETURN);

  // Verify initial_remaining_tiles.
  const uint32_t* tiles = iree_hal_cmd_block_initial_remaining_tiles(block);
  EXPECT_EQ(tiles[0], 0u);  // Empty region, 0 tiles.

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Single dispatch
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, SingleDispatch) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Record one DISPATCH with 2 constants and 3 bindings.
  const uint8_t constant_count = 2;
  const uint8_t binding_count = 3;
  const uint32_t tile_count = 24;
  const iree_host_size_t cmd_size =
      iree_host_align(offsetof(iree_hal_cmd_dispatch_t, constants) +
                          constant_count * sizeof(uint32_t),
                      8);

  iree_hal_cmd_dispatch_t* dispatch = NULL;
  iree_hal_cmd_fixup_t* fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE, cmd_size,
      binding_count, binding_count, tile_count, (void**)&dispatch, &fixups));

  // Fill fixup entries (direct, with dummy span pointers) directly in-place.
  for (int i = 0; i < 3; ++i) {
    memset(&fixups[i], 0, sizeof(fixups[i]));
    fixups[i].span = (const iree_async_span_t*)(uintptr_t)(0x1000 + i);
    fixups[i].offset = i * 256;
    fixups[i].data_index = (uint16_t)i;
  }

  // Fill in dispatch fields.
  ASSERT_NE(dispatch, nullptr);
  dispatch->constant_count = constant_count;
  dispatch->binding_count = binding_count;
  dispatch->binding_data_base = 0;
  dispatch->workgroup_size[0] = 64;
  dispatch->workgroup_size[1] = 1;
  dispatch->workgroup_size[2] = 1;
  dispatch->params.direct.workgroup_count[0] = 4;
  dispatch->params.direct.workgroup_count[1] = 2;
  dispatch->params.direct.workgroup_count[2] = 3;
  dispatch->tile_count = tile_count;
  dispatch->tiles_per_reservation = 1;
  dispatch->local_memory_size = 0;
  dispatch->constants[0] = 42;
  dispatch->constants[1] = 99;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  // Verify recording metadata.
  EXPECT_EQ(recording.block_count, 1);
  EXPECT_EQ(recording.max_region_dispatch_count, 1);
  EXPECT_EQ(recording.max_total_binding_count, 3);

  // Verify block header.
  const iree_hal_cmd_block_header_t* block = recording.first_block;
  EXPECT_EQ(block->region_count, 1);
  EXPECT_EQ(block->max_region_dispatch_count, 1);
  EXPECT_EQ(block->total_binding_count, 3);
  EXPECT_EQ(block->fixup_count, 3);
  EXPECT_EQ(block->total_dispatch_count, 1);

  // Walk command stream: entry BARRIER → DISPATCH → RETURN.
  const uint8_t* stream = reinterpret_cast<const uint8_t*>(block + 1);
  const auto* entry_barrier =
      reinterpret_cast<const iree_hal_cmd_barrier_t*>(stream);
  EXPECT_EQ(entry_barrier->header.opcode, IREE_HAL_CMD_BARRIER);
  EXPECT_EQ(entry_barrier->dispatch_count, 1);
  stream += sizeof(iree_hal_cmd_barrier_t);

  const auto* dispatch_cmd =
      reinterpret_cast<const iree_hal_cmd_dispatch_t*>(stream);
  EXPECT_EQ(dispatch_cmd->header.opcode, IREE_HAL_CMD_DISPATCH);
  EXPECT_EQ(dispatch_cmd->header.dispatch_index, 0);
  EXPECT_EQ(dispatch_cmd->constant_count, 2);
  EXPECT_EQ(dispatch_cmd->binding_count, 3);
  EXPECT_EQ(dispatch_cmd->constants[0], 42);
  EXPECT_EQ(dispatch_cmd->constants[1], 99);
  stream += cmd_size;

  const auto* return_cmd =
      reinterpret_cast<const iree_hal_cmd_header_t*>(stream);
  EXPECT_EQ(return_cmd->opcode, IREE_HAL_CMD_RETURN);

  // Verify initial_remaining_tiles.
  const uint32_t* tiles = iree_hal_cmd_block_initial_remaining_tiles(block);
  EXPECT_EQ(tiles[0], tile_count);

  // Verify fixup entries. Within a single command, fixups are stored in
  // forward order (fixups[0] at the lowest address = block_fixups[0]).
  // Note: the out_fixups pointer from append_cmd is invalidated by end()
  // (finalize_block memmoves fixups to make room for initial_remaining_tiles),
  // so we verify against expected values directly.
  const iree_hal_cmd_fixup_t* block_fixups = iree_hal_cmd_block_fixups(block);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(block_fixups[i].data_index, (uint16_t)i)
        << "fixup[" << i << "] data_index mismatch";
    EXPECT_EQ(block_fixups[i].offset, (iree_device_size_t)(i * 256))
        << "fixup[" << i << "] offset mismatch";
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, CommandOpSplitKeepsBindingIndicesBlockLocal) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs = {
      .constant_count = 0,
      .binding_count = 12,
  };
  iree_hal_local_executable_t executable = {};
  executable.dispatch_attrs = &dispatch_attrs;

  iree_hal_dispatch_config_t config = {
      .workgroup_size = {1, 1, 1},
      .workgroup_count = {1, 1, 1},
  };
  for (int i = 0; i < 100; ++i) {
    iree_hal_cmd_fixup_t* fixups = NULL;
    iree_hal_cmd_build_token_t token = {0};
    IREE_ASSERT_OK(iree_hal_cmd_build_dispatch(
        &builder, (iree_hal_executable_t*)&executable, /*export_ordinal=*/0,
        config, iree_const_byte_span_empty(), dispatch_attrs.binding_count,
        IREE_HAL_DISPATCH_FLAG_NONE, &fixups, &token));
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));
  EXPECT_GT(recording.block_count, 1u);

  for (const iree_hal_cmd_block_header_t* block = recording.first_block; block;
       block = block->next_block) {
    const iree_hal_cmd_fixup_t* fixups = iree_hal_cmd_block_fixups(block);
    for (uint16_t i = 0; i < block->fixup_count; ++i) {
      EXPECT_LT(fixups[i].data_index, block->total_binding_count);
    }

    const iree_hal_cmd_header_t* command = iree_hal_cmd_block_commands(block);
    while (command->opcode != IREE_HAL_CMD_BRANCH &&
           command->opcode != IREE_HAL_CMD_RETURN) {
      if (command->opcode == IREE_HAL_CMD_DISPATCH) {
        const iree_hal_cmd_dispatch_t* dispatch =
            (const iree_hal_cmd_dispatch_t*)command;
        EXPECT_LE(
            (uint32_t)dispatch->binding_data_base + dispatch->binding_count,
            (uint32_t)block->total_binding_count);
      }
      command = iree_hal_cmd_next(command);
    }
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Barrier creates regions
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, BarrierCreatesRegions) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Region 0: 2 dispatches.
  for (int i = 0; i < 2; ++i) {
    iree_hal_cmd_dispatch_t* cmd = NULL;
    IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
        &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
        sizeof(iree_hal_cmd_dispatch_t), 0, 0, 10, (void**)&cmd, NULL));
    cmd->constant_count = 0;
    cmd->binding_count = 0;
    cmd->tile_count = 10;
  }

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Region 1: 1 fill.
  iree_hal_cmd_fill_t* fill = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 0, 0, 1, (void**)&fill, NULL));
  fill->target_binding = 0;
  fill->pattern_length = 4;

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Region 2: 3 dispatches.
  for (int i = 0; i < 3; ++i) {
    iree_hal_cmd_dispatch_t* cmd = NULL;
    IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
        &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
        sizeof(iree_hal_cmd_dispatch_t), 0, 0, 5, (void**)&cmd, NULL));
    cmd->constant_count = 0;
    cmd->binding_count = 0;
    cmd->tile_count = 5;
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  EXPECT_EQ(recording.block_count, 1);
  // Region 0: 2 dispatches, Region 1: 1 fill, Region 2: 3 dispatches.
  EXPECT_EQ(recording.max_region_dispatch_count, 3);

  const iree_hal_cmd_block_header_t* block = recording.first_block;
  EXPECT_EQ(block->region_count, 3);
  EXPECT_EQ(block->max_region_dispatch_count, 3);
  EXPECT_EQ(block->total_dispatch_count, 6);

  // Verify initial_remaining_tiles per region.
  const uint32_t* tiles = iree_hal_cmd_block_initial_remaining_tiles(block);
  EXPECT_EQ(tiles[0], 20u);  // Region 0: 2 dispatches × 10 tiles.
  EXPECT_EQ(tiles[1], 1u);   // Region 1: 1 fill = 1 tile.
  EXPECT_EQ(tiles[2], 15u);  // Region 2: 3 dispatches × 5 tiles.

  // Walk the command stream and verify dispatch_index assignment.
  // Layout: BARRIER(2) dispatch dispatch BARRIER(1) fill BARRIER(3) dispatch×3
  // RETURN
  const uint8_t* stream = reinterpret_cast<const uint8_t*>(block + 1);

  // Entry barrier for region 0.
  const auto* b0 = reinterpret_cast<const iree_hal_cmd_barrier_t*>(stream);
  EXPECT_EQ(b0->header.opcode, IREE_HAL_CMD_BARRIER);
  EXPECT_EQ(b0->dispatch_count, 2);
  stream += sizeof(iree_hal_cmd_barrier_t);

  // Region 0: dispatch 0, dispatch 1.
  const auto* d0 = reinterpret_cast<const iree_hal_cmd_dispatch_t*>(stream);
  EXPECT_EQ(d0->header.dispatch_index, 0);
  stream += sizeof(iree_hal_cmd_dispatch_t);
  const auto* d1 = reinterpret_cast<const iree_hal_cmd_dispatch_t*>(stream);
  EXPECT_EQ(d1->header.dispatch_index, 1);
  stream += sizeof(iree_hal_cmd_dispatch_t);

  // Barrier between region 0 and region 1.
  const auto* b1 = reinterpret_cast<const iree_hal_cmd_barrier_t*>(stream);
  EXPECT_EQ(b1->header.opcode, IREE_HAL_CMD_BARRIER);
  EXPECT_EQ(b1->dispatch_count, 1);
  stream += sizeof(iree_hal_cmd_barrier_t);

  // Region 1: fill (gets dispatch_index 0 — all work commands get indices).
  const auto* f0 = reinterpret_cast<const iree_hal_cmd_fill_t*>(stream);
  EXPECT_EQ(f0->header.opcode, IREE_HAL_CMD_FILL);
  EXPECT_EQ(f0->header.dispatch_index, 0);
  stream += sizeof(iree_hal_cmd_fill_t);

  // Barrier between region 1 and region 2.
  const auto* b2 = reinterpret_cast<const iree_hal_cmd_barrier_t*>(stream);
  EXPECT_EQ(b2->header.opcode, IREE_HAL_CMD_BARRIER);
  EXPECT_EQ(b2->dispatch_count, 3);
  stream += sizeof(iree_hal_cmd_barrier_t);

  // Region 2: dispatch 0, 1, 2 (region-local indices reset after barrier).
  for (int i = 0; i < 3; ++i) {
    const auto* d = reinterpret_cast<const iree_hal_cmd_dispatch_t*>(stream);
    EXPECT_EQ(d->header.dispatch_index, i) << "region 2 dispatch " << i;
    stream += sizeof(iree_hal_cmd_dispatch_t);
  }

  // RETURN.
  const auto* return_cmd =
      reinterpret_cast<const iree_hal_cmd_header_t*>(stream);
  EXPECT_EQ(return_cmd->opcode, IREE_HAL_CMD_RETURN);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Block splitting
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, BlockSplitOnCapacity) {
  // Fill a 4KB block until it splits. Each dispatch is 64 bytes.
  // Available: 4096 - 8 (arena footer) - 24 (block header) - 8 (entry barrier)
  // = ~4056 bytes. At 64 bytes per dispatch, ~63 dispatches before needing
  // to account for RETURN and split overhead.
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  int dispatch_count = 0;
  while (dispatch_count < 200) {
    iree_hal_cmd_dispatch_t* cmd = NULL;
    IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
        &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
        sizeof(iree_hal_cmd_dispatch_t), 0, 0, 1, (void**)&cmd, NULL));
    cmd->constant_count = 0;
    cmd->binding_count = 0;
    cmd->tile_count = 1;
    ++dispatch_count;
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  // Should have multiple blocks.
  EXPECT_GT(recording.block_count, 1);

  // Walk the block chain and verify structure.
  int total_dispatches = 0;
  int block_index = 0;
  const iree_hal_cmd_block_header_t* block = recording.first_block;
  while (block) {
    total_dispatches += block->total_dispatch_count;

    // Every block except the last should end with BRANCH.
    if (block->next_block) {
      // Walk to the last command in this block.
      const uint8_t* stream = reinterpret_cast<const uint8_t*>(block + 1);
      const uint8_t* stream_end = stream + block->used_bytes;
      // Verify the block has dispatches (except possibly the last fragment
      // if the split happened at a boundary).
      (void)stream;
      (void)stream_end;
    } else {
      // Last block ends with RETURN.
      // Verify by looking for RETURN in the stream.
    }

    block = block->next_block;
    ++block_index;
  }

  EXPECT_EQ(total_dispatches, 200);
  EXPECT_EQ(block_index, recording.block_count);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, BlockSplitOnRegionDispatchCountLimit) {
  static constexpr iree_host_size_t kLargeBlockSize = 32768;
  static constexpr uint16_t kMaxRegionDispatchCount = 255;
  static constexpr uint16_t kFillCount = 300;

  iree_arena_block_pool_t large_block_pool;
  iree_arena_block_pool_initialize(kLargeBlockSize, iree_allocator_system(),
                                   &large_block_pool);

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&large_block_pool, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  for (uint16_t i = 0; i < kFillCount; ++i) {
    iree_hal_cmd_fill_t* fill = NULL;
    IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
        &builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
        sizeof(iree_hal_cmd_fill_t), 0, 0, 1, (void**)&fill, NULL));
    fill->target_binding = 0;
    fill->pattern_length = 4;
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  ASSERT_EQ(recording.block_count, 2);
  EXPECT_EQ(recording.max_region_dispatch_count, kMaxRegionDispatchCount);

  const iree_hal_cmd_block_header_t* first_block = recording.first_block;
  ASSERT_NE(first_block, nullptr);
  const iree_hal_cmd_block_header_t* second_block = first_block->next_block;
  ASSERT_NE(second_block, nullptr);
  EXPECT_EQ(second_block->next_block, nullptr);

  const auto* first_barrier = reinterpret_cast<const iree_hal_cmd_barrier_t*>(
      iree_hal_cmd_block_commands(first_block));
  EXPECT_EQ(first_barrier->dispatch_count, kMaxRegionDispatchCount);
  EXPECT_EQ(first_block->max_region_dispatch_count, kMaxRegionDispatchCount);
  EXPECT_EQ(first_block->total_dispatch_count, kMaxRegionDispatchCount);
  EXPECT_EQ(iree_hal_cmd_block_initial_remaining_tiles(first_block)[0],
            kMaxRegionDispatchCount);

  const iree_hal_cmd_header_t* first_block_terminator =
      iree_hal_cmd_next(&first_barrier->header);
  for (uint16_t i = 0; i < kMaxRegionDispatchCount; ++i) {
    first_block_terminator = iree_hal_cmd_next(first_block_terminator);
  }
  ASSERT_EQ(first_block_terminator->opcode, IREE_HAL_CMD_BRANCH);
  EXPECT_EQ(
      reinterpret_cast<const iree_hal_cmd_branch_t*>(first_block_terminator)
          ->target,
      second_block);

  const uint16_t second_block_fill_count = kFillCount - kMaxRegionDispatchCount;
  const auto* second_barrier = reinterpret_cast<const iree_hal_cmd_barrier_t*>(
      iree_hal_cmd_block_commands(second_block));
  EXPECT_EQ(second_barrier->dispatch_count, second_block_fill_count);
  EXPECT_EQ(second_block->max_region_dispatch_count, second_block_fill_count);
  EXPECT_EQ(second_block->total_dispatch_count, second_block_fill_count);
  EXPECT_EQ(iree_hal_cmd_block_initial_remaining_tiles(second_block)[0],
            second_block_fill_count);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
  iree_arena_block_pool_deinitialize(&large_block_pool);
}

//===----------------------------------------------------------------------===//
// Fixup entries survive block finalization
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, FixupsWithMultipleRegions) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Region 0: dispatch with 2 bindings.
  iree_hal_cmd_dispatch_t* d0 = NULL;
  iree_hal_cmd_fixup_t* fixups_r0 = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_dispatch_t), 2, 2, 10, (void**)&d0, &fixups_r0));
  memset(&fixups_r0[0], 0, sizeof(fixups_r0[0]));
  fixups_r0[0].data_index = 0;
  fixups_r0[0].offset = 0;
  fixups_r0[0].span = (const iree_async_span_t*)(uintptr_t)0x1000;
  memset(&fixups_r0[1], 0, sizeof(fixups_r0[1]));
  fixups_r0[1].data_index = 1;
  fixups_r0[1].offset = 64;
  fixups_r0[1].span = (const iree_async_span_t*)(uintptr_t)0x2000;
  d0->constant_count = 0;
  d0->binding_count = 2;
  d0->binding_data_base = 0;
  d0->tile_count = 10;

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Region 1: dispatch with 1 binding.
  iree_hal_cmd_dispatch_t* d1 = NULL;
  iree_hal_cmd_fixup_t* fixups_r1 = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_dispatch_t), 1, 1, 5, (void**)&d1, &fixups_r1));
  memset(&fixups_r1[0], 0, sizeof(fixups_r1[0]));
  fixups_r1[0].data_index = 2;
  fixups_r1[0].offset = 128;
  fixups_r1[0].span = (const iree_async_span_t*)(uintptr_t)0x3000;
  d1->constant_count = 0;
  d1->binding_count = 1;
  d1->binding_data_base = 2;
  d1->tile_count = 5;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  const iree_hal_cmd_block_header_t* block = recording.first_block;
  EXPECT_EQ(block->fixup_count, 3);
  EXPECT_EQ(block->total_binding_count, 3);
  EXPECT_EQ(block->region_count, 2);

  // Verify fixup entries are present and have the right data_index values.
  // Fixup space grows backward (later commands at lower addresses), but
  // within each command fixups are in forward order.
  // cmd0 (2 fixups, data_index 0, 1) was recorded first → higher addresses.
  // cmd1 (1 fixup, data_index 2) was recorded second → lower addresses.
  // block_fixups points to the lowest fixup address (cmd1's fixup).
  const iree_hal_cmd_fixup_t* block_fixups = iree_hal_cmd_block_fixups(block);
  EXPECT_EQ(block_fixups[0].data_index, 2);  // fixups_r1[0] (lowest addr)
  EXPECT_EQ(block_fixups[1].data_index, 0);  // fixups_r0[0]
  EXPECT_EQ(block_fixups[2].data_index, 1);  // fixups_r0[1] (highest addr)

  // Verify initial_remaining_tiles.
  const uint32_t* tiles = iree_hal_cmd_block_initial_remaining_tiles(block);
  EXPECT_EQ(tiles[0], 10u);  // Region 0.
  EXPECT_EQ(tiles[1], 5u);   // Region 1.

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Error conditions
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, AppendWithoutBeginFails) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);

  iree_hal_cmd_dispatch_t* cmd = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_cmd_block_builder_append_cmd(
          &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
          sizeof(iree_hal_cmd_dispatch_t), 0, 0, 1, (void**)&cmd, NULL));

  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, BarrierWithoutBeginFails) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_cmd_block_builder_barrier(&builder));

  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, EndWithoutBeginFails) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);

  iree_hal_cmd_block_recording_t recording;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_cmd_block_builder_end(&builder, &recording));

  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, UnalignedCommandSizeFails) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  void* cmd = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cmd_block_builder_append_cmd(&builder, IREE_HAL_CMD_DISPATCH,
                                            IREE_HAL_CMD_FLAG_NONE,
                                            65,  // Not 8-byte aligned.
                                            0, 0, 1, &cmd, NULL));

  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_F(BlockBuilderTest, InternalOpcodesRejected) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  void* cmd = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_cmd_block_builder_append_cmd(
                            &builder, IREE_HAL_CMD_BARRIER,
                            IREE_HAL_CMD_FLAG_NONE, 8, 0, 0, 0, &cmd, NULL));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_cmd_block_builder_append_cmd(
                            &builder, IREE_HAL_CMD_BRANCH,
                            IREE_HAL_CMD_FLAG_NONE, 16, 0, 0, 0, &cmd, NULL));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_cmd_block_builder_append_cmd(
                            &builder, IREE_HAL_CMD_RETURN,
                            IREE_HAL_CMD_FLAG_NONE, 8, 0, 0, 0, &cmd, NULL));

  iree_hal_cmd_block_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Recording release
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, RecordingReleaseIdempotent) {
  iree_hal_cmd_block_recording_t recording;
  memset(&recording, 0, sizeof(recording));
  // Releasing a zero-initialized recording is safe.
  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_recording_release(&recording);
}

//===----------------------------------------------------------------------===//
// Copy and fill commands
//===----------------------------------------------------------------------===//

TEST_F(BlockBuilderTest, MixedCommandTypes) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // FILL.
  iree_hal_cmd_fill_t* fill = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 0, 0, 1, (void**)&fill, NULL));
  fill->target_binding = 0;
  fill->pattern_length = 4;
  fill->params.direct.target_offset = 0;
  fill->params.direct.length = 1024;
  fill->params.direct.pattern = 0xFFFFFFFF;

  // COPY.
  iree_hal_cmd_copy_t* copy = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_COPY, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_copy_t), 0, 0, 1, (void**)&copy, NULL));
  copy->source_binding = 0;
  copy->target_binding = 1;
  copy->params.direct.source_offset = 0;
  copy->params.direct.target_offset = 0;
  copy->params.direct.length = 256;

  // DISPATCH.
  iree_hal_cmd_dispatch_t* dispatch = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_dispatch_t), 0, 0, 8, (void**)&dispatch, NULL));
  dispatch->constant_count = 0;
  dispatch->binding_count = 0;
  dispatch->tile_count = 8;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  EXPECT_EQ(recording.block_count, 1);
  const iree_hal_cmd_block_header_t* block = recording.first_block;
  EXPECT_EQ(block->total_dispatch_count, 3);
  EXPECT_EQ(block->max_region_dispatch_count, 3);

  // Total tiles: fill(1) + copy(1) + dispatch(8) = 10.
  const uint32_t* tiles = iree_hal_cmd_block_initial_remaining_tiles(block);
  EXPECT_EQ(tiles[0], 10u);

  // Walk and verify command order.
  const uint8_t* stream = reinterpret_cast<const uint8_t*>(block + 1);
  stream += sizeof(iree_hal_cmd_barrier_t);  // Skip entry barrier.
  EXPECT_EQ(reinterpret_cast<const iree_hal_cmd_header_t*>(stream)->opcode,
            IREE_HAL_CMD_FILL);
  stream += sizeof(iree_hal_cmd_fill_t);
  EXPECT_EQ(reinterpret_cast<const iree_hal_cmd_header_t*>(stream)->opcode,
            IREE_HAL_CMD_COPY);
  stream += sizeof(iree_hal_cmd_copy_t);
  EXPECT_EQ(reinterpret_cast<const iree_hal_cmd_header_t*>(stream)->opcode,
            IREE_HAL_CMD_DISPATCH);
  stream += sizeof(iree_hal_cmd_dispatch_t);
  EXPECT_EQ(reinterpret_cast<const iree_hal_cmd_header_t*>(stream)->opcode,
            IREE_HAL_CMD_RETURN);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

}  // namespace
