// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_isa.h"

#include <cstring>

#include "iree/testing/gtest.h"

namespace {

//===----------------------------------------------------------------------===//
// Command stream construction and walking
//===----------------------------------------------------------------------===//

// Helper: initialize a command header with the given opcode and byte size.
static void init_header(iree_hal_cmd_header_t* header, uint8_t opcode,
                        uint8_t size_bytes) {
  ASSERT_EQ(size_bytes % 8, 0) << "command size must be 8-byte aligned";
  ASSERT_LE(size_bytes, 255 * 8) << "command size exceeds maximum";
  header->opcode = opcode;
  header->flags = IREE_HAL_CMD_FLAG_NONE;
  header->size_qwords = size_bytes / 8;
  header->dispatch_index = 0;
}

// Construct a sequence of commands in a stack buffer and verify that walking
// via size_qwords lands at each expected command in order.
TEST(BlockISATest, CommandStreamWalk) {
  // A realistic command sequence: dispatch, fill, barrier, copy, return.
  alignas(8) uint8_t buffer[512];
  memset(buffer, 0, sizeof(buffer));

  uint8_t* cursor = buffer;

  // Command 0: DISPATCH (64 bytes fixed + 0 constants).
  auto* dispatch = reinterpret_cast<iree_hal_cmd_dispatch_t*>(cursor);
  init_header(&dispatch->header, IREE_HAL_CMD_DISPATCH,
              sizeof(iree_hal_cmd_dispatch_t));
  dispatch->constant_count = 0;
  dispatch->binding_count = 3;
  dispatch->binding_data_base = 0;
  dispatch->workgroup_size[0] = 64;
  dispatch->workgroup_size[1] = 1;
  dispatch->workgroup_size[2] = 1;
  dispatch->params.direct.workgroup_count[0] = 4;
  dispatch->params.direct.workgroup_count[1] = 2;
  dispatch->params.direct.workgroup_count[2] = 1;
  dispatch->tile_count = 8;
  dispatch->tiles_per_reservation = 1;
  cursor += sizeof(iree_hal_cmd_dispatch_t);

  // Command 1: FILL (32 bytes).
  auto* fill = reinterpret_cast<iree_hal_cmd_fill_t*>(cursor);
  init_header(&fill->header, IREE_HAL_CMD_FILL, sizeof(iree_hal_cmd_fill_t));
  fill->target_binding = 0;
  fill->pattern_length = 4;
  fill->params.direct.target_offset = 0;
  fill->params.direct.length = 1024;
  fill->params.direct.pattern = 0xFFFFFFFF;
  cursor += sizeof(iree_hal_cmd_fill_t);

  // Command 2: BARRIER (8 bytes).
  auto* barrier = reinterpret_cast<iree_hal_cmd_barrier_t*>(cursor);
  init_header(&barrier->header, IREE_HAL_CMD_BARRIER,
              sizeof(iree_hal_cmd_barrier_t));
  barrier->dispatch_count = 0;
  barrier->reserved = 0;
  barrier->worker_budget = 0;
  cursor += sizeof(iree_hal_cmd_barrier_t);

  // Command 3: COPY (32 bytes).
  auto* copy = reinterpret_cast<iree_hal_cmd_copy_t*>(cursor);
  init_header(&copy->header, IREE_HAL_CMD_COPY, sizeof(iree_hal_cmd_copy_t));
  copy->source_binding = 0;
  copy->target_binding = 1;
  copy->params.direct.source_offset = 0;
  copy->params.direct.target_offset = 0;
  copy->params.direct.length = 256;
  cursor += sizeof(iree_hal_cmd_copy_t);

  // Command 4: RETURN (8 bytes).
  auto* ret = reinterpret_cast<iree_hal_cmd_return_t*>(cursor);
  init_header(ret, IREE_HAL_CMD_RETURN, 8);
  cursor += 8;

  // Walk the command stream and verify opcodes in order.
  const auto* cmd = reinterpret_cast<const iree_hal_cmd_header_t*>(buffer);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_DISPATCH);
  cmd = iree_hal_cmd_next(cmd);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_FILL);
  cmd = iree_hal_cmd_next(cmd);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_BARRIER);
  cmd = iree_hal_cmd_next(cmd);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_COPY);
  cmd = iree_hal_cmd_next(cmd);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_RETURN);

  // Verify the final cursor position matches.
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(cmd) +
                (iree_host_size_t)cmd->size_qwords * 8,
            cursor);
}

// Verify a DISPATCH with trailing constants is navigable.
TEST(BlockISATest, DispatchWithConstants) {
  // Dispatch with 4 push constants: 60 + 4*4 = 76 bytes, aligned to 80.
  const size_t dispatch_size = iree_host_align(
      offsetof(iree_hal_cmd_dispatch_t, constants) + 4 * sizeof(uint32_t), 8);
  ASSERT_EQ(dispatch_size % 8, 0);

  alignas(8) uint8_t buffer[128];
  memset(buffer, 0, sizeof(buffer));

  auto* dispatch = reinterpret_cast<iree_hal_cmd_dispatch_t*>(buffer);
  init_header(&dispatch->header, IREE_HAL_CMD_DISPATCH,
              static_cast<uint8_t>(dispatch_size));
  dispatch->constant_count = 4;
  dispatch->binding_count = 2;

  // Write constants via the FAM.
  dispatch->constants[0] = 100;
  dispatch->constants[1] = 200;
  dispatch->constants[2] = 300;
  dispatch->constants[3] = 400;

  // Place a RETURN after the dispatch.
  auto* ret = reinterpret_cast<iree_hal_cmd_header_t*>(buffer + dispatch_size);
  init_header(ret, IREE_HAL_CMD_RETURN, 8);

  // Walk: dispatch → return.
  const auto* cmd = reinterpret_cast<const iree_hal_cmd_header_t*>(buffer);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_DISPATCH);
  cmd = iree_hal_cmd_next(cmd);
  EXPECT_EQ(cmd->opcode, IREE_HAL_CMD_RETURN);

  // Verify constant access via FAM.
  auto* read_dispatch =
      reinterpret_cast<const iree_hal_cmd_dispatch_t*>(buffer);
  EXPECT_EQ(read_dispatch->constants[0], 100);
  EXPECT_EQ(read_dispatch->constants[1], 200);
  EXPECT_EQ(read_dispatch->constants[2], 300);
  EXPECT_EQ(read_dispatch->constants[3], 400);
}

//===----------------------------------------------------------------------===//
// Block header navigation
//===----------------------------------------------------------------------===//

// Verify the dual-cursor block layout: commands at the front (after header),
// fixups and initial_remaining_tiles at the end.
TEST(BlockISATest, BlockHeaderNavigation) {
  // Simulate a 512-byte block pool block with:
  //   - 3 regions
  //   - 5 fixup entries (5 × 24 = 120 bytes)
  //   - 8 bytes of commands (just a RETURN)
  const uint16_t block_size = 512;
  const uint16_t region_count = 3;
  const uint16_t fixup_count = 5;
  const uint16_t used_bytes = 8;

  // Layout from end of block:
  //   initial_remaining_tiles: 3 × 4 = 12 bytes at [500..512)
  //   tile reservation: rounded up to fixup alignment = 16 bytes
  //   fixups: 5 × 24 = 120 bytes at [376..496)
  // Commands start at sizeof(header) = 24.
  const size_t tiles_offset = block_size - region_count * sizeof(uint32_t);
  const size_t tile_reservation =
      iree_hal_cmd_block_tile_reservation_size(region_count);
  const size_t fixups_offset = block_size - tile_reservation -
                               fixup_count * sizeof(iree_hal_cmd_fixup_t);
  const size_t commands_offset = sizeof(iree_hal_cmd_block_header_t);

  alignas(8) uint8_t buffer[512];
  memset(buffer, 0, sizeof(buffer));

  auto* header = reinterpret_cast<iree_hal_cmd_block_header_t*>(buffer);
  header->next_block = nullptr;
  header->used_bytes = used_bytes;
  header->region_count = region_count;
  header->max_region_dispatch_count = 5;
  header->total_binding_count = 10;
  header->fixup_count = fixup_count;
  header->total_dispatch_count = 5;
  header->block_size = block_size;

  // Write initial_remaining_tiles at the end.
  auto* tiles = reinterpret_cast<uint32_t*>(buffer + tiles_offset);
  tiles[0] = 100;
  tiles[1] = 200;
  tiles[2] = 50;

  // Verify commands start right after header.
  const auto* commands = iree_hal_cmd_block_commands(header);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(commands),
            buffer + commands_offset);

  // Verify initial_remaining_tiles navigation.
  const auto* nav_tiles = iree_hal_cmd_block_initial_remaining_tiles(header);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(nav_tiles), buffer + tiles_offset);
  EXPECT_EQ(nav_tiles[0], 100u);
  EXPECT_EQ(nav_tiles[1], 200u);
  EXPECT_EQ(nav_tiles[2], 50u);

  // Verify fixup navigation.
  const auto* fixups = iree_hal_cmd_block_fixups(header);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(fixups), buffer + fixups_offset);

  // Verify the gap between commands end and fixups start.
  size_t commands_end = commands_offset + used_bytes;
  EXPECT_LE(commands_end, fixups_offset)
      << "commands must not overlap with fixups";

  // Place a RETURN command and verify we can walk to it.
  auto* ret =
      reinterpret_cast<iree_hal_cmd_header_t*>(buffer + commands_offset);
  init_header(ret, IREE_HAL_CMD_RETURN, 8);
  EXPECT_EQ(commands->opcode, IREE_HAL_CMD_RETURN);
}

// Edge case: block with 0 regions and 0 fixups.
TEST(BlockISATest, BlockHeaderEmpty) {
  const uint16_t block_size = 256;

  alignas(8) uint8_t buffer[256];
  memset(buffer, 0, sizeof(buffer));

  auto* header = reinterpret_cast<iree_hal_cmd_block_header_t*>(buffer);
  header->region_count = 0;
  header->fixup_count = 0;
  header->used_bytes = 8;
  header->block_size = block_size;

  // Commands start right after header.
  const auto* commands = iree_hal_cmd_block_commands(header);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(commands),
            buffer + sizeof(iree_hal_cmd_block_header_t));

  // With 0 regions: initial_remaining_tiles is at block_end (empty).
  const auto* nav_tiles = iree_hal_cmd_block_initial_remaining_tiles(header);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(nav_tiles), buffer + block_size);

  // With 0 fixups: fixup pointer equals tiles pointer (empty).
  const auto* fixups = iree_hal_cmd_block_fixups(header);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(fixups),
            reinterpret_cast<const uint8_t*>(nav_tiles));
}

//===----------------------------------------------------------------------===//
// .data block state accessors
//===----------------------------------------------------------------------===//

// Verify that the accessor functions return non-overlapping, correctly-aligned
// pointers and that the sizing function matches.
TEST(BlockISATest, BlockStateLayout) {
  const uint16_t max_region_dispatch_count = 7;
  const uint16_t total_binding_count = 15;

  // Allocate .data.
  const iree_host_size_t state_size = iree_hal_cmd_block_state_size(
      max_region_dispatch_count, total_binding_count);
  EXPECT_GT(state_size, sizeof(iree_hal_cmd_block_state_t));

  alignas(64) uint8_t data[4096];
  ASSERT_LE(state_size, sizeof(data));
  memset(data, 0xCD, sizeof(data));  // Poison pattern for stale read detection.

  auto* state = reinterpret_cast<iree_hal_cmd_block_state_t*>(data);

  // Verify cache-line isolation of the fixed fields.
  auto active_addr = reinterpret_cast<uintptr_t>(&state->active_region_index);
  auto remaining_addr = reinterpret_cast<uintptr_t>(&state->remaining_tiles);
  EXPECT_EQ(active_addr % 64, 0)
      << "active_region_index must be cache-line aligned";
  EXPECT_EQ(remaining_addr % 64, 0)
      << "remaining_tiles must be cache-line aligned";
  EXPECT_GE(remaining_addr - active_addr, 64u)
      << "active_region_index and remaining_tiles must be on different cache "
         "lines";

  // Verify tile_index entries are cache-line strided.
  for (uint16_t i = 0; i < max_region_dispatch_count; ++i) {
    auto* tile_index = iree_hal_cmd_block_state_tile_index(state, i);
    auto tile_addr = reinterpret_cast<uintptr_t>(tile_index);
    auto state_addr = reinterpret_cast<uintptr_t>(state);
    // Each tile_index should be at state + sizeof(state) + i*64.
    EXPECT_EQ(tile_addr,
              state_addr + sizeof(iree_hal_cmd_block_state_t) +
                  (iree_host_size_t)i * IREE_HAL_CMD_TILE_INDEX_STRIDE)
        << "tile_index[" << i << "] at wrong offset";
    // Each tile_index must not share a cache line with its neighbors.
    if (i > 0) {
      auto* prev = iree_hal_cmd_block_state_tile_index(state, i - 1);
      auto prev_addr = reinterpret_cast<uintptr_t>(prev);
      EXPECT_GE(tile_addr - prev_addr, 64u)
          << "tile_index[" << i << "] shares cache line with tile_index["
          << (i - 1) << "]";
    }
  }

  // Verify binding arrays come after all tile indices.
  void** binding_ptrs =
      iree_hal_cmd_block_state_binding_ptrs(state, max_region_dispatch_count);
  size_t* binding_lengths = iree_hal_cmd_block_state_binding_lengths(
      state, max_region_dispatch_count, total_binding_count);

  auto ptrs_addr = reinterpret_cast<uintptr_t>(binding_ptrs);
  auto lengths_addr = reinterpret_cast<uintptr_t>(binding_lengths);
  auto state_addr = reinterpret_cast<uintptr_t>(state);

  // binding_ptrs must be after all tile indices.
  auto tiles_end = state_addr + sizeof(iree_hal_cmd_block_state_t) +
                   (iree_host_size_t)max_region_dispatch_count *
                       IREE_HAL_CMD_TILE_INDEX_STRIDE;
  EXPECT_GE(ptrs_addr, tiles_end);

  // Verify alignment.
  EXPECT_EQ(ptrs_addr % alignof(void*), 0)
      << "binding_ptrs must be pointer-aligned";
  EXPECT_EQ(lengths_addr % alignof(size_t), 0)
      << "binding_lengths must be size_t-aligned";

  // Verify ordering: binding_ptrs < binding_lengths.
  EXPECT_GT(lengths_addr, ptrs_addr);

  // Verify no overlap between binding_ptrs and binding_lengths.
  auto ptrs_end = ptrs_addr + total_binding_count * sizeof(void*);
  EXPECT_LE(ptrs_end, lengths_addr);

  // Verify the total size encompasses everything.
  auto lengths_end = lengths_addr + total_binding_count * sizeof(size_t);
  EXPECT_LE(lengths_end, state_addr + state_size);
}

// Verify state sizing accounts for cache-line padding.
TEST(BlockISATest, BlockStateSizingExtremes) {
  // Minimal: 0 dispatches per region, 0 bindings.
  EXPECT_GE(iree_hal_cmd_block_state_size(0, 0),
            sizeof(iree_hal_cmd_block_state_t));

  // Large: 255 dispatches per region (uint8_t max dispatch_index), 1000
  // bindings. Each tile_index is cache-line padded.
  iree_host_size_t large_size = iree_hal_cmd_block_state_size(255, 1000);
  EXPECT_GE(large_size, sizeof(iree_hal_cmd_block_state_t) +
                            255 * IREE_HAL_CMD_TILE_INDEX_STRIDE +
                            1000 * sizeof(void*) + 1000 * sizeof(size_t));

  // Single dispatch, single binding. The tile_index takes a full cache
  // line even though the atomic itself is only 4 bytes.
  iree_host_size_t tiny_size = iree_hal_cmd_block_state_size(1, 1);
  EXPECT_GE(tiny_size, sizeof(iree_hal_cmd_block_state_t) +
                           1 * IREE_HAL_CMD_TILE_INDEX_STRIDE +
                           1 * sizeof(void*) + 1 * sizeof(size_t));

  // Verify the struct itself is 2 cache lines (active_region_index +
  // remaining_tiles, each on their own line).
  EXPECT_EQ(sizeof(iree_hal_cmd_block_state_t), 128u);
}

// Verify that the accessor functions work at the sizes reported by
// iree_hal_cmd_block_state_size (i.e., no off-by-one in sizing).
TEST(BlockISATest, BlockStateSizeConsistency) {
  // Try several parameter combinations.
  struct TestCase {
    uint16_t max_dispatch_count;
    uint16_t total_binding_count;
  };
  TestCase cases[] = {
      {0, 0}, {1, 1}, {1, 100}, {10, 10}, {50, 3}, {255, 500}, {7, 240},
  };

  for (const auto& tc : cases) {
    iree_host_size_t size = iree_hal_cmd_block_state_size(
        tc.max_dispatch_count, tc.total_binding_count);

    // Allocate at the exact size, cache-line aligned (required for the
    // struct's alignas(64) fields to work correctly).
    std::vector<uint8_t> storage(size + 64, 0);
    auto aligned_base = reinterpret_cast<uintptr_t>(storage.data());
    aligned_base = (aligned_base + 63) & ~(uintptr_t)63;
    auto* state = reinterpret_cast<iree_hal_cmd_block_state_t*>(aligned_base);

    // Verify all tile_indices are within bounds.
    for (uint16_t i = 0; i < tc.max_dispatch_count; ++i) {
      auto* tile = iree_hal_cmd_block_state_tile_index(state, i);
      auto tile_addr = reinterpret_cast<uintptr_t>(tile);
      EXPECT_GE(tile_addr, aligned_base)
          << "tile_index[" << i << "] below allocation for "
          << "max_dispatch_count=" << tc.max_dispatch_count;
      EXPECT_LT(tile_addr, aligned_base + size)
          << "tile_index[" << i << "] beyond allocation for "
          << "max_dispatch_count=" << tc.max_dispatch_count;
    }

    if (tc.total_binding_count > 0) {
      size_t* lengths = iree_hal_cmd_block_state_binding_lengths(
          state, tc.max_dispatch_count, tc.total_binding_count);
      auto last_byte =
          reinterpret_cast<uintptr_t>(&lengths[tc.total_binding_count]);
      EXPECT_LE(last_byte - aligned_base, size)
          << "binding_lengths overflows state allocation for "
          << "max_dispatch_count=" << tc.max_dispatch_count
          << " total_binding_count=" << tc.total_binding_count;
    }
  }
}

//===----------------------------------------------------------------------===//
// DISPATCH command field access patterns
//===----------------------------------------------------------------------===//

// Verify the dispatch command's union discriminates correctly between
// direct and indirect modes.
TEST(BlockISATest, DispatchDirectVsIndirect) {
  iree_hal_cmd_dispatch_t dispatch;
  memset(&dispatch, 0, sizeof(dispatch));

  // Direct mode: inline workgroup_count.
  dispatch.header.flags = IREE_HAL_CMD_FLAG_NONE;
  dispatch.params.direct.workgroup_count[0] = 4;
  dispatch.params.direct.workgroup_count[1] = 2;
  dispatch.params.direct.workgroup_count[2] = 1;
  EXPECT_EQ(dispatch.params.direct.workgroup_count[0], 4);
  EXPECT_EQ(dispatch.params.direct.workgroup_count[1], 2);
  EXPECT_EQ(dispatch.params.direct.workgroup_count[2], 1);

  // Indirect mode: params_binding + params_offset + tile_count_hint
  // occupy the same memory as workgroup_count[3].
  dispatch.header.flags = IREE_HAL_CMD_FLAG_INDIRECT;
  memset(&dispatch.params.indirect, 0, sizeof(dispatch.params.indirect));
  dispatch.params.indirect.params_binding = 5;
  dispatch.params.indirect.params_offset = 0x100;
  dispatch.params.indirect.tile_count_hint = 42;
  EXPECT_EQ(dispatch.params.indirect.params_binding, 5);
  EXPECT_EQ(dispatch.params.indirect.params_offset, 0x100);
  EXPECT_EQ(dispatch.params.indirect.tile_count_hint, 42);

  // Verify the union overlaps: writing indirect should have clobbered
  // workgroup_count.
  EXPECT_NE(dispatch.params.direct.workgroup_count[0], 4);
}

// Verify FAM access for constants.
TEST(BlockISATest, DispatchConstantAccess) {
  alignas(8) uint8_t buffer[128];
  memset(buffer, 0, sizeof(buffer));

  auto* dispatch = reinterpret_cast<iree_hal_cmd_dispatch_t*>(buffer);
  dispatch->constant_count = 3;

  dispatch->constants[0] = 0xDEAD;
  dispatch->constants[1] = 0xBEEF;
  dispatch->constants[2] = 0xCAFE;

  // FAM starts at the offsetof, not sizeof (which includes trailing padding).
  EXPECT_EQ(reinterpret_cast<uintptr_t>(dispatch->constants),
            reinterpret_cast<uintptr_t>(buffer) +
                offsetof(iree_hal_cmd_dispatch_t, constants));
  EXPECT_EQ(dispatch->constants[0], 0xDEAD);
  EXPECT_EQ(dispatch->constants[1], 0xBEEF);
  EXPECT_EQ(dispatch->constants[2], 0xCAFE);
}

//===----------------------------------------------------------------------===//
// FILL and COPY direct/indirect unions
//===----------------------------------------------------------------------===//

TEST(BlockISATest, FillDirectIndirect) {
  iree_hal_cmd_fill_t fill;
  memset(&fill, 0, sizeof(fill));

  // Direct mode.
  fill.header.flags = IREE_HAL_CMD_FLAG_NONE;
  fill.target_binding = 3;
  fill.pattern_length = 4;
  fill.params.direct.target_offset = 0x1000;
  fill.params.direct.length = 0x2000;
  fill.params.direct.pattern = 0xABABABAB;
  EXPECT_EQ(fill.params.direct.target_offset, 0x1000);
  EXPECT_EQ(fill.params.direct.length, 0x2000);
  EXPECT_EQ(fill.params.direct.pattern, 0xABABABABu);

  // Indirect mode overwrites the same storage.
  fill.header.flags = IREE_HAL_CMD_FLAG_INDIRECT;
  memset(&fill.params.indirect, 0, sizeof(fill.params.indirect));
  fill.params.indirect.params_binding = 7;
  fill.params.indirect.params_offset = 0x400;
  EXPECT_EQ(fill.params.indirect.params_binding, 7);
  EXPECT_EQ(fill.params.indirect.params_offset, 0x400);
}

TEST(BlockISATest, CopyDirectIndirect) {
  iree_hal_cmd_copy_t copy;
  memset(&copy, 0, sizeof(copy));

  // Direct mode.
  copy.header.flags = IREE_HAL_CMD_FLAG_NONE;
  copy.source_binding = 0;
  copy.target_binding = 1;
  copy.params.direct.source_offset = 0;
  copy.params.direct.target_offset = 0x800;
  copy.params.direct.length = 0x400;
  EXPECT_EQ(copy.params.direct.source_offset, 0);
  EXPECT_EQ(copy.params.direct.target_offset, 0x800);
  EXPECT_EQ(copy.params.direct.length, 0x400);

  // Indirect mode.
  copy.header.flags = IREE_HAL_CMD_FLAG_INDIRECT;
  memset(&copy.params.indirect, 0, sizeof(copy.params.indirect));
  copy.params.indirect.params_binding = 2;
  copy.params.indirect.params_offset = 0x100;
  EXPECT_EQ(copy.params.indirect.params_binding, 2);
  EXPECT_EQ(copy.params.indirect.params_offset, 0x100);
}

//===----------------------------------------------------------------------===//
// BRANCH command
//===----------------------------------------------------------------------===//

TEST(BlockISATest, BranchTarget) {
  // Verify branch stores and retrieves a target block header pointer.
  alignas(8) uint8_t target_block[64];
  auto* target_header =
      reinterpret_cast<iree_hal_cmd_block_header_t*>(target_block);

  iree_hal_cmd_branch_t branch;
  memset(&branch, 0, sizeof(branch));
  init_header(&branch.header, IREE_HAL_CMD_BRANCH,
              sizeof(iree_hal_cmd_branch_t));
  branch.target = target_header;

  EXPECT_EQ(branch.target, target_header);
  EXPECT_EQ(branch.header.size_qwords, 2);
}

//===----------------------------------------------------------------------===//
// Block header flexible array member
//===----------------------------------------------------------------------===//

TEST(BlockISATest, InitialRemainingTilesAtEnd) {
  // Verify initial_remaining_tiles is located at the end of the block,
  // accessed via the navigation helper.
  const uint16_t block_size = 256;
  alignas(8) uint8_t buffer[256];
  memset(buffer, 0, sizeof(buffer));

  auto* header = reinterpret_cast<iree_hal_cmd_block_header_t*>(buffer);
  header->region_count = 4;
  header->fixup_count = 0;
  header->block_size = block_size;

  // Write tiles at the end of the block.
  auto* tiles =
      reinterpret_cast<uint32_t*>(buffer + block_size - 4 * sizeof(uint32_t));
  tiles[0] = 10;
  tiles[1] = 50;
  tiles[2] = 1;
  tiles[3] = 200;

  // Verify navigation helper finds them.
  const auto* nav_tiles = iree_hal_cmd_block_initial_remaining_tiles(header);
  EXPECT_EQ(nav_tiles[0], 10u);
  EXPECT_EQ(nav_tiles[1], 50u);
  EXPECT_EQ(nav_tiles[2], 1u);
  EXPECT_EQ(nav_tiles[3], 200u);

  // Verify the tiles are at the expected absolute position.
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(nav_tiles),
            buffer + block_size - 4 * sizeof(uint32_t));
}

}  // namespace
