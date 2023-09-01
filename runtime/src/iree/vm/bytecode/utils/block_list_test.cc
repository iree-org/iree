// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/utils/block_list.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::IsOkAndHolds;
using iree::testing::status::StatusIs;
using testing::ElementsAre;
using testing::Eq;

// Tests usage on empty lists.
TEST(BlockListTest, Empty) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(0u, allocator, &block_list));

  // Try finding with an empty list. Note that we expect the ordinal to be valid
  // even though we can't insert anything.
  iree_host_size_t ordinal = 0;
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_find(&block_list, 0u, &ordinal)),
      StatusIs(StatusCode::kNotFound));
  EXPECT_EQ(ordinal, 0);
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_find(&block_list, 123u, &ordinal)),
      StatusIs(StatusCode::kNotFound));
  EXPECT_EQ(ordinal, 0);

  // No blocks inserted to verify.
  IREE_EXPECT_OK(iree_vm_bytecode_block_list_verify(
      &block_list, iree_const_byte_span_empty()));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Valid IR usage for 3 blocks. Note that we insert them out of order: 1 2 0.
// These should be stored inline in the block list struct.
TEST(BlockListTest, Valid) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(3u, allocator, &block_list));

  // Try finding blocks before anything is defined.
  iree_host_size_t ordinal = 0;
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_find(&block_list, 0u, &ordinal)),
      StatusIs(StatusCode::kNotFound));
  EXPECT_EQ(ordinal, 0);
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_find(&block_list, 123u, &ordinal)),
      StatusIs(StatusCode::kNotFound));
  EXPECT_EQ(ordinal, 0);

  iree_vm_bytecode_block_t* block = NULL;

  // Define block 1.
  block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 1u, &block));
  EXPECT_EQ(block_list.count, 1);
  EXPECT_EQ(block->defined, 0);
  EXPECT_EQ(block->pc, 1u);
  block->defined = 1;

  // Define block 2.
  block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 2u, &block));
  EXPECT_EQ(block_list.count, 2);
  EXPECT_EQ(block->defined, 0);
  EXPECT_EQ(block->pc, 2u);
  block->defined = 1;

  // Define block 0.
  block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));
  EXPECT_EQ(block_list.count, 3);
  EXPECT_EQ(block->defined, 0);
  EXPECT_EQ(block->pc, 0u);
  block->defined = 1;

  // Re-insert block 1. Should be a no-op as it is defined.
  block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 1u, &block));
  EXPECT_EQ(block_list.count, 3);
  EXPECT_EQ(block->defined, 1);
  EXPECT_EQ(block->pc, 1u);

  // Find each block and ensure they match.
  IREE_EXPECT_OK(iree_vm_bytecode_block_list_find(&block_list, 0u, &ordinal));
  EXPECT_EQ(ordinal, 0);
  IREE_EXPECT_OK(iree_vm_bytecode_block_list_find(&block_list, 1u, &ordinal));
  EXPECT_EQ(ordinal, 1);
  IREE_EXPECT_OK(iree_vm_bytecode_block_list_find(&block_list, 2u, &ordinal));
  EXPECT_EQ(ordinal, 2);

  // Verify blocks are all defined and have block markers.
  std::vector<uint8_t> bytecode_data = {
      IREE_VM_OP_CORE_Block, IREE_VM_OP_CORE_Block, IREE_VM_OP_CORE_Block,
      IREE_VM_OP_CORE_AbsI32,  // need at least one op in a block
  };
  IREE_EXPECT_OK(iree_vm_bytecode_block_list_verify(
      &block_list,
      iree_make_const_byte_span(bytecode_data.data(), bytecode_data.size())));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests that a declared block that was never defined errors on verification.
TEST(BlockListTest, Undefined) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(1u, allocator, &block_list));

  // Declare the block.
  iree_vm_bytecode_block_t* block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));

  // Fail verification because it hasn't been defined.
  std::vector<uint8_t> bytecode_data = {
      IREE_VM_OP_CORE_Block,
  };
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_verify(
          &block_list, iree_make_const_byte_span(bytecode_data.data(),
                                                 bytecode_data.size()))),
      StatusIs(StatusCode::kInvalidArgument));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests adding fewer blocks than expected by the capacity.
TEST(BlockListTest, Underflow) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(2u, allocator, &block_list));

  // Declaring; OK (count = 1, capacity = 2).
  iree_vm_bytecode_block_t* block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));

  // Defining; OK (no change in count).
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));
  block->defined = 1;

  // Fail verification because we're missing a block.
  std::vector<uint8_t> bytecode_data = {
      IREE_VM_OP_CORE_Block,
      IREE_VM_OP_CORE_AbsI32,  // need at least one op in a block
  };
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_verify(
          &block_list, iree_make_const_byte_span(bytecode_data.data(),
                                                 bytecode_data.size()))),
      StatusIs(StatusCode::kInvalidArgument));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests adding more blocks than allowed by the capacity.
TEST(BlockListTest, Overflow) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(1u, allocator, &block_list));

  // Declaring; OK (count = 1, capacity = 1).
  iree_vm_bytecode_block_t* block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));
  EXPECT_EQ(block_list.count, 1);

  // Defining; OK (no change in count).
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));
  block->defined = 1;

  // Error: too many blocks.
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_insert(&block_list, 1u, &block)),
      StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(block_list.count, 1);

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests adding any blocks to an expected-empty list.
TEST(BlockListTest, OverflowEmpty) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(0u, allocator, &block_list));

  // Error: too many blocks.
  iree_vm_bytecode_block_t* block = NULL;
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block)),
      StatusIs(StatusCode::kInvalidArgument));
  EXPECT_EQ(block_list.count, 0);

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests a block that is missing its marker in the bytecode.
TEST(BlockListTest, MissingMarker) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(1u, allocator, &block_list));

  iree_vm_bytecode_block_t* block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 0u, &block));
  block->defined = 1;

  std::vector<uint8_t> bytecode_data = {
      IREE_VM_OP_CORE_AbsI32,  // *not* the block marker
  };
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_verify(
          &block_list, iree_make_const_byte_span(bytecode_data.data(),
                                                 bytecode_data.size()))),
      StatusIs(StatusCode::kInvalidArgument));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests a block with a pc outside of the bytecode range.
TEST(BlockListTest, OutOfBoundsPC) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(1u, allocator, &block_list));

  iree_vm_bytecode_block_t* block = NULL;
  IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, 1u, &block));
  block->defined = 1;

  std::vector<uint8_t> bytecode_data = {
      IREE_VM_OP_CORE_AbsI32,  // *not* the block marker
  };
  EXPECT_THAT(
      Status(iree_vm_bytecode_block_list_verify(
          &block_list, iree_make_const_byte_span(bytecode_data.data(),
                                                 bytecode_data.size()))),
      StatusIs(StatusCode::kInvalidArgument));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests inserting a block with a PC outside of what we can track. This should
// be really rare in practice.
TEST(BlockListTest, OverMaxPC) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(1u, allocator, &block_list));

  iree_vm_bytecode_block_t* block = NULL;
  EXPECT_THAT(Status(iree_vm_bytecode_block_list_insert(
                  &block_list, IREE_VM_PC_BLOCK_MAX + 1, &block)),
              StatusIs(StatusCode::kOutOfRange));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

// Tests adding a lot of blocks such that we trigger a heap storage allocation.
TEST(BlockListTest, HeapStorage) {
  uint32_t count = IREE_VM_BYTECODE_INLINE_BLOCK_LIST_CAPACITY * 8;
  iree_allocator_t allocator = iree_allocator_system();
  iree_vm_bytecode_block_list_t block_list;
  IREE_ASSERT_OK(
      iree_vm_bytecode_block_list_initialize(count, allocator, &block_list));

  // Declare all blocks in reverse, for fun.
  for (uint32_t i = 0; i < count; ++i) {
    iree_vm_bytecode_block_t* block = NULL;
    IREE_ASSERT_OK(
        iree_vm_bytecode_block_list_insert(&block_list, count - i - 1, &block));
  }

  // Ensure sorted.
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(block_list.values[i].pc, i);
  }

  // Define all blocks forward.
  for (uint32_t i = 0; i < count; ++i) {
    iree_vm_bytecode_block_t* block = NULL;
    IREE_ASSERT_OK(iree_vm_bytecode_block_list_insert(&block_list, i, &block));
    block->defined = 1;
  }

  // Fake block data (+1 trailing block op for padding) to verify.
  std::vector<uint8_t> bytecode_data(count + 1, IREE_VM_OP_CORE_Block);
  IREE_EXPECT_OK(iree_vm_bytecode_block_list_verify(
      &block_list,
      iree_make_const_byte_span(bytecode_data.data(), bytecode_data.size())));

  iree_vm_bytecode_block_list_deinitialize(&block_list, allocator);
}

}  // namespace
