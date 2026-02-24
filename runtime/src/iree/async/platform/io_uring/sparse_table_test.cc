// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for the sparse table slot allocator.
//
// The underlying bitmap operations (find_first_unset_span, set_span,
// reset_span) are tested in bitmap_test.cc. These tests focus on the allocator
// semantics: correct index returns, fragmentation recovery, full-table
// rejection, and multi-size interleaving.

#include "iree/async/platform/io_uring/sparse_table.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SparseTableTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = iree_allocator_system(); }

  void TearDown() override {
    iree_io_uring_sparse_table_free(table_, allocator_);
  }

  void AllocateTable(uint16_t capacity) {
    IREE_ASSERT_OK(
        iree_io_uring_sparse_table_allocate(capacity, allocator_, &table_));
  }

  iree_allocator_t allocator_ = iree_allocator_null();
  iree_io_uring_sparse_table_t* table_ = nullptr;
};

TEST_F(SparseTableTest, AcquireAndRelease) {
  AllocateTable(64);

  iree_io_uring_sparse_table_lock(table_);
  int32_t slot = iree_io_uring_sparse_table_acquire(table_, 1);
  EXPECT_EQ(slot, 0);

  iree_io_uring_sparse_table_release(table_, 0, 1);
  int32_t slot2 = iree_io_uring_sparse_table_acquire(table_, 1);
  EXPECT_EQ(slot2, 0);
  iree_io_uring_sparse_table_release(table_, 0, 1);
  iree_io_uring_sparse_table_unlock(table_);
}

TEST_F(SparseTableTest, ContiguousRange) {
  AllocateTable(256);

  iree_io_uring_sparse_table_lock(table_);
  int32_t first = iree_io_uring_sparse_table_acquire(table_, 64);
  EXPECT_EQ(first, 0);

  int32_t second = iree_io_uring_sparse_table_acquire(table_, 64);
  EXPECT_EQ(second, 64);

  iree_io_uring_sparse_table_release(table_, 0, 64);
  iree_io_uring_sparse_table_release(table_, 64, 64);
  iree_io_uring_sparse_table_unlock(table_);
}

TEST_F(SparseTableTest, FragmentationRecovery) {
  AllocateTable(256);

  iree_io_uring_sparse_table_lock(table_);

  // Allocate A(64 slots), then B(1 slot).
  int32_t a = iree_io_uring_sparse_table_acquire(table_, 64);
  EXPECT_EQ(a, 0);
  int32_t b = iree_io_uring_sparse_table_acquire(table_, 1);
  EXPECT_EQ(b, 64);

  // Free A. The first 64 slots are now free, but slot 64 is still held.
  iree_io_uring_sparse_table_release(table_, 0, 64);

  // Acquire C(64 slots) — should fit in the freed range at 0.
  int32_t c = iree_io_uring_sparse_table_acquire(table_, 64);
  EXPECT_EQ(c, 0);

  iree_io_uring_sparse_table_release(table_, 0, 64);
  iree_io_uring_sparse_table_release(table_, 64, 1);
  iree_io_uring_sparse_table_unlock(table_);
}

TEST_F(SparseTableTest, FullTable) {
  AllocateTable(128);

  iree_io_uring_sparse_table_lock(table_);

  // Fill all slots.
  int32_t all = iree_io_uring_sparse_table_acquire(table_, 128);
  EXPECT_EQ(all, 0);

  // Next acquire should fail.
  int32_t overflow = iree_io_uring_sparse_table_acquire(table_, 1);
  EXPECT_EQ(overflow, -1);

  iree_io_uring_sparse_table_release(table_, 0, 128);
  iree_io_uring_sparse_table_unlock(table_);
}

TEST_F(SparseTableTest, MixedSizes) {
  AllocateTable(256);

  iree_io_uring_sparse_table_lock(table_);

  // Interleave single-slot and multi-slot acquisitions.
  int32_t single1 = iree_io_uring_sparse_table_acquire(table_, 1);
  EXPECT_EQ(single1, 0);
  int32_t block = iree_io_uring_sparse_table_acquire(table_, 32);
  EXPECT_EQ(block, 1);
  int32_t single2 = iree_io_uring_sparse_table_acquire(table_, 1);
  EXPECT_EQ(single2, 33);

  // Free the block, acquire a larger block that fits in the gap.
  iree_io_uring_sparse_table_release(table_, 1, 32);
  int32_t reuse = iree_io_uring_sparse_table_acquire(table_, 16);
  EXPECT_EQ(reuse, 1);

  iree_io_uring_sparse_table_release(table_, 0, 1);
  iree_io_uring_sparse_table_release(table_, 1, 16);
  iree_io_uring_sparse_table_release(table_, 33, 1);
  iree_io_uring_sparse_table_unlock(table_);
}

TEST_F(SparseTableTest, Capacity) {
  AllocateTable(512);
  EXPECT_EQ(iree_io_uring_sparse_table_capacity(table_), 512);
}

TEST_F(SparseTableTest, AcquireZeroReturnsNegative) {
  AllocateTable(64);

  iree_io_uring_sparse_table_lock(table_);
  int32_t slot = iree_io_uring_sparse_table_acquire(table_, 0);
  EXPECT_EQ(slot, -1);
  iree_io_uring_sparse_table_unlock(table_);
}

TEST_F(SparseTableTest, FreeNull) {
  // Freeing a NULL table must be a no-op.
  iree_io_uring_sparse_table_free(nullptr, iree_allocator_system());
}

}  // namespace
