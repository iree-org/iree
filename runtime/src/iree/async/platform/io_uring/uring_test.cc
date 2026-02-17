// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/uring.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Test fixture that handles io_uring availability.
// Skips tests if io_uring is not supported on this system.
class RingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(&ring_, 0, sizeof(ring_));
    iree_io_uring_ring_options_t options = iree_io_uring_ring_options_default();
    iree_status_t status = iree_io_uring_ring_initialize(options, &ring_);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "io_uring not available on this system";
    }
    IREE_ASSERT_OK(status);
    ring_initialized_ = true;
  }

  void TearDown() override {
    if (ring_initialized_) {
      iree_io_uring_ring_deinitialize(&ring_);
    }
  }

  iree_io_uring_ring_t ring_;
  bool ring_initialized_ = false;
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(RingTest, InitializePopulatesFields) {
  // Ring fd should be valid.
  EXPECT_GE(ring_.ring_fd, 0);

  // Ring pointers should be non-null.
  EXPECT_NE(ring_.sq_ring_ptr, nullptr);
  EXPECT_NE(ring_.sqes, nullptr);
  EXPECT_NE(ring_.cq_ring_ptr, nullptr);

  // Cached values should be set.
  EXPECT_GT(ring_.sq_entries, 0u);
  EXPECT_GT(ring_.cq_entries, 0u);
  EXPECT_GT(ring_.sq_mask, 0u);
  EXPECT_GT(ring_.cq_mask, 0u);

  // Entries should be power of 2.
  EXPECT_EQ(ring_.sq_entries & (ring_.sq_entries - 1), 0u);
  EXPECT_EQ(ring_.cq_entries & (ring_.cq_entries - 1), 0u);
}

TEST_F(RingTest, DefaultOptionsUses256Entries) {
  // Default options request 256 entries.
  iree_io_uring_ring_options_t options = iree_io_uring_ring_options_default();
  EXPECT_EQ(options.sq_entries, 256u);

  // Ring should have at least 256 entries (may be clamped by kernel).
  EXPECT_GE(ring_.sq_entries, 256u);
}

TEST(RingLifecycleTest, DeinitializeOnZeroInitializedIsSafe) {
  iree_io_uring_ring_t ring;
  memset(&ring, 0, sizeof(ring));
  ring.ring_fd = -1;  // Mark as not opened.

  // Should not crash.
  iree_io_uring_ring_deinitialize(&ring);
}

TEST(RingLifecycleTest, InitializeWithCustomEntries) {
  iree_io_uring_ring_t ring;
  memset(&ring, 0, sizeof(ring));

  iree_io_uring_ring_options_t options = iree_io_uring_ring_options_default();
  options.sq_entries = 64;

  iree_status_t status = iree_io_uring_ring_initialize(options, &ring);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "io_uring not available";
  }
  IREE_ASSERT_OK(status);

  // Should have at least 64 entries (rounded to power of 2).
  EXPECT_GE(ring.sq_entries, 64u);

  iree_io_uring_ring_deinitialize(&ring);
}

TEST(RingLifecycleTest, InitializeWithZeroEntriesUsesDefault) {
  iree_io_uring_ring_t ring;
  memset(&ring, 0, sizeof(ring));

  iree_io_uring_ring_options_t options = {0};
  options.sq_entries = 0;  // Should use default 256.

  iree_status_t status = iree_io_uring_ring_initialize(options, &ring);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "io_uring not available";
  }
  IREE_ASSERT_OK(status);

  EXPECT_GE(ring.sq_entries, 256u);

  iree_io_uring_ring_deinitialize(&ring);
}

TEST(RingLifecycleTest, InitializeRoundsUpToPowerOf2) {
  iree_io_uring_ring_t ring;
  memset(&ring, 0, sizeof(ring));

  iree_io_uring_ring_options_t options = iree_io_uring_ring_options_default();
  options.sq_entries = 100;  // Not a power of 2.

  iree_status_t status = iree_io_uring_ring_initialize(options, &ring);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "io_uring not available";
  }
  IREE_ASSERT_OK(status);

  // Should round up to 128.
  EXPECT_GE(ring.sq_entries, 128u);
  EXPECT_EQ(ring.sq_entries & (ring.sq_entries - 1), 0u);  // Power of 2.

  iree_io_uring_ring_deinitialize(&ring);
}

//===----------------------------------------------------------------------===//
// Submission queue tests
//===----------------------------------------------------------------------===//

TEST_F(RingTest, FreshRingHasFullSpaceLeft) {
  uint32_t space = iree_io_uring_ring_sq_space_left(&ring_);
  EXPECT_EQ(space, ring_.sq_entries);
}

TEST_F(RingTest, GetSqeReturnsNonNullZeroedSqe) {
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);

  // SQE should be zeroed.
  EXPECT_EQ(sqe->opcode, 0u);
  EXPECT_EQ(sqe->flags, 0u);
  EXPECT_EQ(sqe->fd, 0);
  EXPECT_EQ(sqe->user_data, 0u);
}

TEST_F(RingTest, GetSqeReducesSpaceLeft) {
  uint32_t initial_space = iree_io_uring_ring_sq_space_left(&ring_);
  ASSERT_GT(initial_space, 0u);

  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);

  uint32_t space_after = iree_io_uring_ring_sq_space_left(&ring_);
  EXPECT_EQ(space_after, initial_space - 1);
}

TEST_F(RingTest, GetSqeReturnsDistinctSlots) {
  iree_io_uring_sqe_t* sqe1 = iree_io_uring_ring_get_sqe(&ring_);
  iree_io_uring_sqe_t* sqe2 = iree_io_uring_ring_get_sqe(&ring_);

  ASSERT_NE(sqe1, nullptr);
  ASSERT_NE(sqe2, nullptr);
  EXPECT_NE(sqe1, sqe2);
}

TEST_F(RingTest, GetSqeReturnsNullWhenFull) {
  // Fill the entire SQ.
  for (uint32_t i = 0; i < ring_.sq_entries; ++i) {
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
    ASSERT_NE(sqe, nullptr) << "Failed to get SQE " << i;
  }

  // Next get should return NULL.
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  EXPECT_EQ(sqe, nullptr);
  EXPECT_EQ(iree_io_uring_ring_sq_space_left(&ring_), 0u);
}

//===----------------------------------------------------------------------===//
// Two-phase commit tests (pending/rollback)
//===----------------------------------------------------------------------===//

TEST_F(RingTest, GetSqeIncrementsPendingNotKernelTail) {
  // Initially no pending SQEs.
  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 0u);

  // Get an SQE - should increment pending but not kernel tail.
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);

  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 1u);
}

TEST_F(RingTest, MultipleSqesAccumulatePending) {
  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 0u);

  for (int i = 0; i < 5; ++i) {
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
    ASSERT_NE(sqe, nullptr);
    EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_),
              static_cast<uint32_t>(i + 1));
  }
}

TEST_F(RingTest, RollbackRestoresPendingCount) {
  // Prepare 5 SQEs.
  for (int i = 0; i < 5; ++i) {
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
    ASSERT_NE(sqe, nullptr);
  }
  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 5u);

  // Rollback 3.
  iree_io_uring_ring_sq_rollback(&ring_, 3);
  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 2u);

  // Rollback remaining.
  iree_io_uring_ring_sq_rollback(&ring_, 2);
  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 0u);
}

TEST_F(RingTest, RollbackRestoresSpaceLeft) {
  uint32_t initial_space = iree_io_uring_ring_sq_space_left(&ring_);

  // Prepare 3 SQEs.
  for (int i = 0; i < 3; ++i) {
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
    ASSERT_NE(sqe, nullptr);
  }
  EXPECT_EQ(iree_io_uring_ring_sq_space_left(&ring_), initial_space - 3);

  // Rollback all 3.
  iree_io_uring_ring_sq_rollback(&ring_, 3);
  EXPECT_EQ(iree_io_uring_ring_sq_space_left(&ring_), initial_space);
}

TEST_F(RingTest, SubmitClearsPending) {
  // Prepare a NOP.
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;

  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 1u);

  // Submit flushes pending to kernel.
  IREE_ASSERT_OK(iree_io_uring_ring_submit(&ring_,
                                           /*min_complete=*/0, /*flags=*/0));

  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 0u);
}

TEST_F(RingTest, WaitCqeWithFlushSubmitsPendingSqe) {
  // Prepare a NOP but don't submit it yet.
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;
  sqe->user_data = 0xDEADBEEF;

  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 1u);

  // Wait with flush_pending=true should submit and wait.
  IREE_ASSERT_OK(iree_io_uring_ring_wait_cqe(&ring_, /*min_complete=*/1,
                                             /*flush_pending=*/true,
                                             IREE_DURATION_INFINITE));

  // Pending should be cleared.
  EXPECT_EQ(iree_io_uring_ring_sq_pending(&ring_), 0u);

  // CQE should be available with our user_data.
  iree_io_uring_cqe_t* cqe = iree_io_uring_ring_peek_cqe(&ring_);
  ASSERT_NE(cqe, nullptr);
  EXPECT_EQ(cqe->user_data, 0xDEADBEEFu);

  iree_io_uring_ring_cq_advance(&ring_, 1);
}

//===----------------------------------------------------------------------===//
// Completion queue tests
//===----------------------------------------------------------------------===//

TEST_F(RingTest, FreshRingHasEmptyCq) {
  EXPECT_FALSE(iree_io_uring_ring_cq_ready(&ring_));
  EXPECT_EQ(iree_io_uring_ring_cq_count(&ring_), 0u);
  EXPECT_EQ(iree_io_uring_ring_peek_cqe(&ring_), nullptr);
}

//===----------------------------------------------------------------------===//
// Submit and completion tests
//===----------------------------------------------------------------------===//

TEST_F(RingTest, SubmitNopAndWait) {
  // Prepare a NOP operation.
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;
  sqe->user_data = 0x12345678;

  // Submit and wait for completion.
  IREE_ASSERT_OK(iree_io_uring_ring_submit(&ring_,
                                           /*min_complete=*/1,
                                           IREE_IORING_ENTER_GETEVENTS));

  // CQE should be available.
  EXPECT_TRUE(iree_io_uring_ring_cq_ready(&ring_));
  EXPECT_GE(iree_io_uring_ring_cq_count(&ring_), 1u);

  // Verify CQE contents.
  iree_io_uring_cqe_t* cqe = iree_io_uring_ring_peek_cqe(&ring_);
  ASSERT_NE(cqe, nullptr);
  EXPECT_EQ(cqe->user_data, 0x12345678u);
  EXPECT_EQ(cqe->res, 0);  // NOP returns 0 on success.

  // Advance CQ.
  iree_io_uring_ring_cq_advance(&ring_, 1);
  EXPECT_FALSE(iree_io_uring_ring_cq_ready(&ring_));
}

TEST_F(RingTest, SubmitMultipleNops) {
  const int kNopCount = 5;

  // Prepare multiple NOP operations.
  for (int i = 0; i < kNopCount; ++i) {
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
    ASSERT_NE(sqe, nullptr);
    sqe->opcode = IREE_IORING_OP_NOP;
    sqe->fd = -1;
    sqe->user_data = static_cast<uint64_t>(i);
  }

  // Submit all and wait.
  IREE_ASSERT_OK(iree_io_uring_ring_submit(&ring_,
                                           /*min_complete=*/kNopCount,
                                           IREE_IORING_ENTER_GETEVENTS));

  // All should complete.
  EXPECT_GE(iree_io_uring_ring_cq_count(&ring_),
            static_cast<uint32_t>(kNopCount));

  // Drain all CQEs.
  for (int i = 0; i < kNopCount; ++i) {
    iree_io_uring_cqe_t* cqe = iree_io_uring_ring_peek_cqe(&ring_);
    ASSERT_NE(cqe, nullptr);
    EXPECT_EQ(cqe->res, 0);
    iree_io_uring_ring_cq_advance(&ring_, 1);
  }

  EXPECT_FALSE(iree_io_uring_ring_cq_ready(&ring_));
}

TEST_F(RingTest, SubmitWithNoPendingIsNoop) {
  // Submitting with no pending SQEs should succeed and do nothing.
  IREE_ASSERT_OK(iree_io_uring_ring_submit(&ring_,
                                           /*min_complete=*/0, /*flags=*/0));
  EXPECT_FALSE(iree_io_uring_ring_cq_ready(&ring_));
}

//===----------------------------------------------------------------------===//
// Wait tests
//===----------------------------------------------------------------------===//

TEST_F(RingTest, WaitCqeWithAvailableCqeReturnsImmediately) {
  // Submit a NOP.
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&ring_);
  ASSERT_NE(sqe, nullptr);
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;

  IREE_ASSERT_OK(iree_io_uring_ring_submit(&ring_,
                                           /*min_complete=*/1,
                                           IREE_IORING_ENTER_GETEVENTS));

  // Wait should return immediately since CQE is already available.
  IREE_ASSERT_OK(iree_io_uring_ring_wait_cqe(&ring_, /*min_complete=*/1,
                                             /*flush_pending=*/false,
                                             IREE_DURATION_ZERO));

  iree_io_uring_ring_cq_advance(&ring_, 1);
}

TEST_F(RingTest, WaitCqeTimesOutWhenEmpty) {
  // No submissions, CQ is empty.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_io_uring_ring_wait_cqe(&ring_, /*min_complete=*/1,
                                  /*flush_pending=*/false,
                                  /*timeout_ns=*/1000000));  // 1ms
}

}  // namespace
