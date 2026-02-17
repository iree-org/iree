// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/operation_pool.h"

#include <set>
#include <thread>
#include <vector>

#include "iree/async/operation.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class OperationPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_operation_pool_options_t options =
        iree_async_operation_pool_options_default();
    IREE_ASSERT_OK(iree_async_operation_pool_allocate(
        options, iree_allocator_system(), &pool_));
  }

  void TearDown() override {
    iree_async_operation_pool_free(pool_);
    pool_ = nullptr;
  }

  iree_async_operation_pool_t* pool_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Basic acquire/release tests
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, AcquireReleaseBasic) {
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool_, sizeof(iree_async_operation_t), &operation));
  ASSERT_NE(operation, nullptr);

  // Verify operation is zeroed.
  EXPECT_EQ(operation->next, nullptr);
  EXPECT_EQ(operation->type, 0);
  EXPECT_EQ(operation->flags, 0u);
  EXPECT_EQ(operation->completion_fn, nullptr);
  EXPECT_EQ(operation->user_data, nullptr);
  EXPECT_EQ(operation->pool, nullptr);

  // Release back to pool.
  iree_async_operation_pool_release(pool_, operation);
}

TEST_F(OperationPoolTest, AcquireReturnsZeroedMemory) {
  // Acquire, write some data, release.
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool_, sizeof(iree_async_operation_t), &operation));
  ASSERT_NE(operation, nullptr);

  operation->type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  operation->flags = 0xDEADBEEF;
  operation->user_data = (void*)0x12345678;

  iree_async_operation_pool_release(pool_, operation);

  // Acquire again - should get same slot back (LIFO) and it should be zeroed.
  iree_async_operation_t* operation2 = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool_, sizeof(iree_async_operation_t), &operation2));
  ASSERT_NE(operation2, nullptr);

  // Same slot should be returned (LIFO freelist).
  EXPECT_EQ(operation, operation2);

  // Fields should be zeroed.
  EXPECT_EQ(operation2->type, 0);
  EXPECT_EQ(operation2->flags, 0u);
  EXPECT_EQ(operation2->user_data, nullptr);

  iree_async_operation_pool_release(pool_, operation2);
}

//===----------------------------------------------------------------------===//
// Size class tests
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, SmallestSizeClass) {
  // Request very small size - should get minimum size class (64 bytes).
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(pool_, 1, &operation));
  ASSERT_NE(operation, nullptr);
  iree_async_operation_pool_release(pool_, operation);
}

TEST_F(OperationPoolTest, VariousSizeClasses) {
  // Acquire operations of various sizes to exercise different size classes.
  std::vector<iree_async_operation_t*> operations;

  // Test sizes that map to different size classes.
  // Size classes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
  // Subtract 8 for header to get usable size.
  iree_host_size_t test_sizes[] = {
      16,    // -> 64 byte class
      64,    // -> 128 byte class (64 + 8 header > 64)
      120,   // -> 128 byte class
      200,   // -> 256 byte class
      500,   // -> 512 byte class
      1000,  // -> 1024 byte class
      2000,  // -> 2048 byte class
      4000,  // -> 4096 byte class
      8000,  // -> 8192 byte class
  };

  for (iree_host_size_t size : test_sizes) {
    iree_async_operation_t* operation = nullptr;
    IREE_ASSERT_OK(iree_async_operation_pool_acquire(pool_, size, &operation));
    ASSERT_NE(operation, nullptr);
    operations.push_back(operation);
  }

  // Release all.
  for (auto* op : operations) {
    iree_async_operation_pool_release(pool_, op);
  }
}

TEST_F(OperationPoolTest, MaxPooledSize) {
  // Request exactly at the max pooled size boundary (16KB - 8 byte header).
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(
      iree_async_operation_pool_acquire(pool_, 16384 - 8, &operation));
  ASSERT_NE(operation, nullptr);
  iree_async_operation_pool_release(pool_, operation);
}

//===----------------------------------------------------------------------===//
// Oversized allocation tests
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, OversizedAllocation) {
  // Request larger than max pooled size - should fall back to direct alloc.
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(
      iree_async_operation_pool_acquire(pool_, 32 * 1024, &operation));
  ASSERT_NE(operation, nullptr);

  // Verify it's zeroed.
  EXPECT_EQ(operation->next, nullptr);

  // Release - should free directly.
  iree_async_operation_pool_release(pool_, operation);
}

TEST_F(OperationPoolTest, OversizedDoesNotReuse) {
  // Oversized allocations don't go to freelist, so each acquire is fresh.
  iree_async_operation_t* op1 = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(pool_, 32 * 1024, &op1));

  iree_async_operation_t* op2 = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(pool_, 32 * 1024, &op2));

  // Should be different allocations.
  EXPECT_NE(op1, op2);

  iree_async_operation_pool_release(pool_, op1);
  iree_async_operation_pool_release(pool_, op2);
}

//===----------------------------------------------------------------------===//
// Pool growth tests
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, GrowsOnDemand) {
  // Acquire many operations to force multiple block allocations.
  std::vector<iree_async_operation_t*> operations;
  const int count = 1000;

  for (int i = 0; i < count; ++i) {
    iree_async_operation_t* operation = nullptr;
    IREE_ASSERT_OK(iree_async_operation_pool_acquire(
        pool_, sizeof(iree_async_operation_t), &operation));
    ASSERT_NE(operation, nullptr);
    operations.push_back(operation);
  }

  // All should be unique.
  std::set<iree_async_operation_t*> unique_ops(operations.begin(),
                                               operations.end());
  EXPECT_EQ(unique_ops.size(), static_cast<size_t>(count));

  // Release all.
  for (auto* op : operations) {
    iree_async_operation_pool_release(pool_, op);
  }
}

TEST_F(OperationPoolTest, ReusesReleasedSlots) {
  // Acquire and release repeatedly - should reuse slots.
  iree_async_operation_t* first_operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool_, sizeof(iree_async_operation_t), &first_operation));
  iree_async_operation_pool_release(pool_, first_operation);

  // Acquire again - should get same slot (LIFO).
  iree_async_operation_t* second_operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool_, sizeof(iree_async_operation_t), &second_operation));
  EXPECT_EQ(first_operation, second_operation);
  iree_async_operation_pool_release(pool_, second_operation);
}

//===----------------------------------------------------------------------===//
// Thread safety tests
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, ConcurrentAcquireRelease) {
  const int thread_count = 4;
  const int ops_per_thread = 100;

  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([this, ops_per_thread]() {
      for (int i = 0; i < ops_per_thread; ++i) {
        iree_async_operation_t* operation = nullptr;
        iree_status_t status = iree_async_operation_pool_acquire(
            pool_, sizeof(iree_async_operation_t), &operation);
        ASSERT_TRUE(iree_status_is_ok(status));
        ASSERT_NE(operation, nullptr);

        // Simulate some work.
        operation->type = IREE_ASYNC_OPERATION_TYPE_NOP;

        iree_async_operation_pool_release(pool_, operation);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(OperationPoolTest, ConcurrentMixedSizes) {
  const int thread_count = 4;
  const int ops_per_thread = 50;

  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([this, t, ops_per_thread]() {
      // Each thread uses different size classes.
      iree_host_size_t sizes[] = {64, 256, 1024, 4096};
      iree_host_size_t size = sizes[t % 4];

      for (int i = 0; i < ops_per_thread; ++i) {
        iree_async_operation_t* operation = nullptr;
        iree_status_t status =
            iree_async_operation_pool_acquire(pool_, size, &operation);
        ASSERT_TRUE(iree_status_is_ok(status));
        ASSERT_NE(operation, nullptr);

        iree_async_operation_pool_release(pool_, operation);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

//===----------------------------------------------------------------------===//
// Options tests
//===----------------------------------------------------------------------===//

TEST(OperationPoolOptionsTest, DefaultOptions) {
  iree_async_operation_pool_options_t options =
      iree_async_operation_pool_options_default();
  EXPECT_EQ(options.block_size, 0u);
  EXPECT_EQ(options.max_pooled_size, 0u);
}

TEST(OperationPoolOptionsTest, CustomBlockSize) {
  iree_async_operation_pool_options_t options =
      iree_async_operation_pool_options_default();
  options.block_size = 128 * 1024;  // 128KB blocks.

  iree_async_operation_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_allocate(
      options, iree_allocator_system(), &pool));

  // Should still work normally.
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool, sizeof(iree_async_operation_t), &operation));
  ASSERT_NE(operation, nullptr);
  iree_async_operation_pool_release(pool, operation);

  iree_async_operation_pool_free(pool);
}

TEST(OperationPoolOptionsTest, SmallBlockSize) {
  iree_async_operation_pool_options_t options =
      iree_async_operation_pool_options_default();
  options.block_size = 4 * 1024;  // Minimum block size.

  iree_async_operation_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_allocate(
      options, iree_allocator_system(), &pool));

  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool, sizeof(iree_async_operation_t), &operation));
  ASSERT_NE(operation, nullptr);
  iree_async_operation_pool_release(pool, operation);

  iree_async_operation_pool_free(pool);
}

TEST(OperationPoolOptionsTest, CustomMaxPooledSize) {
  iree_async_operation_pool_options_t options =
      iree_async_operation_pool_options_default();
  options.max_pooled_size = 1024;  // Only pool up to 1KB.

  iree_async_operation_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_allocate(
      options, iree_allocator_system(), &pool));

  // Small allocation should be pooled.
  iree_async_operation_t* small_op = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(pool, 64, &small_op));
  iree_async_operation_pool_release(pool, small_op);

  // Large allocation should be direct (> max_pooled_size - 8 header).
  iree_async_operation_t* large_op = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(pool, 2048, &large_op));
  iree_async_operation_pool_release(pool, large_op);

  iree_async_operation_pool_free(pool);
}

//===----------------------------------------------------------------------===//
// Trim tests
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, TrimIsNoOp) {
  // Trim is currently unimplemented - just verify it doesn't crash.
  iree_host_size_t trimmed = iree_async_operation_pool_trim(pool_, 100);
  EXPECT_EQ(trimmed, 0u);
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(OperationPoolTest, ReleaseNull) {
  // Should not crash.
  iree_async_operation_pool_release(pool_, nullptr);
}

TEST_F(OperationPoolTest, ReleaseWithNullPool) {
  iree_async_operation_t* operation = nullptr;
  IREE_ASSERT_OK(iree_async_operation_pool_acquire(
      pool_, sizeof(iree_async_operation_t), &operation));

  // Should not crash (no-op).
  iree_async_operation_pool_release(nullptr, operation);

  // Clean up properly.
  iree_async_operation_pool_release(pool_, operation);
}

TEST_F(OperationPoolTest, FreeNullPool) {
  // Should not crash.
  iree_async_operation_pool_free(nullptr);
}

}  // namespace
