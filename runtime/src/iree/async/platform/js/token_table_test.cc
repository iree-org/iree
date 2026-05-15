// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/js/token_table.h"

#include <cstring>

#include "iree/async/operation.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TokenTableTest : public ::testing::Test {
 protected:
  static constexpr iree_host_size_t kDefaultCapacity = 4;

  void SetUp() override {
    IREE_ASSERT_OK(iree_async_js_token_table_initialize(
        kDefaultCapacity, iree_allocator_system(), &table_));
  }

  void TearDown() override { iree_async_js_token_table_deinitialize(&table_); }

  void InitOperation(iree_async_operation_t* operation) {
    memset(operation, 0, sizeof(*operation));
    operation->type = IREE_ASYNC_OPERATION_TYPE_NOP;
  }

  iree_async_js_token_table_t table_;
};

TEST_F(TokenTableTest, InitializeDeinitialize) {
  EXPECT_TRUE(iree_async_js_token_table_is_empty(&table_));
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 0);
}

TEST_F(TokenTableTest, AcquireReturnsSequentialTokens) {
  iree_async_operation_t operations[4];
  for (int i = 0; i < 4; ++i) {
    InitOperation(&operations[i]);
  }

  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t token = UINT32_MAX;
    IREE_ASSERT_OK(
        iree_async_js_token_table_acquire(&table_, &operations[i], &token));
    EXPECT_EQ(token, i);
    EXPECT_EQ(iree_async_js_token_table_count(&table_), i + 1);
  }

  // Clean up for TearDown.
  for (uint32_t i = 0; i < 4; ++i) {
    iree_async_js_token_table_release(&table_, i);
  }
}

TEST_F(TokenTableTest, LookupReturnsCorrectOperation) {
  iree_async_operation_t operation;
  InitOperation(&operation);

  uint32_t token = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operation, &token));

  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token), &operation);

  iree_async_js_token_table_release(&table_, token);
}

TEST_F(TokenTableTest, LookupFreeSlotReturnsNull) {
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, 0), nullptr);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, 1), nullptr);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, 3), nullptr);
}

TEST_F(TokenTableTest, ReleaseSlot) {
  iree_async_operation_t operation;
  InitOperation(&operation);

  uint32_t token = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operation, &token));
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 1);

  iree_async_js_token_table_release(&table_, token);
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 0);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token), nullptr);
  EXPECT_TRUE(iree_async_js_token_table_is_empty(&table_));
}

TEST_F(TokenTableTest, TableFullReturnsResourceExhausted) {
  iree_async_operation_t operations[5];
  for (int i = 0; i < 5; ++i) {
    InitOperation(&operations[i]);
  }

  // Fill all 4 slots.
  uint32_t tokens[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        iree_async_js_token_table_acquire(&table_, &operations[i], &tokens[i]));
  }

  // 5th acquire fails.
  uint32_t overflow_token = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_js_token_table_acquire(
                            &table_, &operations[4], &overflow_token));

  // Clean up.
  for (int i = 0; i < 4; ++i) {
    iree_async_js_token_table_release(&table_, tokens[i]);
  }
}

TEST_F(TokenTableTest, AcquireAfterRelease) {
  iree_async_operation_t operations[5];
  for (int i = 0; i < 5; ++i) {
    InitOperation(&operations[i]);
  }

  // Fill all 4 slots (tokens 0-3).
  uint32_t tokens[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        iree_async_js_token_table_acquire(&table_, &operations[i], &tokens[i]));
  }

  // Free token 0 (slot 0 is now free).
  iree_async_js_token_table_release(&table_, tokens[0]);
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 3);

  // Acquire again: next_token=4, index=4%4=0, slot 0 is free → succeeds.
  uint32_t new_token = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operations[4], &new_token));
  EXPECT_EQ(new_token, 4u);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, new_token),
            &operations[4]);
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 4);

  // Clean up.
  for (int i = 1; i < 4; ++i) {
    iree_async_js_token_table_release(&table_, tokens[i]);
  }
  iree_async_js_token_table_release(&table_, new_token);
}

TEST_F(TokenTableTest, AcquireCollisionReturnsResourceExhausted) {
  iree_async_operation_t operations[5];
  for (int i = 0; i < 5; ++i) {
    InitOperation(&operations[i]);
  }

  // Fill all 4 slots (tokens 0-3).
  uint32_t tokens[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        iree_async_js_token_table_acquire(&table_, &operations[i], &tokens[i]));
  }

  // Free token 1 (slot 1 is free, count=3).
  iree_async_js_token_table_release(&table_, tokens[1]);

  // Acquire: next_token=4, index=4%4=0, slot 0 is OCCUPIED.
  // Returns RESOURCE_EXHAUSTED despite count(3) < capacity(4).
  uint32_t collision_token = UINT32_MAX;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_js_token_table_acquire(
                            &table_, &operations[4], &collision_token));

  // Clean up (slots 0, 2, 3 are occupied).
  iree_async_js_token_table_release(&table_, tokens[0]);
  iree_async_js_token_table_release(&table_, tokens[2]);
  iree_async_js_token_table_release(&table_, tokens[3]);
}

TEST_F(TokenTableTest, InterleavedAcquireAndRelease) {
  iree_async_operation_t operations[4];
  for (int i = 0; i < 4; ++i) {
    InitOperation(&operations[i]);
  }

  // Allocate token 0.
  uint32_t token0 = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operations[0], &token0));
  EXPECT_EQ(token0, 0u);

  // Allocate token 1.
  uint32_t token1 = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operations[1], &token1));
  EXPECT_EQ(token1, 1u);

  // Free token 0.
  iree_async_js_token_table_release(&table_, token0);
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 1);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token0), nullptr);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token1), &operations[1]);

  // Allocate token 2.
  uint32_t token2 = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operations[2], &token2));
  EXPECT_EQ(token2, 2u);
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 2);

  // Free token 1.
  iree_async_js_token_table_release(&table_, token1);

  // Allocate token 3.
  uint32_t token3 = UINT32_MAX;
  IREE_ASSERT_OK(
      iree_async_js_token_table_acquire(&table_, &operations[3], &token3));
  EXPECT_EQ(token3, 3u);

  // Verify final state.
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token0), nullptr);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token1), nullptr);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token2), &operations[2]);
  EXPECT_EQ(iree_async_js_token_table_lookup(&table_, token3), &operations[3]);
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 2);

  // Clean up.
  iree_async_js_token_table_release(&table_, token2);
  iree_async_js_token_table_release(&table_, token3);
}

TEST_F(TokenTableTest, CompletionEntryLayout) {
  EXPECT_EQ(sizeof(iree_async_js_completion_entry_t), 8);
  EXPECT_EQ(offsetof(iree_async_js_completion_entry_t, token), 0);
  EXPECT_EQ(offsetof(iree_async_js_completion_entry_t, status_code), 4);
}

TEST_F(TokenTableTest, LargerCapacity) {
  // Use a separate table with larger capacity.
  iree_async_js_token_table_deinitialize(&table_);

  static constexpr iree_host_size_t kLargeCapacity = 256;
  IREE_ASSERT_OK(iree_async_js_token_table_initialize(
      kLargeCapacity, iree_allocator_system(), &table_));

  iree_async_operation_t operations[16];
  uint32_t tokens[16];
  for (int i = 0; i < 16; ++i) {
    InitOperation(&operations[i]);
    IREE_ASSERT_OK(
        iree_async_js_token_table_acquire(&table_, &operations[i], &tokens[i]));
  }
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 16);

  // Free even-numbered tokens.
  for (int i = 0; i < 16; i += 2) {
    iree_async_js_token_table_release(&table_, tokens[i]);
  }
  EXPECT_EQ(iree_async_js_token_table_count(&table_), 8);

  // Verify odd-numbered tokens still resolve.
  for (int i = 1; i < 16; i += 2) {
    EXPECT_EQ(iree_async_js_token_table_lookup(&table_, tokens[i]),
              &operations[i]);
  }

  // Clean up remaining.
  for (int i = 1; i < 16; i += 2) {
    iree_async_js_token_table_release(&table_, tokens[i]);
  }
}

TEST_F(TokenTableTest, ZeroCapacityReturnsInvalidArgument) {
  // Use a separate table.
  iree_async_js_token_table_t zero_table;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_js_token_table_initialize(
                            0, iree_allocator_system(), &zero_table));
}

}  // namespace
