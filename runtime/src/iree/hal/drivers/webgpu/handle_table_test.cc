// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/handle_table.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Sentinel objects used as distinguishable non-NULL pointers.
// Each has a unique address — we never dereference them.
static int kObjectA;
static int kObjectB;
static int kObjectC;
static int kObjectD;

class HandleTableTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kDefaultCapacity = 8;

  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_webgpu_handle_table_initialize(
        kDefaultCapacity, iree_allocator_system(), &table_));
  }

  void TearDown() override {
    iree_hal_webgpu_handle_table_deinitialize(&table_);
  }

  iree_hal_webgpu_handle_table_t table_;
};

TEST_F(HandleTableTest, InitializeDeinitialize) {
  EXPECT_TRUE(iree_hal_webgpu_handle_table_is_empty(&table_));
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 0u);
}

TEST_F(HandleTableTest, InsertReturnsNonZeroHandles) {
  iree_hal_webgpu_handle_t handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectA, &handle));
  EXPECT_NE(handle, IREE_HAL_WEBGPU_HANDLE_NULL);
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 1u);

  iree_hal_webgpu_handle_table_remove(&table_, handle);
}

TEST_F(HandleTableTest, InsertReturnsSequentialHandles) {
  iree_hal_webgpu_handle_t handles[3];
  void* objects[] = {&kObjectA, &kObjectB, &kObjectC};
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(
        iree_hal_webgpu_handle_table_insert(&table_, objects[i], &handles[i]));
  }
  // Handles should be sequential starting at 1 (0 is reserved).
  EXPECT_EQ(handles[0], 1u);
  EXPECT_EQ(handles[1], 2u);
  EXPECT_EQ(handles[2], 3u);
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 3u);

  for (int i = 0; i < 3; ++i) {
    iree_hal_webgpu_handle_table_remove(&table_, handles[i]);
  }
}

TEST_F(HandleTableTest, GetReturnsCorrectObject) {
  iree_hal_webgpu_handle_t handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectA, &handle));
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handle), &kObjectA);

  iree_hal_webgpu_handle_table_remove(&table_, handle);
}

TEST_F(HandleTableTest, GetNullHandleReturnsNull) {
  EXPECT_EQ(
      iree_hal_webgpu_handle_table_get(&table_, IREE_HAL_WEBGPU_HANDLE_NULL),
      nullptr);
}

TEST_F(HandleTableTest, GetOutOfRangeReturnsNull) {
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, 999), nullptr);
}

TEST_F(HandleTableTest, GetUnoccupiedSlotReturnsNull) {
  // Slot 1 has never been used.
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, 1), nullptr);
}

TEST_F(HandleTableTest, RemoveReturnsObject) {
  iree_hal_webgpu_handle_t handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectA, &handle));

  void* removed = iree_hal_webgpu_handle_table_remove(&table_, handle);
  EXPECT_EQ(removed, &kObjectA);
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 0u);
  EXPECT_TRUE(iree_hal_webgpu_handle_table_is_empty(&table_));
}

TEST_F(HandleTableTest, GetAfterRemoveReturnsNull) {
  iree_hal_webgpu_handle_t handle = IREE_HAL_WEBGPU_HANDLE_NULL;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectA, &handle));
  iree_hal_webgpu_handle_table_remove(&table_, handle);
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handle), nullptr);
}

TEST_F(HandleTableTest, FreeListReuse) {
  // Insert three objects.
  iree_hal_webgpu_handle_t handle_a, handle_b, handle_c;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectA, &handle_a));
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectB, &handle_b));
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectC, &handle_c));

  // Remove B (middle).
  iree_hal_webgpu_handle_table_remove(&table_, handle_b);
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 2u);

  // Insert D — should reuse B's slot (LIFO free stack).
  iree_hal_webgpu_handle_t handle_d;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &kObjectD, &handle_d));
  EXPECT_EQ(handle_d, handle_b);  // Same index reused.
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handle_d), &kObjectD);
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 3u);

  // A and C are still there.
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handle_a), &kObjectA);
  EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handle_c), &kObjectC);

  iree_hal_webgpu_handle_table_remove(&table_, handle_a);
  iree_hal_webgpu_handle_table_remove(&table_, handle_c);
  iree_hal_webgpu_handle_table_remove(&table_, handle_d);
}

TEST_F(HandleTableTest, GrowsWhenFull) {
  // Default capacity is 8, so 7 usable slots (index 0 reserved).
  // Fill all 7 slots.
  iree_hal_webgpu_handle_t handles[8];
  int objects[8];
  for (int i = 0; i < 7; ++i) {
    IREE_ASSERT_OK(
        iree_hal_webgpu_handle_table_insert(&table_, &objects[i], &handles[i]));
  }
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 7u);

  // 8th insert triggers growth.
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&table_, &objects[7], &handles[7]));
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 8u);
  EXPECT_EQ(handles[7], 8u);  // First index in the expanded region.

  // All previous handles still valid.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handles[i]),
              &objects[i]);
  }

  for (int i = 0; i < 8; ++i) {
    iree_hal_webgpu_handle_table_remove(&table_, handles[i]);
  }
}

TEST_F(HandleTableTest, MultipleGrowCycles) {
  // Start with capacity 2 (1 usable slot). Grow repeatedly.
  iree_hal_webgpu_handle_table_deinitialize(&table_);
  IREE_ASSERT_OK(iree_hal_webgpu_handle_table_initialize(
      2, iree_allocator_system(), &table_));

  iree_hal_webgpu_handle_t handles[20];
  int objects[20];
  for (int i = 0; i < 20; ++i) {
    IREE_ASSERT_OK(
        iree_hal_webgpu_handle_table_insert(&table_, &objects[i], &handles[i]));
  }
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 20u);

  // All handles valid.
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, handles[i]),
              &objects[i]);
  }

  for (int i = 0; i < 20; ++i) {
    iree_hal_webgpu_handle_table_remove(&table_, handles[i]);
  }
}

TEST_F(HandleTableTest, InsertAfterRemoveAndGrow) {
  // Fill capacity, remove some, insert enough to trigger growth.
  int objects[13];

  // Fill 7 slots (capacity 8, index 0 reserved).
  iree_hal_webgpu_handle_t first_handles[7];
  for (int i = 0; i < 7; ++i) {
    IREE_ASSERT_OK(iree_hal_webgpu_handle_table_insert(&table_, &objects[i],
                                                       &first_handles[i]));
  }

  // Remove first 3 (these slots go on the free stack).
  for (int i = 0; i < 3; ++i) {
    iree_hal_webgpu_handle_table_remove(&table_, first_handles[i]);
  }
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 4u);

  // Insert 6 more: 3 reuse freed slots, then high_water fills to capacity
  // (triggering growth), then allocate from the expanded region.
  iree_hal_webgpu_handle_t second_handles[6];
  for (int i = 0; i < 6; ++i) {
    IREE_ASSERT_OK(iree_hal_webgpu_handle_table_insert(&table_, &objects[7 + i],
                                                       &second_handles[i]));
  }
  EXPECT_EQ(iree_hal_webgpu_handle_table_count(&table_), 10u);

  // Verify all handles from the first round that weren't removed are valid.
  for (int i = 3; i < 7; ++i) {
    EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, first_handles[i]),
              &objects[i])
        << "first round handle " << i;
  }

  // Verify all handles from the second round are valid.
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(iree_hal_webgpu_handle_table_get(&table_, second_handles[i]),
              &objects[7 + i])
        << "second round handle " << i;
  }

  // Clean up all live entries.
  for (int i = 3; i < 7; ++i) {
    iree_hal_webgpu_handle_table_remove(&table_, first_handles[i]);
  }
  for (int i = 0; i < 6; ++i) {
    iree_hal_webgpu_handle_table_remove(&table_, second_handles[i]);
  }
}

TEST_F(HandleTableTest, MinimumCapacityClamped) {
  // Request capacity 0 — should be clamped to 2.
  iree_hal_webgpu_handle_table_t small_table;
  IREE_ASSERT_OK(iree_hal_webgpu_handle_table_initialize(
      0, iree_allocator_system(), &small_table));

  // Should have room for at least 1 entry.
  iree_hal_webgpu_handle_t handle;
  IREE_ASSERT_OK(
      iree_hal_webgpu_handle_table_insert(&small_table, &kObjectA, &handle));
  EXPECT_NE(handle, IREE_HAL_WEBGPU_HANDLE_NULL);
  iree_hal_webgpu_handle_table_remove(&small_table, handle);
  iree_hal_webgpu_handle_table_deinitialize(&small_table);
}

TEST_F(HandleTableTest, NullHandleSentinel) {
  EXPECT_EQ(IREE_HAL_WEBGPU_HANDLE_NULL, 0u);
}

}  // namespace
