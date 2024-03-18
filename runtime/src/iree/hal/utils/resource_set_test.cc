// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/resource_set.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace {

using ::iree::testing::status::IsOkAndHolds;
using ::iree::testing::status::StatusIs;
using ::testing::Eq;

typedef struct iree_hal_test_resource_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  uint32_t index;
  uint32_t* live_bitmap;
} iree_hal_test_resource_t;

typedef struct iree_hal_test_resource_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_test_resource_t* resource);
} iree_hal_test_resource_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_test_resource_vtable_t);

extern const iree_hal_test_resource_vtable_t iree_hal_test_resource_vtable;

static iree_status_t iree_hal_test_resource_create(
    uint32_t index, uint32_t* live_bitmap, iree_allocator_t host_allocator,
    iree_hal_resource_t** out_resource) {
  iree_hal_test_resource_t* test_resource = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*test_resource), (void**)&test_resource));
  iree_hal_resource_initialize(&iree_hal_test_resource_vtable,
                               &test_resource->resource);
  test_resource->host_allocator = host_allocator;
  test_resource->index = index;
  test_resource->live_bitmap = live_bitmap;
  *live_bitmap |= 1 << index;
  *out_resource = (iree_hal_resource_t*)test_resource;
  return iree_ok_status();
}

static void iree_hal_test_resource_destroy(iree_hal_test_resource_t* resource) {
  iree_allocator_t host_allocator = resource->host_allocator;
  *resource->live_bitmap &= ~(1 << resource->index);
  iree_allocator_free(host_allocator, resource);
}

const iree_hal_test_resource_vtable_t iree_hal_test_resource_vtable = {
    /*.destroy=*/iree_hal_test_resource_destroy,
};

struct ResourceSetTest : public ::testing::Test {
  // We could check the allocator to ensure all memory is freed if we wanted to
  // reduce the reliance on asan.
  iree_allocator_t host_allocator = iree_allocator_default();
  iree_arena_block_pool_t block_pool;

  void SetUp() override {
    memset(&block_pool, 0, sizeof(block_pool));
    iree_arena_block_pool_initialize(128, host_allocator, &block_pool);
  }

  void TearDown() override {
    // This may assert (or at least trigger asan) if there are blocks
    // outstanding.
    iree_arena_block_pool_deinitialize(&block_pool);
  }
};

using resource_set_ptr = std::unique_ptr<iree_hal_resource_set_t,
                                         decltype(&iree_hal_resource_set_free)>;
static resource_set_ptr make_resource_set(iree_arena_block_pool_t* block_pool) {
  iree_hal_resource_set_t* set = NULL;
  IREE_CHECK_OK(iree_hal_resource_set_allocate(block_pool, &set));
  return resource_set_ptr(set, iree_hal_resource_set_free);
}

// Tests a set that has no resources added to it.
TEST_F(ResourceSetTest, Empty) {
  iree_hal_resource_set_t* set = NULL;
  IREE_ASSERT_OK(iree_hal_resource_set_allocate(&block_pool, &set));
  iree_hal_resource_set_free(set);
}

// Tests insertion of a single resource.
TEST_F(ResourceSetTest, Insert1) {
  auto resource_set = make_resource_set(&block_pool);

  // Create test resource; it'll set its bit in the live_bitmap.
  iree_hal_resource_t* resource = NULL;
  uint32_t live_bitmap = 0u;
  IREE_ASSERT_OK(iree_hal_test_resource_create(0, &live_bitmap, host_allocator,
                                               &resource));
  EXPECT_EQ(live_bitmap, 1u);

  // Insert the resource and drop the reference; it should still be live as the
  // set retains it.
  IREE_ASSERT_OK(
      iree_hal_resource_set_insert(resource_set.get(), 1, &resource));
  iree_hal_resource_release(resource);
  EXPECT_EQ(live_bitmap, 1u);

  // Drop the set and expect the resource to be destroyed as it loses its last
  // reference.
  resource_set.reset();
  EXPECT_EQ(live_bitmap, 0u);
}

// Tests inserting multiple resources at a time.
TEST_F(ResourceSetTest, Insert5) {
  auto resource_set = make_resource_set(&block_pool);

  // Allocate 5 resources - this lets us test for special paths that may handle
  // 4 at a time (to fit in SIMD registers) as well as the leftovers.
  iree_hal_resource_t* resources[5] = {NULL};
  uint32_t live_bitmap = 0u;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    IREE_ASSERT_OK(iree_hal_test_resource_create(
        i, &live_bitmap, host_allocator, &resources[i]));
  }
  EXPECT_EQ(live_bitmap, 0x1Fu);

  // Transfer ownership of the resources to the set.
  IREE_ASSERT_OK(iree_hal_resource_set_insert(
      resource_set.get(), IREE_ARRAYSIZE(resources), resources));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    iree_hal_resource_release(resources[i]);
  }
  EXPECT_EQ(live_bitmap, 0x1Fu);

  // Ensure the set releases the resources.
  resource_set.reset();
  EXPECT_EQ(live_bitmap, 0u);
}

// Tests inserting enough resources to force set growth. This is ensured by
// choosing a sufficiently small block size such that even 32 elements triggers
// a growth. Of course, real usage should have at least ~4KB for the block size.
TEST_F(ResourceSetTest, InsertionGrowth) {
  auto resource_set = make_resource_set(&block_pool);

  // Allocate 32 resources (one for each bit in our live map).
  iree_hal_resource_t* resources[32] = {NULL};
  uint32_t live_bitmap = 0u;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    IREE_ASSERT_OK(iree_hal_test_resource_create(
        i, &live_bitmap, host_allocator, &resources[i]));
  }
  EXPECT_EQ(live_bitmap, 0xFFFFFFFFu);

  // Transfer ownership of the resources to the set.
  IREE_ASSERT_OK(iree_hal_resource_set_insert(
      resource_set.get(), IREE_ARRAYSIZE(resources), resources));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    iree_hal_resource_release(resources[i]);
  }
  EXPECT_EQ(live_bitmap, 0xFFFFFFFFu);

  // Ensure the set releases the resources.
  resource_set.reset();
  EXPECT_EQ(live_bitmap, 0u);
}

// Tests insertion of resources multiple times to verify the MRU works.
TEST_F(ResourceSetTest, RedundantInsertion) {
  auto resource_set = make_resource_set(&block_pool);

  // Allocate 32 resources (one for each bit in our live map).
  // We want to be able to miss in the MRU.
  iree_hal_resource_t* resources[32] = {NULL};
  static_assert(IREE_ARRAYSIZE(resources) > IREE_HAL_RESOURCE_SET_MRU_SIZE,
                "need to pick a value that lets us exceed the MRU capacity");
  uint32_t live_bitmap = 0u;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    IREE_ASSERT_OK(iree_hal_test_resource_create(
        i, &live_bitmap, host_allocator, &resources[i]));
  }
  EXPECT_EQ(live_bitmap, 0xFFFFFFFFu);

  // NOTE: the only requirement of the MRU is that it's _mostly_ MRU - we may
  // for performance reasons make it a little fuzzy to avoid additional
  // shuffling. Today it's always a proper MRU and we check the pointers here.

  // NOTE: the MRU size can vary across architectures; we know it should always
  // be at least ~6 though so that's what we work with here.
  static_assert(IREE_HAL_RESOURCE_SET_MRU_SIZE > 6,
                "need at least enough elements to test with");

  // Insert in sequence, MRU should contain:
  //   31 30 29 28 27 ...
  IREE_ASSERT_OK(iree_hal_resource_set_insert(
      resource_set.get(), IREE_ARRAYSIZE(resources), resources));
  EXPECT_EQ(resource_set->mru[0], resources[31]);
  EXPECT_EQ(resource_set->mru[1], resources[30]);
  EXPECT_EQ(resource_set->mru[2], resources[29]);
  EXPECT_EQ(resource_set->mru[3], resources[28]);
  EXPECT_EQ(resource_set->mru[4], resources[27]);

  // Insert 31 again, MRU should remain the same as it's at the head.
  IREE_ASSERT_OK(
      iree_hal_resource_set_insert(resource_set.get(), 1, &resources[31]));
  EXPECT_EQ(resource_set->mru[0], resources[31]);
  EXPECT_EQ(resource_set->mru[1], resources[30]);
  EXPECT_EQ(resource_set->mru[2], resources[29]);
  EXPECT_EQ(resource_set->mru[3], resources[28]);
  EXPECT_EQ(resource_set->mru[4], resources[27]);

  // Insert 28 again, MRU should be updated to move it to the front:
  //   28 31 30 29 27 ...
  IREE_ASSERT_OK(
      iree_hal_resource_set_insert(resource_set.get(), 1, &resources[28]));
  EXPECT_EQ(resource_set->mru[0], resources[28]);
  EXPECT_EQ(resource_set->mru[1], resources[31]);
  EXPECT_EQ(resource_set->mru[2], resources[30]);
  EXPECT_EQ(resource_set->mru[3], resources[29]);
  EXPECT_EQ(resource_set->mru[4], resources[27]);

  // Insert 0 again, which should be a miss as it fell off the end of the MRU:
  //   0 28 31 30 29 27 ...
  IREE_ASSERT_OK(
      iree_hal_resource_set_insert(resource_set.get(), 1, &resources[0]));
  EXPECT_EQ(resource_set->mru[0], resources[0]);
  EXPECT_EQ(resource_set->mru[1], resources[28]);
  EXPECT_EQ(resource_set->mru[2], resources[31]);
  EXPECT_EQ(resource_set->mru[3], resources[30]);
  EXPECT_EQ(resource_set->mru[4], resources[29]);
  EXPECT_EQ(resource_set->mru[5], resources[27]);

  // Release all of the resources - they should still be owned by the set.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(resources); ++i) {
    iree_hal_resource_release(resources[i]);
  }
  EXPECT_EQ(live_bitmap, 0xFFFFFFFFu);

  // Ensure the set releases the resources.
  resource_set.reset();
  EXPECT_EQ(live_bitmap, 0u);
}

}  // namespace
}  // namespace hal
}  // namespace iree
