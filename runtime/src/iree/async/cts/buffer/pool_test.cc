// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for buffer pool operations.
//
// Tests buffer pool allocation, acquire/release semantics, and backend
// integration. These tests verify the buffer pool API works correctly across
// all proactor backends.

#include <cstring>

#include "iree/async/buffer_pool.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/slab.h"

namespace iree::async::cts {

class BufferPoolTest : public CtsTestBase<> {
 protected:
  // Helper to create slab + register + pool in one step.
  // On success, caller must free pool, release region, and release slab.
  struct PoolSetup {
    iree_async_slab_t* slab = nullptr;
    iree_async_region_t* region = nullptr;
    iree_async_buffer_pool_t* pool = nullptr;
  };

  iree_status_t SetupPool(iree_host_size_t buffer_size,
                          iree_host_size_t buffer_count,
                          iree_async_buffer_access_flags_t access_flags,
                          PoolSetup* out_setup) {
    out_setup->slab = nullptr;
    out_setup->region = nullptr;
    out_setup->pool = nullptr;

    // Create slab.
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = buffer_size;
    slab_options.buffer_count = buffer_count;
    IREE_RETURN_IF_ERROR(iree_async_slab_create(
        slab_options, iree_allocator_system(), &out_setup->slab));

    // Register slab with proactor.
    iree_status_t status = iree_async_proactor_register_slab(
        proactor_, out_setup->slab, access_flags, &out_setup->region);
    if (!iree_status_is_ok(status)) {
      iree_async_slab_release(out_setup->slab);
      out_setup->slab = nullptr;
      return status;
    }

    // Create pool over region.
    status = iree_async_buffer_pool_allocate(
        out_setup->region, iree_allocator_system(), &out_setup->pool);
    if (!iree_status_is_ok(status)) {
      iree_async_region_release(out_setup->region);
      iree_async_slab_release(out_setup->slab);
      out_setup->region = nullptr;
      out_setup->slab = nullptr;
      return status;
    }

    return iree_ok_status();
  }

  void TeardownPool(PoolSetup* setup) {
    iree_async_buffer_pool_free(setup->pool);
    setup->pool = nullptr;
    iree_async_region_release(setup->region);
    setup->region = nullptr;
    iree_async_slab_release(setup->slab);
    setup->slab = nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

// Basic create/free cycle.
TEST_P(BufferPoolTest, CreateAndFree) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/4096, /*buffer_count=*/16,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  // Verify initial state.
  EXPECT_EQ(iree_async_buffer_pool_capacity(setup.pool), 16u);
  EXPECT_EQ(iree_async_buffer_pool_available(setup.pool), 16u);
  EXPECT_EQ(iree_async_buffer_pool_buffer_size(setup.pool), 4096u);

  TeardownPool(&setup);
}

// Free with NULL is a no-op.
TEST_P(BufferPoolTest, FreeNullIsNoOp) {
  iree_async_buffer_pool_free(nullptr);  // Should not crash.
}

// Zero buffer_size slab is rejected.
TEST_P(BufferPoolTest, ZeroBufferSizeRejected) {
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 0;  // Invalid.
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
  EXPECT_EQ(slab, nullptr);
}

// Zero buffer_count slab is rejected.
TEST_P(BufferPoolTest, ZeroBufferCountRejected) {
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 4096;
  slab_options.buffer_count = 0;  // Invalid.

  iree_async_slab_t* slab = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));
  EXPECT_EQ(slab, nullptr);
}

//===----------------------------------------------------------------------===//
// Acquire / Release tests
//===----------------------------------------------------------------------===//

// Acquire a single buffer and release it.
TEST_P(BufferPoolTest, AcquireAndRelease) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/1024, /*buffer_count=*/4,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  EXPECT_EQ(iree_async_buffer_pool_available(setup.pool), 4u);

  // Acquire a buffer.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &lease));

  EXPECT_EQ(iree_async_buffer_pool_available(setup.pool), 3u);
  EXPECT_NE(lease.release.fn, nullptr);
  EXPECT_EQ(lease.span.length, 1024u);
  EXPECT_NE(iree_async_span_ptr(lease.span), nullptr);

  // Release the buffer.
  iree_async_buffer_lease_release(&lease);

  EXPECT_EQ(iree_async_buffer_pool_available(setup.pool), 4u);

  TeardownPool(&setup);
}

// Acquire all buffers, then release all.
TEST_P(BufferPoolTest, AcquireAllAndReleaseAll) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/512, /*buffer_count=*/8,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  // Acquire all buffers.
  iree_async_buffer_lease_t leases[8];
  for (int i = 0; i < 8; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &leases[i]));
    EXPECT_EQ(iree_async_buffer_pool_available(setup.pool),
              (iree_host_size_t)(7 - i));
  }

  // All buffers should be leased.
  EXPECT_EQ(iree_async_buffer_pool_available(setup.pool), 0u);

  // Release all in reverse order.
  for (int i = 7; i >= 0; --i) {
    iree_async_buffer_lease_release(&leases[i]);
    EXPECT_EQ(iree_async_buffer_pool_available(setup.pool),
              (iree_host_size_t)(8 - i));
  }

  TeardownPool(&setup);
}

// Resource exhaustion returns RESOURCE_EXHAUSTED.
TEST_P(BufferPoolTest, ResourceExhaustion) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/256, /*buffer_count=*/2,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  // Acquire both buffers.
  iree_async_buffer_lease_t lease1, lease2;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &lease1));
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &lease2));

  EXPECT_EQ(iree_async_buffer_pool_available(setup.pool), 0u);

  // Third acquire should fail.
  iree_async_buffer_lease_t lease3;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_buffer_pool_acquire(setup.pool, &lease3));

  // Release one buffer.
  iree_async_buffer_lease_release(&lease1);

  // Now acquire should succeed.
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &lease3));

  // Cleanup.
  iree_async_buffer_lease_release(&lease2);
  iree_async_buffer_lease_release(&lease3);

  TeardownPool(&setup);
}

//===----------------------------------------------------------------------===//
// Buffer content tests
//===----------------------------------------------------------------------===//

// Verify that buffer memory is writable and readable.
TEST_P(BufferPoolTest, BufferMemoryAccessible) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/128, /*buffer_count=*/2,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &lease));

  // Write a pattern.
  uint8_t* ptr = iree_async_span_ptr(lease.span);
  const uint8_t pattern[] = {0xDE, 0xAD, 0xBE, 0xEF};
  memcpy(ptr, pattern, sizeof(pattern));

  // Verify read back.
  EXPECT_EQ(memcmp(ptr, pattern, sizeof(pattern)), 0);

  iree_async_buffer_lease_release(&lease);
  TeardownPool(&setup);
}

// Verify each buffer has a unique, non-overlapping memory region.
TEST_P(BufferPoolTest, BuffersAreDisjoint) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/64, /*buffer_count=*/4,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  iree_async_buffer_lease_t leases[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &leases[i]));
  }

  // Check all pairs for overlap.
  for (int i = 0; i < 4; ++i) {
    uint8_t* ptr_i = iree_async_span_ptr(leases[i].span);
    for (int j = i + 1; j < 4; ++j) {
      uint8_t* ptr_j = iree_async_span_ptr(leases[j].span);
      // Buffers should not overlap.
      EXPECT_TRUE(ptr_i + 64 <= ptr_j || ptr_j + 64 <= ptr_i)
          << "Buffer " << i << " and " << j << " overlap";
    }
  }

  for (int i = 0; i < 4; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }
  TeardownPool(&setup);
}

//===----------------------------------------------------------------------===//
// Span and region tests
//===----------------------------------------------------------------------===//

// Verify lease span references a valid region.
TEST_P(BufferPoolTest, LeaseHasValidRegion) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/256, /*buffer_count=*/2,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &lease));

  // Span should have a valid region for zero-copy I/O.
  EXPECT_NE(lease.span.region, nullptr);
  EXPECT_EQ(lease.span.length, 256u);

  // Region base_ptr should match span pointer calculation.
  uint8_t* expected_ptr =
      (uint8_t*)lease.span.region->base_ptr + lease.span.offset;
  EXPECT_EQ(iree_async_span_ptr(lease.span), expected_ptr);

  iree_async_buffer_lease_release(&lease);
  TeardownPool(&setup);
}

// Verify buffer indices are within bounds.
TEST_P(BufferPoolTest, LeaseIndicesValid) {
  PoolSetup setup;
  IREE_ASSERT_OK(SetupPool(/*buffer_size=*/128, /*buffer_count=*/8,
                           IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &setup));

  iree_async_buffer_lease_t leases[8];
  for (int i = 0; i < 8; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(setup.pool, &leases[i]));
    // Buffer index should be within [0, buffer_count).
    EXPECT_LT(leases[i].buffer_index, 8u);
  }

  for (int i = 0; i < 8; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }
  TeardownPool(&setup);
}

//===----------------------------------------------------------------------===//
// Power-of-two requirement (io_uring specific, but validated at pool level)
//===----------------------------------------------------------------------===//

// For io_uring, buffer_count must be power of 2. Other backends may accept
// any positive value. This test verifies the backend correctly handles or
// rejects non-power-of-2 counts.
TEST_P(BufferPoolTest, NonPowerOfTwoCount) {
  // First create a slab with non-power-of-2 count.
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 256;
  slab_options.buffer_count = 7;  // Not a power of 2.

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  // Registering may fail for backends that require power-of-2.
  iree_async_region_t* region = nullptr;
  iree_status_t status = iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region);

  if (iree_status_is_ok(status)) {
    // Backend accepted it. Create pool and verify it works.
    iree_async_buffer_pool_t* pool = nullptr;
    IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
        region, iree_allocator_system(), &pool));
    EXPECT_EQ(iree_async_buffer_pool_capacity(pool), 7u);
    iree_async_buffer_pool_free(pool);
    iree_async_region_release(region);
  } else {
    // Backend rejected non-power-of-2 (io_uring behavior).
    EXPECT_TRUE(iree_status_is_invalid_argument(status) ||
                iree_status_is_unavailable(status));
    iree_status_ignore(status);
  }

  iree_async_slab_release(slab);
}

//===----------------------------------------------------------------------===//
// Slab lifecycle tests
//===----------------------------------------------------------------------===//

// Verify slab source is set correctly for allocate.
TEST_P(BufferPoolTest, SlabSourceIsNumaAlloc) {
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 4096;
  slab_options.buffer_count = 8;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  EXPECT_EQ(iree_async_slab_source(slab), IREE_ASYNC_SLAB_SOURCE_NUMA_ALLOC);
  EXPECT_EQ(iree_async_slab_buffer_size(slab), 4096u);
  EXPECT_EQ(iree_async_slab_buffer_count(slab), 8u);
  EXPECT_EQ(iree_async_slab_total_size(slab), 4096u * 8);

  iree_async_slab_release(slab);
}

// Verify slab wrap works correctly.
TEST_P(BufferPoolTest, SlabWrapExternalMemory) {
  // Allocate external memory.
  const iree_host_size_t buffer_size = 512;
  const iree_host_size_t buffer_count = 4;
  void* external_memory = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), buffer_size * buffer_count, &external_memory));

  // Wrap it as a slab.
  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(iree_async_slab_wrap(external_memory, buffer_size,
                                      buffer_count, iree_allocator_system(),
                                      &slab));

  EXPECT_EQ(iree_async_slab_source(slab), IREE_ASYNC_SLAB_SOURCE_WRAPPED);
  EXPECT_EQ(iree_async_slab_base_ptr(slab), external_memory);

  // Register and use.
  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &lease));

  // Verify pointer is within external memory.
  uint8_t* ptr = iree_async_span_ptr(lease.span);
  EXPECT_GE(ptr, (uint8_t*)external_memory);
  EXPECT_LT(ptr, (uint8_t*)external_memory + buffer_size * buffer_count);

  iree_async_buffer_lease_release(&lease);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);

  // External memory must outlive slab - free it now.
  iree_allocator_free(iree_allocator_system(), external_memory);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(BufferPoolTest);

}  // namespace iree::async::cts
