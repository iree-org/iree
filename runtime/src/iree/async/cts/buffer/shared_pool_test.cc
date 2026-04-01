// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for shared (cross-process) buffer pool operations.
//
// Tests the create_shared/open_shared pool lifecycle, cross-handle freelist
// visibility, and concurrent acquire/release across multiple pool handles
// backed by the same shared memory region.
//
// Cross-process access is simulated within a single process using SHM handle
// duplication (iree_shm_handle_dup + iree_shm_open_handle). The atomic
// freelist's CAS operations are identical within-process and cross-process —
// CPU cache coherency operates on physical addresses regardless of process
// boundaries — so this accurately tests the shared freelist semantics.

#include <atomic>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/slab.h"
#include "iree/base/internal/shm.h"

namespace iree::async::cts {

class SharedBufferPoolTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kBufferSize = 4096;
  static constexpr iree_host_size_t kBufferCount = 16;

  // Holds all resources for one side of a shared pool (creator or opener).
  struct SharedPoolSide {
    iree_shm_mapping_t shm = {};
    iree_async_slab_t* slab = nullptr;
    iree_async_region_t* region = nullptr;
    iree_async_buffer_pool_t* pool = nullptr;
  };

  // Creates a shared pool: allocates SHM, wraps the buffer portion as a slab,
  // registers with the proactor, and initializes the shared pool metadata.
  iree_status_t SetupCreator(iree_host_size_t buffer_size,
                             iree_host_size_t buffer_count,
                             SharedPoolSide* out_creator) {
    *out_creator = {};

    iree_host_size_t pool_storage = 0;
    IREE_RETURN_IF_ERROR(iree_async_buffer_pool_shared_storage_size(
        buffer_count, &pool_storage));
    iree_host_size_t total_shm_size = pool_storage + buffer_size * buffer_count;

    // Create anonymous shared memory region.
    IREE_RETURN_IF_ERROR(
        iree_shm_create(NULL, total_shm_size, &out_creator->shm));

    // Wrap the buffer data portion (after pool metadata) as a slab.
    void* buffer_base = (uint8_t*)out_creator->shm.base + pool_storage;
    IREE_RETURN_IF_ERROR(
        iree_async_slab_wrap(buffer_base, buffer_size, buffer_count,
                             iree_allocator_system(), &out_creator->slab));

    // Register slab with proactor.
    iree_status_t status = iree_async_proactor_register_slab(
        proactor_, out_creator->slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
        &out_creator->region);
    if (!iree_status_is_ok(status)) {
      iree_async_slab_release(out_creator->slab);
      out_creator->slab = nullptr;
      iree_shm_close(&out_creator->shm);
      return status;
    }

    // Initialize shared pool metadata in SHM.
    status = iree_async_buffer_pool_create_shared(
        out_creator->shm.base, pool_storage, out_creator->region,
        iree_allocator_system(), &out_creator->pool);
    if (!iree_status_is_ok(status)) {
      iree_async_region_release(out_creator->region);
      out_creator->region = nullptr;
      iree_async_slab_release(out_creator->slab);
      out_creator->slab = nullptr;
      iree_shm_close(&out_creator->shm);
      return status;
    }

    return iree_ok_status();
  }

  // Opens an existing shared pool by duplicating the creator's SHM handle
  // (simulating cross-process handle passing).
  //
  // In real cross-process usage, each process has its own proactor and
  // registers independently. Within our single-process test, some backends
  // (pre-5.19 io_uring) only allow one fixed buffer table per ring, so we
  // share the creator's region when a second registration would fail. This
  // is semantically correct: the region just provides backend handles for
  // the shared proactor, and both pool handles operate on the same physical
  // memory either way.
  iree_status_t SetupOpener(const SharedPoolSide& creator,
                            iree_host_size_t buffer_size,
                            iree_host_size_t buffer_count,
                            SharedPoolSide* out_opener) {
    *out_opener = {};

    iree_host_size_t pool_storage = 0;
    IREE_RETURN_IF_ERROR(iree_async_buffer_pool_shared_storage_size(
        buffer_count, &pool_storage));

    // Duplicate the SHM handle (simulates passing to another process).
    iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
    IREE_RETURN_IF_ERROR(iree_shm_handle_dup(creator.shm.handle, &dup_handle));

    // Open a second mapping from the duplicated handle.
    iree_status_t status =
        iree_shm_open_handle(dup_handle, creator.shm.size, &out_opener->shm);
    // open_handle dups internally; close our copy regardless of outcome.
    iree_shm_handle_close(&dup_handle);
    if (!iree_status_is_ok(status)) {
      return status;
    }

    // Try to register a separate slab with the proactor. If the backend
    // rejects it (e.g., singleton fixed buffer table), share the creator's
    // region instead.
    void* buffer_base = (uint8_t*)out_opener->shm.base + pool_storage;
    status = iree_async_slab_wrap(buffer_base, buffer_size, buffer_count,
                                  iree_allocator_system(), &out_opener->slab);
    if (!iree_status_is_ok(status)) {
      iree_shm_close(&out_opener->shm);
      return status;
    }

    status = iree_async_proactor_register_slab(
        proactor_, out_opener->slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
        &out_opener->region);
    if (iree_status_is_already_exists(status)) {
      // Singleton constraint: share the creator's region.
      iree_status_ignore(status);
      out_opener->region = creator.region;
      iree_async_region_retain(out_opener->region);
    } else if (!iree_status_is_ok(status)) {
      iree_async_slab_release(out_opener->slab);
      out_opener->slab = nullptr;
      iree_shm_close(&out_opener->shm);
      return status;
    }

    // Open the shared pool (binds to existing freelist, does not reinit).
    status = iree_async_buffer_pool_open_shared(
        out_opener->shm.base, pool_storage, out_opener->region,
        iree_allocator_system(), &out_opener->pool);
    if (!iree_status_is_ok(status)) {
      iree_async_region_release(out_opener->region);
      out_opener->region = nullptr;
      iree_async_slab_release(out_opener->slab);
      out_opener->slab = nullptr;
      iree_shm_close(&out_opener->shm);
      return status;
    }

    return iree_ok_status();
  }

  void TeardownSide(SharedPoolSide* side) {
    if (side->pool) {
      iree_async_buffer_pool_free(side->pool);
      side->pool = nullptr;
    }
    if (side->region) {
      iree_async_region_release(side->region);
      side->region = nullptr;
    }
    if (side->slab) {
      iree_async_slab_release(side->slab);
      side->slab = nullptr;
    }
    iree_shm_close(&side->shm);
  }
};

//===----------------------------------------------------------------------===//
// Storage size
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, StorageSizeMonotonic) {
  // Storage size must increase with buffer count.
  iree_host_size_t size_1 = 0, size_16 = 0, size_1024 = 0;
  IREE_ASSERT_OK(iree_async_buffer_pool_shared_storage_size(1, &size_1));
  IREE_ASSERT_OK(iree_async_buffer_pool_shared_storage_size(16, &size_16));
  IREE_ASSERT_OK(iree_async_buffer_pool_shared_storage_size(1024, &size_1024));
  EXPECT_GT(size_1, 0u);
  EXPECT_GT(size_16, size_1);
  EXPECT_GT(size_1024, size_16);
}

TEST_P(SharedBufferPoolTest, StorageSizePerSlotCost) {
  // Each additional buffer should add exactly sizeof(atomic_freelist_slot_t)
  // = 4 bytes.
  iree_host_size_t size_100 = 0, size_101 = 0;
  IREE_ASSERT_OK(iree_async_buffer_pool_shared_storage_size(100, &size_100));
  IREE_ASSERT_OK(iree_async_buffer_pool_shared_storage_size(101, &size_101));
  EXPECT_EQ(size_101 - size_100, sizeof(uint32_t));
}

//===----------------------------------------------------------------------===//
// Create and free
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, CreateAndFree) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  EXPECT_EQ(iree_async_buffer_pool_capacity(creator.pool), kBufferCount);
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), kBufferCount);
  EXPECT_EQ(iree_async_buffer_pool_buffer_size(creator.pool), kBufferSize);

  TeardownSide(&creator);
}

//===----------------------------------------------------------------------===//
// Open validation
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, OpenValidatesMagic) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  iree_host_size_t pool_storage = 0;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_shared_storage_size(kBufferCount, &pool_storage));

  // Corrupt the magic in a dup'd mapping of the same SHM region.
  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.shm.handle, &dup_handle));
  iree_shm_mapping_t corrupted;
  IREE_ASSERT_OK(
      iree_shm_open_handle(dup_handle, creator.shm.size, &corrupted));
  iree_shm_handle_close(&dup_handle);

  // Corrupt the magic field (first 4 bytes).
  uint32_t* magic_ptr = (uint32_t*)corrupted.base;
  uint32_t saved_magic = *magic_ptr;
  *magic_ptr = 0xDEADBEEF;

  // Use the creator's region — the open_shared check fails on magic before
  // it ever looks at the region's buffer geometry.
  iree_async_buffer_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_buffer_pool_open_shared(
                            corrupted.base, pool_storage, creator.region,
                            iree_allocator_system(), &pool));
  EXPECT_EQ(pool, nullptr);

  // Restore magic so creator teardown sees a valid header.
  *magic_ptr = saved_magic;

  iree_shm_close(&corrupted);
  TeardownSide(&creator);
}

TEST_P(SharedBufferPoolTest, OpenValidatesVersionMismatch) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  iree_host_size_t pool_storage = 0;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_shared_storage_size(kBufferCount, &pool_storage));

  // Corrupt the version in a dup'd mapping of the same SHM region.
  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.shm.handle, &dup_handle));
  iree_shm_mapping_t corrupted;
  IREE_ASSERT_OK(
      iree_shm_open_handle(dup_handle, creator.shm.size, &corrupted));
  iree_shm_handle_close(&dup_handle);

  // Corrupt the version field (bytes 4-7, immediately after magic).
  uint32_t* version_ptr = (uint32_t*)corrupted.base + 1;
  uint32_t saved_version = *version_ptr;
  *version_ptr = 0xFFFFFFFF;

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_buffer_pool_open_shared(
                            corrupted.base, pool_storage, creator.region,
                            iree_allocator_system(), &pool));
  EXPECT_EQ(pool, nullptr);

  // Restore version.
  *version_ptr = saved_version;

  iree_shm_close(&corrupted);
  TeardownSide(&creator);
}

TEST_P(SharedBufferPoolTest, OpenValidatesBufferSizeMismatch) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  iree_host_size_t pool_storage = 0;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_shared_storage_size(kBufferCount, &pool_storage));

  // Open a second mapping of the SHM region.
  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.shm.handle, &dup_handle));
  iree_shm_mapping_t opener_shm;
  IREE_ASSERT_OK(
      iree_shm_open_handle(dup_handle, creator.shm.size, &opener_shm));
  iree_shm_handle_close(&dup_handle);

  // Wrap with a different buffer_size to create a mismatched region.
  // Register with WRITE access to avoid the singleton fixed-buffer-table
  // constraint (READ uses fixed buffers, WRITE uses provided buffers).
  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(iree_async_slab_wrap((uint8_t*)opener_shm.base + pool_storage,
                                      /*buffer_size=*/512, kBufferCount,
                                      iree_allocator_system(), &slab));
  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));

  // open_shared should detect the buffer_size mismatch (header says 4096,
  // region says 512).
  iree_async_buffer_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_buffer_pool_open_shared(opener_shm.base, pool_storage, region,
                                         iree_allocator_system(), &pool));
  EXPECT_EQ(pool, nullptr);

  iree_async_region_release(region);
  iree_async_slab_release(slab);
  iree_shm_close(&opener_shm);
  TeardownSide(&creator);
}

TEST_P(SharedBufferPoolTest, OpenValidatesMemoryTooSmall) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  // Try to open with a shared_memory_size that's too small.
  iree_async_buffer_pool_t* pool = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_buffer_pool_open_shared(
                            creator.shm.base,
                            /*shared_memory_size=*/32,  // Way too small.
                            creator.region, iree_allocator_system(), &pool));
  EXPECT_EQ(pool, nullptr);

  TeardownSide(&creator);
}

//===----------------------------------------------------------------------===//
// Cross-handle freelist visibility
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, CreateThenOpen) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  SharedPoolSide opener;
  IREE_ASSERT_OK(SetupOpener(creator, kBufferSize, kBufferCount, &opener));

  // Both handles should see the same pool state.
  EXPECT_EQ(iree_async_buffer_pool_capacity(creator.pool), kBufferCount);
  EXPECT_EQ(iree_async_buffer_pool_capacity(opener.pool), kBufferCount);
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), kBufferCount);
  EXPECT_EQ(iree_async_buffer_pool_available(opener.pool), kBufferCount);

  TeardownSide(&opener);
  TeardownSide(&creator);
}

TEST_P(SharedBufferPoolTest, CrossHandleAcquireRelease) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  SharedPoolSide opener;
  IREE_ASSERT_OK(SetupOpener(creator, kBufferSize, kBufferCount, &opener));

  // Acquire from creator, verify opener sees reduced availability.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(creator.pool, &lease));
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), kBufferCount - 1);
  EXPECT_EQ(iree_async_buffer_pool_available(opener.pool), kBufferCount - 1);

  // Release via the lease callback (which uses the creator's pool handle,
  // but pushes to the shared freelist visible to both).
  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), kBufferCount);
  EXPECT_EQ(iree_async_buffer_pool_available(opener.pool), kBufferCount);

  // Acquire from opener this time.
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(opener.pool, &lease));
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), kBufferCount - 1);

  // Release via the opener's lease.
  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(iree_async_buffer_pool_available(opener.pool), kBufferCount);

  TeardownSide(&opener);
  TeardownSide(&creator);
}

TEST_P(SharedBufferPoolTest, ExhaustionVisibleAcrossHandles) {
  static constexpr iree_host_size_t kSmallCount = 4;
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kSmallCount, &creator));

  SharedPoolSide opener;
  IREE_ASSERT_OK(SetupOpener(creator, kBufferSize, kSmallCount, &opener));

  // Exhaust all buffers from the creator.
  iree_async_buffer_lease_t leases[kSmallCount];
  for (iree_host_size_t i = 0; i < kSmallCount; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(creator.pool, &leases[i]));
  }

  // Both handles should see exhaustion.
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), 0u);
  EXPECT_EQ(iree_async_buffer_pool_available(opener.pool), 0u);

  // Opener should get RESOURCE_EXHAUSTED.
  iree_async_buffer_lease_t extra;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_async_buffer_pool_acquire(opener.pool, &extra));

  // Release one from creator — opener should now be able to acquire.
  iree_async_buffer_lease_release(&leases[0]);
  EXPECT_EQ(iree_async_buffer_pool_available(opener.pool), 1u);

  iree_async_buffer_lease_t recovered;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(opener.pool, &recovered));
  iree_async_buffer_lease_release(&recovered);

  // Release remaining.
  for (iree_host_size_t i = 1; i < kSmallCount; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }

  TeardownSide(&opener);
  TeardownSide(&creator);
}

//===----------------------------------------------------------------------===//
// Buffer data accessibility across mappings
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, BufferDataCoherentAcrossMappings) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  SharedPoolSide opener;
  IREE_ASSERT_OK(SetupOpener(creator, kBufferSize, kBufferCount, &opener));

  // Acquire from creator and write a pattern.
  iree_async_buffer_lease_t creator_lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(creator.pool, &creator_lease));
  uint8_t* creator_ptr = iree_async_span_ptr(creator_lease.span);
  ASSERT_NE(creator_ptr, nullptr);
  memset(creator_ptr, 0xAB, kBufferSize);

  // The opener's mapping of the same physical memory should see the pattern.
  // Calculate the equivalent pointer in the opener's address space.
  iree_host_size_t pool_storage = 0;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_shared_storage_size(kBufferCount, &pool_storage));
  uint8_t* opener_buffer_base = (uint8_t*)opener.shm.base + pool_storage;
  uint8_t* opener_ptr =
      opener_buffer_base + creator_lease.buffer_index * kBufferSize;
  EXPECT_EQ(memcmp(opener_ptr, creator_ptr, kBufferSize), 0);

  iree_async_buffer_lease_release(&creator_lease);
  TeardownSide(&opener);
  TeardownSide(&creator);
}

//===----------------------------------------------------------------------===//
// No duplicate buffer indices
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, NoDuplicateIndices) {
  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kBufferCount, &creator));

  SharedPoolSide opener;
  IREE_ASSERT_OK(SetupOpener(creator, kBufferSize, kBufferCount, &opener));

  // Acquire half from creator, half from opener.
  std::set<iree_async_buffer_index_t> indices;
  iree_async_buffer_lease_t leases[kBufferCount];
  for (iree_host_size_t i = 0; i < kBufferCount; ++i) {
    iree_async_buffer_pool_t* pool = (i % 2 == 0) ? creator.pool : opener.pool;
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &leases[i]));
    // Every index must be unique.
    EXPECT_TRUE(indices.insert(leases[i].buffer_index).second)
        << "Duplicate buffer index " << leases[i].buffer_index << " at acquire "
        << i;
  }
  EXPECT_EQ(indices.size(), kBufferCount);

  for (iree_host_size_t i = 0; i < kBufferCount; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }

  TeardownSide(&opener);
  TeardownSide(&creator);
}

//===----------------------------------------------------------------------===//
// Concurrent acquire/release stress test
//===----------------------------------------------------------------------===//

TEST_P(SharedBufferPoolTest, ConcurrentAcquireRelease) {
  static constexpr iree_host_size_t kStressBufferCount = 64;
  static constexpr int kThreadCount = 4;
  static constexpr int kIterationsPerThread = 1000;

  SharedPoolSide creator;
  IREE_ASSERT_OK(SetupCreator(kBufferSize, kStressBufferCount, &creator));

  SharedPoolSide opener;
  IREE_ASSERT_OK(
      SetupOpener(creator, kBufferSize, kStressBufferCount, &opener));

  // Track which buffer indices are currently held by any thread.
  // Uses an atomic flag per index to detect double-allocation.
  std::vector<std::atomic<bool>> held(kStressBufferCount);
  for (auto& flag : held) flag.store(false);

  std::atomic<int> error_count{0};

  auto worker = [&](iree_async_buffer_pool_t* pool) {
    for (int i = 0; i < kIterationsPerThread; ++i) {
      iree_async_buffer_lease_t lease;
      iree_status_t status = iree_async_buffer_pool_acquire(pool, &lease);
      if (!iree_status_is_ok(status)) {
        // RESOURCE_EXHAUSTED is expected under contention.
        iree_status_ignore(status);
        continue;
      }

      // Mark this index as held. If it was already held, that's a bug.
      bool was_held =
          held[lease.buffer_index].exchange(true, std::memory_order_acq_rel);
      if (was_held) {
        error_count.fetch_add(1, std::memory_order_relaxed);
      }

      // Brief hold to increase contention window.
      // (The compiler cannot optimize this away because held[] is atomic.)
      held[lease.buffer_index].store(false, std::memory_order_release);
      iree_async_buffer_lease_release(&lease);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  // Half the threads use the creator's pool handle, half use the opener's.
  for (int t = 0; t < kThreadCount; ++t) {
    iree_async_buffer_pool_t* pool =
        (t < kThreadCount / 2) ? creator.pool : opener.pool;
    threads.emplace_back(worker, pool);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(error_count.load(), 0)
      << "Detected double-allocation of buffer indices";

  // All buffers should be returned.
  EXPECT_EQ(iree_async_buffer_pool_available(creator.pool), kStressBufferCount);

  TeardownSide(&opener);
  TeardownSide(&creator);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(SharedBufferPoolTest);

}  // namespace iree::async::cts
