// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for buffer registration operations.
//
// Tests host buffer registration, dmabuf registration, and RECV_POOL
// operations. These tests verify the buffer registration API works correctly
// across all proactor backends.

#include <cstring>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/slab.h"

// memfd_create is used as a dmabuf stand-in for tests. Available on Linux
// glibc 2.27+ and Android API level 30+.
#if defined(IREE_PLATFORM_LINUX) && \
    !(defined(__ANDROID_API__) && __ANDROID_API__ < 30)
#define IREE_TEST_HAS_MEMFD 1
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace iree::async::cts {

class BufferRegistrationTest : public SocketTestBase<> {};

//===----------------------------------------------------------------------===//
// Host buffer registration tests
//===----------------------------------------------------------------------===//

// Basic host buffer registration and unregistration.
TEST_P(BufferRegistrationTest, RegisterHostBuffer) {
  std::vector<uint8_t> buffer(4096, 0xAB);

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &state, iree_make_byte_span(buffer.data(), buffer.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &entry));

  ASSERT_NE(entry, nullptr);
  EXPECT_NE(entry->region, nullptr);
  EXPECT_EQ(entry->region->base_ptr, buffer.data());
  EXPECT_EQ(entry->region->length, buffer.size());

  iree_async_buffer_registration_state_deinitialize(&state);
}

// Multiple buffer registrations in the same state.
TEST_P(BufferRegistrationTest, RegisterMultipleBuffers) {
  std::vector<uint8_t> buffer1(1024, 0x11);
  std::vector<uint8_t> buffer2(2048, 0x22);
  std::vector<uint8_t> buffer3(4096, 0x33);

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* entry1 = nullptr;
  iree_async_buffer_registration_entry_t* entry2 = nullptr;
  iree_async_buffer_registration_entry_t* entry3 = nullptr;

  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &state, iree_make_byte_span(buffer1.data(), buffer1.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &entry1));

  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &state, iree_make_byte_span(buffer2.data(), buffer2.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &entry2));

  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &state, iree_make_byte_span(buffer3.data(), buffer3.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
      &entry3));

  ASSERT_NE(entry1, nullptr);
  ASSERT_NE(entry2, nullptr);
  ASSERT_NE(entry3, nullptr);

  // All entries should have valid regions.
  EXPECT_EQ(entry1->region->base_ptr, buffer1.data());
  EXPECT_EQ(entry2->region->base_ptr, buffer2.data());
  EXPECT_EQ(entry3->region->base_ptr, buffer3.data());

  iree_async_buffer_registration_state_deinitialize(&state);
}

// State cleanup unregisters all buffers.
TEST_P(BufferRegistrationTest, StateCleanupUnregistersAll) {
  std::vector<uint8_t> buffer(4096, 0xCD);

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &state, iree_make_byte_span(buffer.data(), buffer.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &entry));

  EXPECT_FALSE(iree_async_buffer_registration_state_is_empty(&state));

  // Cleanup should clear the state.
  iree_async_buffer_registration_state_deinitialize(&state);

  EXPECT_TRUE(iree_async_buffer_registration_state_is_empty(&state));
}

//===----------------------------------------------------------------------===//
// dmabuf registration tests (Linux-only)
//===----------------------------------------------------------------------===//

#if IREE_TEST_HAS_MEMFD

// Basic dmabuf registration using memfd as a test stand-in.
TEST_P(BufferRegistrationTest, RegisterDmabuf) {
  // Check if backend supports dmabuf registration.
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF)) {
    GTEST_SKIP() << "Backend does not support dmabuf registration";
  }

  // Create test dmabuf via memfd (memfd is not a real dmabuf, but the mmap
  // fallback implementation treats any fd similarly).
  int memfd = memfd_create("test_dmabuf", MFD_CLOEXEC);
  ASSERT_GE(memfd, 0) << "memfd_create failed";
  ASSERT_EQ(ftruncate(memfd, 4096), 0) << "ftruncate failed";

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_dmabuf(
      proactor_, &state, memfd, /*offset=*/0, /*length=*/4096,
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
      &entry));

  ASSERT_NE(entry, nullptr);
  EXPECT_NE(entry->region, nullptr);
  EXPECT_EQ(entry->region->length, 4096u);
  EXPECT_NE(entry->region->base_ptr, nullptr);

  // Region type depends on kernel version:
  //   5.19+ (sparse buffer table): IOURING type with fixed buffer index for
  //     zero-copy send. The mmap'd memory is registered in the kernel's buffer
  //     table alongside slab registrations.
  //   Pre-5.19: DMABUF type with mmap-only fallback (no zero-copy send).
  EXPECT_TRUE(entry->region->type == IREE_ASYNC_REGION_TYPE_IOURING ||
              entry->region->type == IREE_ASYNC_REGION_TYPE_DMABUF);

  // Verify memory is accessible via mmap (works regardless of region type).
  uint8_t* ptr = static_cast<uint8_t*>(entry->region->base_ptr);
  ptr[0] = 0xCD;
  EXPECT_EQ(ptr[0], 0xCD);

  // Verify backend-specific handles.
  if (entry->region->type == IREE_ASYNC_REGION_TYPE_IOURING) {
    EXPECT_EQ(entry->region->buffer_count, 1u);
    EXPECT_EQ(entry->region->buffer_size, 4096u);
    EXPECT_LT(entry->region->handles.iouring.buffer_group_id, 0)
        << "dmabuf regions are not provided buffer rings";
  } else {
    EXPECT_EQ(entry->region->handles.dmabuf.fd, memfd);
    EXPECT_EQ(entry->region->handles.dmabuf.offset, 0u);
  }

  iree_async_buffer_registration_state_deinitialize(&state);
  close(memfd);
}

// dmabuf registration with non-zero offset.
TEST_P(BufferRegistrationTest, RegisterDmabufWithOffset) {
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF)) {
    GTEST_SKIP() << "Backend does not support dmabuf registration";
  }

  int memfd = memfd_create("test_dmabuf_offset", MFD_CLOEXEC);
  ASSERT_GE(memfd, 0);
  ASSERT_EQ(ftruncate(memfd, 8192), 0);

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_dmabuf(
      proactor_, &state, memfd, /*offset=*/4096, /*length=*/4096,
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &entry));

  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->region->length, 4096u);

  // On 5.19+ the region may be IOURING (zero-copy registered). On pre-5.19
  // it's DMABUF with the original offset preserved in the handles.
  if (entry->region->type == IREE_ASYNC_REGION_TYPE_DMABUF) {
    EXPECT_EQ(entry->region->handles.dmabuf.offset, 4096u);
  }

  iree_async_buffer_registration_state_deinitialize(&state);
  close(memfd);
}

#endif  // IREE_TEST_HAS_MEMFD

//===----------------------------------------------------------------------===//
// RECV_POOL operation tests (using slab + register_slab + pool)
//===----------------------------------------------------------------------===//

// Basic recv pool test: create slab, register with WRITE access, verify pool.
TEST_P(BufferRegistrationTest, RecvPoolConfiguration) {
  // Check if backend supports multishot (required for RECV_POOL).
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "Backend does not support multishot operations";
  }

  // Create slab for recv buffers (WRITE access for kernel-to-user data path).
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 16;  // Must be power of 2 for recv pools.

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  // Register slab with WRITE access for recv operations.
  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));

  // Create pool over registered region.
  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));
  ASSERT_NE(pool, nullptr);

  // Verify pool configuration.
  EXPECT_EQ(iree_async_buffer_pool_capacity(pool), 16u);
  EXPECT_EQ(iree_async_buffer_pool_buffer_size(pool), 1024u);

  // Verify region is configured (backend-specific type is set).
  iree_async_region_t* pool_region = iree_async_buffer_pool_region(pool);
  EXPECT_NE(pool_region, nullptr);
  EXPECT_NE(pool_region->type, IREE_ASYNC_REGION_TYPE_NONE);

  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Test that non-power-of-2 buffer_count may be rejected for recv pools.
TEST_P(BufferRegistrationTest, RecvPoolRequiresPowerOfTwo) {
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "Backend does not support multishot operations";
  }

  // Create slab with non-power-of-2 count.
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 7;  // Not a power of 2.

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  // Registration should fail for backends requiring power-of-2 for recv pools.
  iree_async_region_t* region = nullptr;
  iree_status_t status = iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(region, nullptr);

  iree_async_slab_release(slab);
}

// Test buffer pool release recycles buffers correctly.
TEST_P(BufferRegistrationTest, RecvPoolBufferRecycling) {
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "Backend does not support multishot operations";
  }

  // Create slab, register, create pool.
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 512;
  slab_options.buffer_count = 4;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Acquire all buffers.
  iree_async_buffer_lease_t leases[4];
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &leases[i]));
  }
  EXPECT_EQ(iree_async_buffer_pool_available(pool), 0u);

  // Release all buffers (recycles to freelist).
  for (int i = 0; i < 4; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }
  EXPECT_EQ(iree_async_buffer_pool_available(pool), 4u);

  // Acquire again to verify buffers are reusable.
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &leases[i]));
  }

  for (int i = 0; i < 4; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }

  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

//===----------------------------------------------------------------------===//
// Multiple registration tests (sparse buffer table, 5.19+)
//===----------------------------------------------------------------------===//

// Verifies that multiple send slabs can be registered on 5.19+ kernels where
// the sparse buffer table allows independent registrations. On pre-5.19 the
// second registration fails with ALREADY_EXISTS because the legacy singleton
// IORING_REGISTER_BUFFERS path only supports one buffer table per ring.
TEST_P(BufferRegistrationTest, MultipleSendSlabRegistrations) {
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 4;

  iree_async_slab_t* slab_a = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab_a));

  iree_async_slab_t* slab_b = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab_b));

  // First slab always succeeds.
  iree_async_region_t* region_a = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab_a, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_a));

  // Second slab: succeeds on 5.19+ (sparse table), fails on pre-5.19 (legacy).
  iree_async_region_t* region_b = nullptr;
  iree_status_t status = iree_async_proactor_register_slab(
      proactor_, slab_b, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_b);

  if (iree_status_is_ok(status)) {
    // 5.19+ path: both registrations succeeded with distinct buffer indices.
    ASSERT_NE(region_b, nullptr);
    EXPECT_EQ(region_a->type, IREE_ASYNC_REGION_TYPE_IOURING);
    EXPECT_EQ(region_b->type, IREE_ASYNC_REGION_TYPE_IOURING);
    EXPECT_NE(region_a->handles.iouring.base_buffer_index,
              region_b->handles.iouring.base_buffer_index)
        << "Distinct slabs must have different buffer table indices";
    iree_async_region_release(region_b);
  } else {
    // Pre-5.19 path: second registration rejected (singleton buffer table).
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS, status);
    EXPECT_EQ(region_b, nullptr);
  }

  iree_async_region_release(region_a);
  iree_async_slab_release(slab_b);
  iree_async_slab_release(slab_a);
}

// Verifies that slab and dmabuf registrations can coexist in the same buffer
// table on 5.19+ kernels. Both share the sparse table's slot space.
#if IREE_TEST_HAS_MEMFD
TEST_P(BufferRegistrationTest, SlabAndDmabufCoexist) {
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_DMABUF)) {
    GTEST_SKIP() << "Backend does not support dmabuf registration";
  }

  // Register a slab with READ access.
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 4;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* slab_region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &slab_region));

  // Register a dmabuf with READ access.
  int memfd = memfd_create("test_coexist", MFD_CLOEXEC);
  ASSERT_GE(memfd, 0);
  ASSERT_EQ(ftruncate(memfd, 4096), 0);

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* dmabuf_entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_dmabuf(
      proactor_, &state, memfd, /*offset=*/0, /*length=*/4096,
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &dmabuf_entry));

  // On 5.19+, both should be IOURING type with non-overlapping buffer indices.
  if (slab_region->type == IREE_ASYNC_REGION_TYPE_IOURING &&
      dmabuf_entry->region->type == IREE_ASYNC_REGION_TYPE_IOURING) {
    // Slab occupies [base, base+count). Dmabuf occupies a single slot.
    // They must not overlap.
    uint16_t slab_base = slab_region->handles.iouring.base_buffer_index;
    uint16_t slab_end =
        slab_base + static_cast<uint16_t>(slab_region->buffer_count);
    uint16_t dmabuf_slot =
        dmabuf_entry->region->handles.iouring.base_buffer_index;
    EXPECT_TRUE(dmabuf_slot < slab_base || dmabuf_slot >= slab_end)
        << "dmabuf slot " << dmabuf_slot << " overlaps slab range ["
        << slab_base << ", " << slab_end << ")";
  }

  iree_async_buffer_registration_state_deinitialize(&state);
  close(memfd);
  iree_async_region_release(slab_region);
  iree_async_slab_release(slab);
}
#endif  // IREE_TEST_HAS_MEMFD

//===----------------------------------------------------------------------===//
// Reference counting and lifetime tests
//===----------------------------------------------------------------------===//

// Verifies that regions use proper reference counting, allowing callers to
// retain a region beyond the lifetime of the registration state. This is
// essential for in-flight operations that hold region references - cleanup of
// the registration state must not invalidate regions that are still in use.
TEST_P(BufferRegistrationTest, RegionRetainedAfterStateCleanup) {
  std::vector<uint8_t> buffer(4096, 0xAB);

  iree_async_buffer_registration_state_t state;
  iree_async_buffer_registration_state_initialize(&state);

  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &state, iree_make_byte_span(buffer.data(), buffer.size()),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &entry));

  ASSERT_NE(entry, nullptr);
  ASSERT_NE(entry->region, nullptr);

  // Retain the region, simulating an in-flight operation holding a reference.
  iree_async_region_t* retained_region = entry->region;
  iree_async_region_retain(retained_region);

  // Cleanup the state. The region must remain valid since we hold a reference.
  iree_async_buffer_registration_state_deinitialize(&state);

  // Region must still be accessible (ASAN will catch use-after-free).
  EXPECT_NE(retained_region->base_ptr, nullptr);
  EXPECT_EQ(retained_region->length, buffer.size());

  // Release our reference, which should trigger the actual free.
  iree_async_region_release(retained_region);
}

// Verifies that RECV_POOL operations are rejected when used with a region that
// lacks a provided buffer ring. Buffer rings are only created for regions
// registered with WRITE access (the kernel writes received data into them).
// READ-only regions cannot be used for RECV_POOL and must fail at submission
// time with a clear error rather than proceeding to submit an invalid SQE.
TEST_P(BufferRegistrationTest, RecvPoolRejectsReadOnlyRegion) {
  iree_async_proactor_capabilities_t caps =
      iree_async_proactor_query_capabilities(proactor_);
  if (!(caps & IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "Backend does not support multishot operations";
  }

  // Create slab and register with READ-only access (no buffer ring created).
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 4;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  // Create pool over the READ-only region.
  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Verify the region has no buffer ring (io_uring uses buffer_group_id = -1).
  if (region->type == IREE_ASYNC_REGION_TYPE_IOURING) {
    EXPECT_LT(region->handles.iouring.buffer_group_id, 0)
        << "READ-only region should not have a buffer ring";
  }

  // Create a connected socket pair via loopback to test actual submission.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  // Submit connect and accept to establish the connection.
  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  iree_async_operation_t* setup_ops[] = {&connect_op.base, &accept_op.base};
  IREE_ASSERT_OK(iree_async_proactor_submit(
      proactor_,
      (iree_async_operation_list_t){setup_ops, IREE_ARRAYSIZE(setup_ops)}));
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_ASSERT_OK(connect_tracker.ConsumeStatus());
  IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
  iree_async_socket_t* server = accept_op.accepted_socket;
  ASSERT_NE(server, nullptr);

  // Submit a RECV_POOL operation using the READ-only pool. This must fail
  // at submission time with FAILED_PRECONDITION, not proceed to the kernel.
  iree_async_socket_recv_pool_operation_t recv_pool_op;
  memset(&recv_pool_op, 0, sizeof(recv_pool_op));
  recv_pool_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL;
  recv_pool_op.socket = server;
  recv_pool_op.pool = pool;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_async_proactor_submit_one(proactor_, &recv_pool_op.base));

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Verifies that releasing a buffer lease is idempotent. Double-releasing the
// same lease must be a safe no-op, not corrupt the freelist by pushing the
// same buffer index twice. Freelist corruption would cause the same buffer to
// be handed out to multiple concurrent operations, leading to data races.
TEST_P(BufferRegistrationTest, LeaseDoubleReleaseIsIdempotent) {
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 256;
  slab_options.buffer_count = 4;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Acquire a lease.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &lease));
  EXPECT_EQ(iree_async_buffer_pool_available(pool), 3u);

  // Release the lease.
  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(iree_async_buffer_pool_available(pool), 4u);

  // Double-release must be a no-op, not push the buffer twice.
  iree_async_buffer_lease_release(&lease);
  EXPECT_EQ(iree_async_buffer_pool_available(pool), 4u)
      << "Double release must be idempotent";

  // Verify no freelist corruption by acquiring all buffers and checking for
  // duplicate pointers (same buffer handed out twice indicates corruption).
  iree_async_buffer_lease_t leases[4];
  void* pointers[4] = {nullptr};
  for (int i = 0; i < 4; ++i) {
    iree_status_t status = iree_async_buffer_pool_acquire(pool, &leases[i]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }
    pointers[i] = iree_async_span_ptr(leases[i].span);
  }

  // All pointers must be unique.
  for (int i = 0; i < 4 && pointers[i]; ++i) {
    for (int j = i + 1; j < 4 && pointers[j]; ++j) {
      EXPECT_NE(pointers[i], pointers[j])
          << "Buffers " << i << " and " << j << " have same pointer; "
          << "freelist corruption detected";
    }
  }

  for (int i = 0; i < 4; ++i) {
    if (pointers[i]) {
      iree_async_buffer_lease_release(&leases[i]);
    }
  }

  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(BufferRegistrationTest);

}  // namespace iree::async::cts
