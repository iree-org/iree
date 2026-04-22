// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/passthrough_pool.h"

#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

typedef struct iree_hal_test_opaque_slab_provider_t {
  iree_hal_slab_provider_t base;
  iree_allocator_t host_allocator;
  iree_atomic_int32_t wrap_count;
} iree_hal_test_opaque_slab_provider_t;

extern const iree_hal_slab_provider_vtable_t
    iree_hal_test_opaque_slab_provider_vtable;

static iree_status_t iree_hal_test_opaque_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  *out_provider = NULL;
  iree_hal_test_opaque_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_test_opaque_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  *out_provider = &provider->base;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_allocator_free(provider->host_allocator, provider);
}

static iree_status_t iree_hal_test_opaque_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_hal_slab_t* out_slab) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));
  void* backing = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      provider->host_allocator, min_length, IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      /*offset=*/0, &backing));
  out_slab->base_ptr = (uint8_t*)(uintptr_t)1;
  out_slab->length = min_length;
  out_slab->provider_handle = (uint64_t)(uintptr_t)backing;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_allocator_free_aligned(provider->host_allocator,
                              (void*)(uintptr_t)slab->provider_handle);
}

static iree_status_t iree_hal_test_opaque_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_atomic_fetch_add(&provider->wrap_count, 1, iree_memory_order_relaxed);
  iree_byte_span_t data = iree_make_byte_span(
      (uint8_t*)(uintptr_t)slab->provider_handle + slab_offset,
      (iree_host_size_t)allocation_size);
  return iree_hal_heap_buffer_wrap(iree_hal_buffer_placement_undefined(),
                                   params.type, params.access, params.usage,
                                   allocation_size, data, release_callback,
                                   provider->host_allocator, out_buffer);
}

static void iree_hal_test_opaque_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, iree_hal_slab_t* slab) {}

static void iree_hal_test_opaque_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_trim_flags_t flags) {}

static void iree_hal_test_opaque_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  iree_hal_slab_provider_visited(visited, base_provider);
}

static void iree_hal_test_opaque_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  *out_memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
      IREE_HAL_MEMORY_TYPE_HOST_COHERENT | IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  *out_supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                         IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                         IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
}

const iree_hal_slab_provider_vtable_t
    iree_hal_test_opaque_slab_provider_vtable = {
        /*.destroy=*/iree_hal_test_opaque_slab_provider_destroy,
        /*.acquire_slab=*/iree_hal_test_opaque_slab_provider_acquire_slab,
        /*.release_slab=*/iree_hal_test_opaque_slab_provider_release_slab,
        /*.wrap_buffer=*/iree_hal_test_opaque_slab_provider_wrap_buffer,
        /*.prefault=*/iree_hal_test_opaque_slab_provider_prefault,
        /*.trim=*/iree_hal_test_opaque_slab_provider_trim,
        /*.query_stats=*/iree_hal_test_opaque_slab_provider_query_stats,
        /*.query_properties=*/
        iree_hal_test_opaque_slab_provider_query_properties,
};

class PassthroughPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = iree_allocator_system();
    IREE_ASSERT_OK(
        iree_hal_cpu_slab_provider_create(allocator_, &slab_provider_));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    iree_hal_passthrough_pool_options_t options = {0};
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        options, slab_provider_, notification_, allocator_, &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_async_notification_release(notification_);
    iree_hal_slab_provider_release(slab_provider_);
  }

  iree_allocator_t allocator_;
  iree_hal_slab_provider_t* slab_provider_ = nullptr;
  iree_async_notification_t* notification_ = nullptr;
  iree_hal_pool_t* pool_ = nullptr;
};

TEST_F(PassthroughPoolTest, ReserveRelease) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 4096, 1, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 0u);
  EXPECT_GE(reservation.length, 4096u);
  EXPECT_NE(reservation.block_handle, 0u);
  EXPECT_EQ(reserve_info.wait_frontier, nullptr);
  EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  iree_hal_pool_release_reservation(pool_, &reservation, NULL);
}

TEST_F(PassthroughPoolTest, ReserveRejectsUnsupportedAlignment) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_pool_acquire_reservation(
                            pool_, 4096, IREE_HAL_HEAP_BUFFER_ALIGNMENT * 2,
                            NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
                            &reserve_info, &result));
}

TEST_F(PassthroughPoolTest, StatsTrackReserveRelease) {
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.release_count, 0u);

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 1024, 1, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &reserve_info, &result));

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  EXPECT_GE(stats.bytes_reserved, 1024u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 0u);
  EXPECT_EQ(stats.fresh_count, 1u);

  iree_hal_pool_release_reservation(pool_, &reservation, NULL);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(PassthroughPoolTest, WrapReservationCreatesBuffer) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 4096, 1, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservation(
      pool_, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer), 4096u);

  // Releasing the buffer should release the reservation back to the pool.
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(PassthroughPoolTest,
       BorrowedMaterializationKeepsReservationUntilExplicitRelease) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 1024, 1, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservation(
      pool_, params, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));

  iree_hal_buffer_release(buffer);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  EXPECT_EQ(stats.release_count, 0u);
  EXPECT_EQ(stats.slab_count, 1u);

  iree_hal_pool_release_reservation(pool_, &reservation,
                                    /*death_frontier=*/NULL);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
  EXPECT_EQ(stats.slab_count, 0u);
}

TEST(PassthroughPool, WrapReservationUsesProviderHook) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* pool = NULL;
  iree_hal_passthrough_pool_options_t options = {0};
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      options, slab_provider, notification, allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 1, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservation(
      pool, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0, 256,
      &mapping));
  memset(mapping.contents.data, 0x6B, 256);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[255], 0x6B);
  iree_hal_buffer_unmap_range(&mapping);

  EXPECT_EQ(
      1,
      iree_atomic_load(
          &((iree_hal_test_opaque_slab_provider_t*)slab_provider)->wrap_count,
          iree_memory_order_relaxed));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(PassthroughPoolTest, BufferMemoryAccess) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 256, 1, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservation(
      pool_, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  // Map, write, read back.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_READ, 0, 256,
      &mapping));
  memset(mapping.contents.data, 0xCD, 256);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[0], 0xCD);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[255], 0xCD);
  iree_hal_buffer_unmap_range(&mapping);

  iree_hal_buffer_release(buffer);
}

TEST_F(PassthroughPoolTest, Capabilities) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool_, &capabilities);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));
}

TEST_F(PassthroughPoolTest, TrimIsNoOp) {
  IREE_EXPECT_OK(iree_hal_pool_trim(pool_));
}

TEST_F(PassthroughPoolTest, MultipleReservations) {
  iree_hal_pool_reservation_t reservations[4];
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
        pool_, 1024 * (i + 1), 1, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        &reservations[i], &reserve_info, &result));
    EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    EXPECT_EQ(reserve_info.wait_frontier, nullptr);
    EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);
  }

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 4u);
  EXPECT_EQ(stats.slab_count, 4u);
  EXPECT_EQ(stats.reserve_count, 4u);

  for (int i = 3; i >= 0; --i) {
    iree_hal_pool_release_reservation(pool_, &reservations[i], NULL);
  }

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.slab_count, 0u);
  EXPECT_EQ(stats.release_count, 4u);
}

TEST_F(PassthroughPoolTest, AllocateBuffer) {
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, params, 2048, NULL, iree_make_timeout_ms(0), &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_GE(iree_hal_buffer_allocation_size(buffer), 2048u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_F(PassthroughPoolTest, WrappedBuffersBorrowPool) {
  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, params, 512, NULL, iree_make_timeout_ms(0), &buffer));

  // Wrapped buffers borrow the pool. Use the buffer while the pool is still
  // alive, then release the buffer before releasing the pool.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_WRITE, 0, 512,
                                           &mapping));
  memset(mapping.contents.data, 0xEF, 512);
  iree_hal_buffer_unmap_range(&mapping);

  iree_hal_buffer_release(buffer);
}

}  // namespace
