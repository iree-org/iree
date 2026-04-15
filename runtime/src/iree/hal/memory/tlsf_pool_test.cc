// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool.h"

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

static iree_async_axis_t TestQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 0, 0, queue_index);
}

static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

static iree_async_frontier_t* BuildFrontier(
    uint8_t* storage, iree_host_size_t storage_size,
    std::initializer_list<iree_async_frontier_entry_t> entries) {
  iree_async_frontier_t* frontier =
      reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(frontier,
                                 static_cast<uint8_t>(entries.size()));
  uint8_t i = 0;
  for (const auto& entry : entries) {
    frontier->entries[i++] = entry;
  }
  return frontier;
}

#define MAKE_FRONTIER(name, capacity, ...)                              \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      BuildFrontier(name##_storage, sizeof(name##_storage), {__VA_ARGS__})

typedef struct iree_hal_test_counting_allocator_t {
  // Allocator that performs the actual memory operations.
  iree_allocator_t backing_allocator;
  // Number of allocation-like commands forwarded to the backing allocator.
  iree_host_size_t allocation_call_count;
  // Number of free commands forwarded to the backing allocator.
  iree_host_size_t free_call_count;
} iree_hal_test_counting_allocator_t;

static iree_status_t iree_hal_test_counting_allocator_ctl(
    void* self, iree_allocator_command_t command, const void* params,
    void** inout_ptr) {
  iree_hal_test_counting_allocator_t* allocator =
      (iree_hal_test_counting_allocator_t*)self;
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      ++allocator->allocation_call_count;
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++allocator->free_call_count;
      break;
  }
  return allocator->backing_allocator.ctl(allocator->backing_allocator.self,
                                          command, params, inout_ptr);
}

static iree_allocator_t iree_hal_test_counting_allocator(
    iree_hal_test_counting_allocator_t* allocator) {
  return iree_allocator_t{
      /*.self=*/allocator,
      /*.ctl=*/iree_hal_test_counting_allocator_ctl,
  };
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
  iree_byte_span_t data = {
      .data = (uint8_t*)(uintptr_t)slab->provider_handle + slab_offset,
      .data_length = (iree_host_size_t)allocation_size,
  };
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
        .destroy = iree_hal_test_opaque_slab_provider_destroy,
        .acquire_slab = iree_hal_test_opaque_slab_provider_acquire_slab,
        .release_slab = iree_hal_test_opaque_slab_provider_release_slab,
        .wrap_buffer = iree_hal_test_opaque_slab_provider_wrap_buffer,
        .prefault = iree_hal_test_opaque_slab_provider_prefault,
        .trim = iree_hal_test_opaque_slab_provider_trim,
        .query_stats = iree_hal_test_opaque_slab_provider_query_stats,
        .query_properties = iree_hal_test_opaque_slab_provider_query_properties,
};

static iree_hal_tlsf_pool_options_t DefaultOptions() {
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = 4096;
  options.tlsf_options.alignment = 16;
  options.tlsf_options.initial_block_capacity = 16;
  options.tlsf_options.frontier_capacity = 2;
  return options;
}

class TLSFPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_hal_cpu_slab_provider_create(allocator_, &slab_provider_));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
        DefaultOptions(), slab_provider_, notification_,
        iree_hal_pool_epoch_query_null(), allocator_, &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_async_notification_release(notification_);
    iree_hal_slab_provider_release(slab_provider_);
  }

  iree_allocator_t allocator_ = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider_ = nullptr;
  iree_async_notification_t* notification_ = nullptr;
  iree_hal_pool_t* pool_ = nullptr;
};

TEST_F(TLSFPoolTest, ReserveReleaseFresh) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 0u);
  EXPECT_GE(reservation.length, 128u);
  EXPECT_NE(reservation.block_handle, 0u);
  EXPECT_EQ(reservation.slab_index, 0u);
  EXPECT_EQ(reserve_info.wait_frontier, nullptr);
  EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  iree_hal_pool_release_reservation(pool_, &reservation, NULL);
}

TEST_F(TLSFPoolTest, ReleaseWithObserverSignalsNotification) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  const uint32_t wait_token =
      iree_async_notification_begin_observe(notification_);
  iree_hal_pool_release_reservation(pool_, &reservation, NULL);
  EXPECT_TRUE(iree_async_notification_wait_for_token(notification_, wait_token,
                                                     iree_make_timeout_ms(0)));
  iree_async_notification_end_observe(notification_);
}

TEST(TLSFPool, ReleaseNodeReuseAvoidsRepeatedHostAllocation) {
  iree_hal_test_counting_allocator_t allocator_state = {
      /*.backing_allocator=*/iree_allocator_system(),
      /*.allocation_call_count=*/0,
      /*.free_call_count=*/0,
  };
  iree_allocator_t allocator =
      iree_hal_test_counting_allocator(&allocator_state);

  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      DefaultOptions(), slab_provider, notification,
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  const iree_host_size_t allocation_call_count_after_create =
      allocator_state.allocation_call_count;

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_GT(allocator_state.allocation_call_count,
            allocation_call_count_after_create);

  const iree_host_size_t allocation_call_count_after_first_reserve =
      allocator_state.allocation_call_count;
  iree_hal_pool_release_reservation(pool, &reservation, NULL);

  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(allocator_state.allocation_call_count,
            allocation_call_count_after_first_reserve);

  iree_hal_pool_release_reservation(pool, &reservation, NULL);
  const iree_host_size_t free_call_count_before_trim =
      allocator_state.free_call_count;
  IREE_ASSERT_OK(iree_hal_pool_trim(pool));
  EXPECT_GT(allocator_state.free_call_count, free_call_count_before_trim);

  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(TLSFPoolTest, ReserveReusesDominatedFrontier) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_release_reservation(pool_, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation,
      &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(reservation.offset, 0u);
  EXPECT_EQ(reserve_info.wait_frontier, nullptr);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reuse_count, 1u);
  EXPECT_EQ(stats.fresh_count, 1u);

  iree_hal_pool_release_reservation(pool_, &reservation, NULL);
}

TEST(TLSFPool, ReserveSkipsStaleHeadAndReturnsFreshLaterBlock) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = 1024;
  options.tlsf_options.alignment = 16;
  options.tlsf_options.initial_block_capacity = 16;
  options.tlsf_options.frontier_capacity = 2;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(options, slab_provider, notification,
                                           iree_hal_pool_epoch_query_null(),
                                           allocator, &pool));

  iree_hal_pool_reservation_t left_stale;
  iree_hal_pool_reservation_t middle_live;
  iree_hal_pool_reservation_t right_fresh;
  iree_hal_pool_reservation_t tail_live;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &left_stale, &reserve_info, &result));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &middle_live, &reserve_info, &result));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &right_fresh, &reserve_info, &result));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &tail_live, &reserve_info, &result));
  EXPECT_EQ(left_stale.offset, 0u);
  EXPECT_EQ(right_fresh.offset, 512u);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  iree_hal_pool_release_reservation(pool, &left_stale, death);
  iree_hal_pool_release_reservation(pool, &right_fresh, NULL);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_reservation_t reservation;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 512u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 1u);
  EXPECT_EQ(stats.wait_count, 0u);

  MAKE_FRONTIER(dominating_requester, 1, E(TestQueueAxis(0), 20));
  iree_hal_pool_reservation_t stale_reservation;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, dominating_requester, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &stale_reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(stale_reservation.offset, 0u);

  iree_hal_pool_release_reservation(pool, &reservation, NULL);
  iree_hal_pool_release_reservation(pool, &stale_reservation, NULL);
  iree_hal_pool_release_reservation(pool, &middle_live, NULL);
  iree_hal_pool_release_reservation(pool, &tail_live, NULL);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReserveCanReturnStaleBlockWhenWaitAllowed) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 256;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(options, slab_provider, notification,
                                           iree_hal_pool_epoch_query_null(),
                                           allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  iree_hal_pool_release_reservation(pool, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(reserve_info.wait_frontier, nullptr);
  EXPECT_EQ(reserve_info.wait_frontier->entry_count, 1u);
  EXPECT_EQ(reserve_info.wait_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(reserve_info.wait_frontier->entries[0].epoch, 20u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 1u);
  EXPECT_EQ(stats.wait_count, 1u);

  iree_hal_pool_release_reservation(pool, &reservation,
                                    reserve_info.wait_frontier);

  MAKE_FRONTIER(dominating_requester, 1, E(TestQueueAxis(0), 20));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, dominating_requester, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(reservation.offset, 0u);

  iree_hal_pool_release_reservation(pool, &reservation, NULL);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(TLSFPool, ReserveRejectedTaintRemainsRejected) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.tlsf_options.range_length = 256;
  options.tlsf_options.frontier_capacity = 1;

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(options, slab_provider, notification,
                                           iree_hal_pool_epoch_query_null(),
                                           allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(oversized_death, 2, E(TestQueueAxis(0), 10),
                E(TestQueueAxis(1), 20));
  iree_hal_pool_release_reservation(pool, &reservation, oversized_death);

  MAKE_FRONTIER(requester, 2, E(TestQueueAxis(0), 100),
                E(TestQueueAxis(1), 100));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);

  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 256, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 2u);
  EXPECT_EQ(stats.exhausted_count, 2u);
  EXPECT_EQ(stats.wait_count, 0u);

  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(TLSFPoolTest, WrapReservationCreatesBuffer) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservation(
      pool_, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0, 128,
      &mapping));
  memset(mapping.contents.data, 0x7C, 128);
  iree_hal_buffer_unmap_range(&mapping);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(TLSFPoolTest, BorrowedMaterializationDoesNotReleaseReservation) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
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

  iree_hal_pool_release_reservation(pool_, &reservation,
                                    /*death_frontier=*/NULL);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST(TLSFPool, WrapReservationUsesProviderHook) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      DefaultOptions(), slab_provider, notification,
      iree_hal_pool_epoch_query_null(), allocator, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool, 128, 16, /*requester_frontier=*/NULL,
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
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0, 128,
      &mapping));
  memset(mapping.contents.data, 0x4D, 128);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[127], 0x4D);
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

TEST_F(TLSFPoolTest, QueryCapabilitiesAndBudget) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool_, &capabilities);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_EQ(capabilities.min_allocation_size, 16u);
  EXPECT_EQ(capabilities.max_allocation_size, 4096u);

  iree_hal_pool_release(pool_);
  pool_ = NULL;

  iree_hal_tlsf_pool_options_t options = DefaultOptions();
  options.budget_limit = 63;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create(
      options, slab_provider_, notification_, iree_hal_pool_epoch_query_null(),
      allocator_, &pool_));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservation(
      pool_, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
}

}  // namespace
