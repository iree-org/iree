// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct HostQueueTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_async_proactor_t* proactor;

  static void SetUpTestSuite() {
    IREE_TRACE_SCOPE();
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_ignore(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), host_allocator, &proactor));
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_async_proactor_release(proactor);
    proactor = NULL;
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  // Simulates GPU completion of N submissions by storing the appropriate
  // signal value (INITIAL - N).
  void SimulateCompletions(iree_hal_amdgpu_host_queue_t* queue, uint64_t n) {
    hsa_signal_value_t target_value =
        (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - n);
    iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), queue->epoch.signal,
                                    target_value);
  }
};
iree_allocator_t HostQueueTest::host_allocator;
iree_hal_amdgpu_libhsa_t HostQueueTest::libhsa;
iree_async_proactor_t* HostQueueTest::proactor = NULL;

TEST_F(HostQueueTest, InitDeinit) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa,
      IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY, host_allocator, &queue));
  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

TEST_F(HostQueueTest, InvalidCapacity) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_host_queue_initialize(
                            /*device=*/NULL, proactor, &libhsa,
                            /*notification_capacity=*/100,  // not power of two
                            host_allocator, &queue));
}

TEST_F(HostQueueTest, DrainEmpty) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa,
      IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY, host_allocator, &queue));

  // Drain with no submissions and no notifications should be a no-op.
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 0u);

  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

TEST_F(HostQueueTest, SingleNotification) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa,
      IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY, host_allocator, &queue));

  // Create a semaphore to signal.
  iree_async_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, /*initial_value=*/0,
      IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, host_allocator,
      &semaphore));

  // Simulate a submission that signals the semaphore to value 1.
  uint64_t epoch = iree_hal_amdgpu_host_queue_advance_epoch(&queue);
  EXPECT_EQ(epoch, 0u);
  iree_hal_amdgpu_host_queue_push_notification(&queue, epoch, semaphore, 1,
                                               /*frontier=*/NULL);

  // Drain before completion: nothing should happen.
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 0u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);

  // Simulate GPU completing epoch 0.
  SimulateCompletions(&queue, 1);

  // Drain should now signal the semaphore.
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  // Draining again should be a no-op.
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 0u);

  iree_async_semaphore_release(semaphore);
  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

TEST_F(HostQueueTest, MultipleNotificationsPerSubmission) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa,
      IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY, host_allocator, &queue));

  iree_async_semaphore_t* semaphore_a = NULL;
  iree_async_semaphore_t* semaphore_b = NULL;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      host_allocator, &semaphore_a));
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      host_allocator, &semaphore_b));

  // One submission signals two semaphores.
  uint64_t epoch = iree_hal_amdgpu_host_queue_advance_epoch(&queue);
  iree_hal_amdgpu_host_queue_push_notification(&queue, epoch, semaphore_a, 1,
                                               NULL);
  iree_hal_amdgpu_host_queue_push_notification(&queue, epoch, semaphore_b, 5,
                                               NULL);

  SimulateCompletions(&queue, 1);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 2u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore_a), 1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore_b), 5u);

  iree_async_semaphore_release(semaphore_b);
  iree_async_semaphore_release(semaphore_a);
  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

TEST_F(HostQueueTest, SparseEpochs) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa,
      IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY, host_allocator, &queue));

  iree_async_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      host_allocator, &semaphore));

  // Epoch 0: no notification (non-signaling submission).
  iree_hal_amdgpu_host_queue_advance_epoch(&queue);

  // Epoch 1: signals semaphore.
  uint64_t epoch = iree_hal_amdgpu_host_queue_advance_epoch(&queue);
  EXPECT_EQ(epoch, 1u);
  iree_hal_amdgpu_host_queue_push_notification(&queue, epoch, semaphore, 3,
                                               NULL);

  // Epoch 2: no notification.
  iree_hal_amdgpu_host_queue_advance_epoch(&queue);

  // Complete all 3 submissions.
  SimulateCompletions(&queue, 3);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 3u);

  iree_async_semaphore_release(semaphore);
  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

TEST_F(HostQueueTest, IncrementalDrain) {
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa,
      IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY, host_allocator, &queue));

  iree_async_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      host_allocator, &semaphore));

  // Submit 3 submissions, each signaling the same semaphore to increasing
  // values.
  for (uint64_t i = 0; i < 3; ++i) {
    uint64_t epoch = iree_hal_amdgpu_host_queue_advance_epoch(&queue);
    iree_hal_amdgpu_host_queue_push_notification(&queue, epoch, semaphore,
                                                 i + 1, NULL);
  }

  // Complete only epoch 0.
  SimulateCompletions(&queue, 1);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  // Complete epochs 1 and 2.
  SimulateCompletions(&queue, 3);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), 2u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 3u);

  iree_async_semaphore_release(semaphore);
  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

TEST_F(HostQueueTest, RingWrapAround) {
  // Use a small ring to test wrap-around.
  const uint32_t capacity = 4;
  iree_hal_amdgpu_host_queue_t queue;
  IREE_ASSERT_OK(iree_hal_amdgpu_host_queue_initialize(
      /*device=*/NULL, proactor, &libhsa, capacity, host_allocator, &queue));

  iree_async_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      host_allocator, &semaphore));

  // Fill the ring, drain, repeat — exercises wrap-around.
  for (int round = 0; round < 3; ++round) {
    for (uint32_t i = 0; i < capacity; ++i) {
      uint64_t epoch = iree_hal_amdgpu_host_queue_advance_epoch(&queue);
      iree_hal_amdgpu_host_queue_push_notification(
          &queue, epoch, semaphore, (round * capacity) + i + 1, NULL);
    }
    SimulateCompletions(&queue, queue.epoch.next_submission);
    EXPECT_EQ(iree_hal_amdgpu_host_queue_drain(&queue), capacity);
  }

  // Final semaphore value should be the last value signaled.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 3u * capacity);

  iree_async_semaphore_release(semaphore);
  iree_hal_amdgpu_host_queue_deinitialize(&queue);
}

}  // namespace
}  // namespace iree::hal::amdgpu
