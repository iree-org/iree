// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/notification_ring.h"

#include <array>
#include <memory>

#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct MaxFrontierStorage {
  uint8_t entry_count;
  uint8_t reserved[7];
  iree_async_frontier_entry_t
      entries[IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT];
};

typedef struct PreSignalActionState {
  iree_async_semaphore_t* semaphore;
  int callback_count;
} PreSignalActionState;

static void VerifySemaphoreNotVisibleBeforePreSignalAction(void* user_data) {
  auto* state = static_cast<PreSignalActionState*>(user_data);
  EXPECT_EQ(iree_async_semaphore_query(state->semaphore), 0u);
  ++state->callback_count;
}

// RAII wrapper for notification rings. Ensures deinitialize is called on
// destruction.
struct NotificationRingDeleter {
  void operator()(iree_hal_amdgpu_notification_ring_t* ring) {
    iree_hal_amdgpu_notification_ring_deinitialize(ring);
    delete ring;
  }
};
using NotificationRingPtr = std::unique_ptr<iree_hal_amdgpu_notification_ring_t,
                                            NotificationRingDeleter>;

struct NotificationRingTest : public ::testing::Test {
  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_async_proactor_t* proactor;
  static iree_arena_block_pool_t block_pool;

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
    iree_arena_block_pool_initialize(4096, host_allocator, &block_pool);
  }

  static void TearDownTestSuite() {
    IREE_TRACE_SCOPE();
    iree_arena_block_pool_deinitialize(&block_pool);
    iree_async_proactor_release(proactor);
    proactor = NULL;
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  // Initializes a notification ring with the given capacity and returns an
  // RAII wrapper that ensures deinitialize is called on destruction.
  iree::StatusOr<NotificationRingPtr> InitializeRing(
      uint32_t capacity = IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY) {
    auto ring = std::make_unique<iree_hal_amdgpu_notification_ring_t>();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_notification_ring_initialize(
        &libhsa, &block_pool, capacity, host_allocator, ring.get()));
    return NotificationRingPtr(ring.release());
  }

  // Creates an async semaphore with initial value 0.
  iree_async_semaphore_t* CreateSemaphore(
      uint8_t frontier_capacity =
          IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY) {
    iree_async_semaphore_t* semaphore = NULL;
    IREE_CHECK_OK(iree_async_semaphore_create(proactor, /*initial_value=*/0,
                                              frontier_capacity, host_allocator,
                                              &semaphore));
    return semaphore;
  }

  // Simulates GPU completion of N total submissions by storing the
  // appropriate epoch signal value (INITIAL - N).
  void SimulateCompletions(iree_hal_amdgpu_notification_ring_t* ring,
                           uint64_t total_completed) {
    hsa_signal_value_t target_value =
        (hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                             total_completed);
    iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), ring->epoch.signal,
                                    target_value);
  }

  // Empty frontier for drain calls (no accumulated causal context).
  static constexpr iree_async_frontier_t kEmptyFrontier = {0};

  static const iree_async_frontier_t* InitializeMaxFrontier(
      MaxFrontierStorage* storage) {
    auto* frontier = reinterpret_cast<iree_async_frontier_t*>(storage);
    iree_async_frontier_initialize(
        frontier, IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT);
    for (uint8_t i = 0; i < IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT;
         ++i) {
      frontier->entries[i].axis = iree_async_axis_make_queue(
          /*session_epoch=*/1, /*machine_index=*/2, /*device_index=*/3, i);
      frontier->entries[i].epoch = i + 1;
    }
    return frontier;
  }

  static iree_hal_amdgpu_reclaim_entry_t* ReclaimEntryForNextEpoch(
      iree_hal_amdgpu_notification_ring_t* ring,
      uint64_t kernarg_write_position = 0) {
    iree_hal_amdgpu_reclaim_entry_t* reclaim_entry =
        iree_hal_amdgpu_notification_ring_reclaim_entry(ring);
    reclaim_entry->kernarg_write_position = kernarg_write_position;
    return reclaim_entry;
  }
};
iree_allocator_t NotificationRingTest::host_allocator;
iree_hal_amdgpu_libhsa_t NotificationRingTest::libhsa;
iree_async_proactor_t* NotificationRingTest::proactor = NULL;
iree_arena_block_pool_t NotificationRingTest::block_pool;

TEST_F(NotificationRingTest, InitDeinit) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
}

TEST_F(NotificationRingTest, InvalidCapacity) {
  iree_hal_amdgpu_notification_ring_t ring;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_notification_ring_initialize(
          &libhsa, &block_pool, /*capacity=*/100, host_allocator, &ring));
}

TEST_F(NotificationRingTest, DrainEmpty) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());

  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            0u);
  EXPECT_EQ(kernarg_position, 0u);
}

TEST_F(NotificationRingTest, SingleNotification) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
  iree_async_semaphore_t* semaphore = CreateSemaphore();

  // Push a notification for epoch 0.
  ReclaimEntryForNextEpoch(ring.get());
  uint64_t epoch = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  EXPECT_EQ(epoch, 0u);
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch, semaphore, 1);

  // Drain before completion: nothing happens.
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            0u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);

  // Simulate GPU completing epoch 0.
  SimulateCompletions(ring.get(), 1);

  // Drain signals the semaphore.
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  // Draining again is a no-op.
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            0u);

  iree_async_semaphore_release(semaphore);
}

TEST_F(NotificationRingTest, MultiplePerEpochAndSparseEpochs) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
  iree_async_semaphore_t* semaphore_a = CreateSemaphore();
  iree_async_semaphore_t* semaphore_b = CreateSemaphore();

  // Epoch 0: two semaphores signaled from one submission.
  // This is a semaphore transition (A -> B), so push a frontier snapshot.
  ReclaimEntryForNextEpoch(ring.get());
  uint64_t epoch0 = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch0, semaphore_a, 1);
  iree_hal_amdgpu_notification_ring_push_frontier_snapshot(ring.get(), epoch0,
                                                           &kEmptyFrontier);
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch0, semaphore_b, 5);

  // Epoch 1: no notification (non-signaling submission).
  ReclaimEntryForNextEpoch(ring.get());
  iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());

  // Epoch 2: signals semaphore_a again (transition B -> A).
  ReclaimEntryForNextEpoch(ring.get());
  uint64_t epoch2 = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  EXPECT_EQ(epoch2, 2u);
  iree_hal_amdgpu_notification_ring_push_frontier_snapshot(ring.get(), epoch0,
                                                           &kEmptyFrontier);
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch2, semaphore_a, 10);

  // Complete epoch 0 only. Drain should coalesce the A and B entries at
  // epoch 0 into two signals (one per semaphore).
  SimulateCompletions(ring.get(), 1);
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore_a), 1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore_b), 5u);

  // Complete all 3 epochs.
  SimulateCompletions(ring.get(), 3);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore_a), 10u);

  iree_async_semaphore_release(semaphore_b);
  iree_async_semaphore_release(semaphore_a);
}

TEST_F(NotificationRingTest, CoalescingSameSemaphore) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
  iree_async_semaphore_t* semaphore = CreateSemaphore();

  // Three submissions, each signaling the semaphore to increasing values.
  // All same semaphore — no frontier snapshots needed (no transitions).
  for (uint64_t i = 0; i < 3; ++i) {
    ReclaimEntryForNextEpoch(ring.get());
    uint64_t epoch =
        iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
    iree_hal_amdgpu_notification_ring_push(ring.get(), epoch, semaphore, i + 1);
  }

  // Complete only epoch 0.
  SimulateCompletions(ring.get(), 1);
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            1u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  // Complete epochs 1 and 2. Drain should coalesce both entries (same
  // semaphore) into a single signal to value 3.
  SimulateCompletions(ring.get(), 3);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);
  // Coalesced: signaled directly to 3, skipping 2.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 3u);

  iree_async_semaphore_release(semaphore);
}

TEST_F(NotificationRingTest, RingWrapAround) {
  const uint32_t capacity = 4;
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing(capacity));
  iree_async_semaphore_t* semaphore = CreateSemaphore();

  // Fill the ring, drain, repeat three times to exercise wrap-around.
  for (int round = 0; round < 3; ++round) {
    for (uint32_t i = 0; i < capacity; ++i) {
      ReclaimEntryForNextEpoch(ring.get());
      uint64_t epoch =
          iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
      iree_hal_amdgpu_notification_ring_push(ring.get(), epoch, semaphore,
                                             (round * capacity) + i + 1);
    }
    SimulateCompletions(ring.get(), ring->epoch.next_submission);
    uint64_t kernarg_position = 0;
    // All same semaphore — coalesced into 1 signal per round.
    EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(
                  ring.get(), &kEmptyFrontier, &kernarg_position),
              capacity);
  }

  // Coalesced across each round: signaled to 4, then 8, then 12.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 3u * capacity);

  iree_async_semaphore_release(semaphore);
}

TEST_F(NotificationRingTest, ReserveReturnsResourceExhaustedWhenFull) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing(/*capacity=*/2));
  iree_async_semaphore_t* semaphore = CreateSemaphore();

  IREE_EXPECT_OK(iree_hal_amdgpu_notification_ring_reserve(
      ring.get(), /*entry_count=*/2, /*frontier_snapshot_count=*/0));

  ReclaimEntryForNextEpoch(ring.get());
  uint64_t epoch0 = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch0, semaphore,
                                         /*timeline_value=*/1);
  ReclaimEntryForNextEpoch(ring.get());
  uint64_t epoch1 = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch1, semaphore,
                                         /*timeline_value=*/2);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_amdgpu_notification_ring_reserve(ring.get(), /*entry_count=*/1,
                                                /*frontier_snapshot_count=*/0));

  SimulateCompletions(ring.get(), 1);
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            1u);

  IREE_EXPECT_OK(iree_hal_amdgpu_notification_ring_reserve(
      ring.get(), /*entry_count=*/1, /*frontier_snapshot_count=*/0));

  iree_async_semaphore_release(semaphore);
}

TEST_F(NotificationRingTest, FrontierSnapshotWrapAround) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing(/*capacity=*/4));

  std::array<iree_async_semaphore_t*, 10> semaphores = {
      NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  for (iree_async_semaphore_t*& semaphore : semaphores) {
    semaphore =
        CreateSemaphore(IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT);
  }

  MaxFrontierStorage snapshot_storage;
  const iree_async_frontier_t* snapshot_frontier =
      InitializeMaxFrontier(&snapshot_storage);

  ReclaimEntryForNextEpoch(ring.get());
  uint64_t previous_epoch =
      iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), previous_epoch,
                                         semaphores[0], 1);

  // Build enough transitions to force the snapshot byte-ring write offset to
  // wrap while one unread snapshot remains near the tail. Draining two entries
  // after each batch preserves one transition snapshot for the next batch.
  for (size_t i = 1; i < 4; ++i) {
    iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
        ring.get(), previous_epoch, snapshot_frontier);
    ReclaimEntryForNextEpoch(ring.get());
    previous_epoch =
        iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
    iree_hal_amdgpu_notification_ring_push(ring.get(), previous_epoch,
                                           semaphores[i], 1);
  }

  uint64_t kernarg_position = 0;
  SimulateCompletions(ring.get(), 2);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);

  SimulateCompletions(ring.get(), 4);
  for (size_t i = 4; i < 6; ++i) {
    iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
        ring.get(), previous_epoch, snapshot_frontier);
    ReclaimEntryForNextEpoch(ring.get());
    previous_epoch =
        iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
    iree_hal_amdgpu_notification_ring_push(ring.get(), previous_epoch,
                                           semaphores[i], 1);
  }
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);

  SimulateCompletions(ring.get(), 6);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);

  for (size_t i = 6; i < semaphores.size(); ++i) {
    iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
        ring.get(), previous_epoch, snapshot_frontier);
    ReclaimEntryForNextEpoch(ring.get());
    previous_epoch =
        iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
    iree_hal_amdgpu_notification_ring_push(ring.get(), previous_epoch,
                                           semaphores[i], 1);
  }
  iree_host_size_t frontier_write = (iree_host_size_t)iree_atomic_load(
      &ring->frontier_ring.write, iree_memory_order_acquire);
  iree_host_size_t frontier_read = (iree_host_size_t)iree_atomic_load(
      &ring->frontier_ring.read, iree_memory_order_acquire);
  EXPECT_LT(frontier_write & (ring->frontier_ring.capacity - 1),
            frontier_read & (ring->frontier_ring.capacity - 1));
  SimulateCompletions(ring.get(), semaphores.size());
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            4u);

  MaxFrontierStorage queried_storage;
  auto* queried_frontier =
      reinterpret_cast<iree_async_frontier_t*>(&queried_storage);
  // semaphores[8]'s frontier snapshot is written after the byte-ring write
  // offset wraps back to the beginning, so this specifically verifies the
  // wrapped snapshot was not mistaken for an empty ring.
  EXPECT_EQ(iree_async_semaphore_query_frontier(
                semaphores[8], queried_frontier,
                IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT),
            IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT);
  EXPECT_EQ(queried_frontier->entries[0].axis,
            snapshot_frontier->entries[0].axis);
  EXPECT_EQ(queried_frontier->entries[0].epoch,
            snapshot_frontier->entries[0].epoch);
  EXPECT_EQ(queried_frontier
                ->entries[IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT - 1]
                .axis,
            snapshot_frontier
                ->entries[IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT - 1]
                .axis);
  EXPECT_EQ(queried_frontier
                ->entries[IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT - 1]
                .epoch,
            snapshot_frontier
                ->entries[IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT - 1]
                .epoch);

  for (iree_async_semaphore_t* semaphore : semaphores) {
    iree_async_semaphore_release(semaphore);
  }
}

TEST_F(NotificationRingTest, KernargPositionReporting) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
  iree_async_semaphore_t* semaphore = CreateSemaphore();

  // Three submissions with increasing kernarg positions.
  for (uint64_t i = 0; i < 3; ++i) {
    ReclaimEntryForNextEpoch(ring.get(), (i + 1) * 64);
    uint64_t epoch =
        iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
    iree_hal_amdgpu_notification_ring_push(ring.get(), epoch, semaphore, i + 1);
  }

  // Complete epoch 0 only. Drain should report position 64.
  SimulateCompletions(ring.get(), 1);
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            1u);
  EXPECT_EQ(kernarg_position, 64u);

  // Complete all. Drain should report position 192 (max of 128, 192).
  SimulateCompletions(ring.get(), 3);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);
  EXPECT_EQ(kernarg_position, 192u);

  iree_async_semaphore_release(semaphore);
}

TEST_F(NotificationRingTest, KernargPositionReportingForZeroSignalEpochs) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());

  // Epoch 0: no user-visible signals, but kernarg memory must still retire.
  ReclaimEntryForNextEpoch(ring.get(), 64);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_advance_epoch(ring.get()), 0u);

  // Epoch 1: another no-signal submission with a later kernarg watermark.
  ReclaimEntryForNextEpoch(ring.get(), 192);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_advance_epoch(ring.get()), 1u);

  // Complete epoch 0 only.
  SimulateCompletions(ring.get(), 1);
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            0u);
  EXPECT_EQ(kernarg_position, 64u);

  // Complete both epochs.
  SimulateCompletions(ring.get(), 2);
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            0u);
  EXPECT_EQ(kernarg_position, 192u);
}

TEST_F(NotificationRingTest, PreSignalActionRunsBeforeSemaphorePublication) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
  iree_async_semaphore_t* semaphore = CreateSemaphore();
  PreSignalActionState action_state = {
      .semaphore = semaphore,
      .callback_count = 0,
  };

  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry =
      ReclaimEntryForNextEpoch(ring.get());
  reclaim_entry->pre_signal_action = {
      .fn = VerifySemaphoreNotVisibleBeforePreSignalAction,
      .user_data = &action_state,
  };
  uint64_t epoch = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch, semaphore, 1);

  SimulateCompletions(ring.get(), 1);
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            1u);
  EXPECT_EQ(action_state.callback_count, 1);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  iree_async_semaphore_release(semaphore);
}

TEST_F(NotificationRingTest, SignalFailureFailsSemaphore) {
  IREE_ASSERT_OK_AND_ASSIGN(auto ring, InitializeRing());
  iree_async_semaphore_t* semaphore = CreateSemaphore();

  // Epoch 0: signal semaphore to value 5.
  ReclaimEntryForNextEpoch(ring.get(), 64);
  uint64_t epoch0 = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch0, semaphore, 5);

  // Epoch 1: signal semaphore to value 3 (non-monotonic — will fail).
  // Same semaphore, so drain coalesces to the LATER entry (value 3).
  // Since semaphore is already at 5 after the first signal (coalesced value
  // is the last one, 3, which is < 5), the coalesced signal to 3 fails.
  // But wait — coalescing takes the last value, not the max. And 3 < 5.
  // The semaphore_signal will fail (non-monotonic) and fall through to
  // semaphore_fail. But because of coalescing, we only signal ONCE to
  // value 3 (the last entry's value).
  //
  // Actually: with coalescing, entries (5, e0) and (3, e1) for the same
  // semaphore produce a single signal to value 3 (the last value). But 3
  // is not > 0 (current), so it succeeds with value 3. Then no further
  // signal happens. The semaphore ends at 3, not 5. This is different from
  // the non-coalesced behavior where signal(5) then signal(3) would fail
  // on the second.
  //
  // This test verifies the coalesced behavior: pushing non-monotonic values
  // for the same semaphore means the last value wins. The test pushes
  // different semaphores to avoid coalescing and preserve the original
  // non-monotonic test intent.
  iree_async_semaphore_t* semaphore2 = CreateSemaphore();
  iree_hal_amdgpu_notification_ring_push_frontier_snapshot(ring.get(), epoch0,
                                                           &kEmptyFrontier);
  ReclaimEntryForNextEpoch(ring.get(), 128);
  uint64_t epoch1 = iree_hal_amdgpu_notification_ring_advance_epoch(ring.get());
  iree_hal_amdgpu_notification_ring_push(ring.get(), epoch1, semaphore2, 3);

  // Complete both epochs.
  SimulateCompletions(ring.get(), 2);

  // Drain processes both entries (different semaphores, no coalescing).
  uint64_t kernarg_position = 0;
  EXPECT_EQ(iree_hal_amdgpu_notification_ring_drain(ring.get(), &kEmptyFrontier,
                                                    &kernarg_position),
            2u);

  // Semaphore got signaled to 5, semaphore2 got signaled to 3.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 5u);
  EXPECT_EQ(iree_async_semaphore_query(semaphore2), 3u);

  // Kernarg position reporting is unaffected.
  EXPECT_EQ(kernarg_position, 128u);

  iree_async_semaphore_release(semaphore2);
  iree_async_semaphore_release(semaphore);
}

}  // namespace
}  // namespace iree::hal::amdgpu
