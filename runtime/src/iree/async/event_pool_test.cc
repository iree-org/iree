// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event_pool.h"

#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class EventPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    iree_status_t status = iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      GTEST_SKIP() << "Platform proactor unavailable";
    }
    IREE_ASSERT_OK(status);
  }

  void TearDown() override {
    iree_async_proactor_release(proactor_);
    proactor_ = nullptr;
  }

  iree_async_proactor_t* proactor_ = nullptr;
};

TEST_F(EventPoolTest, InitializeDeinitializeEmpty) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/0, &pool));
  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, InitializeWithCapacity) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/4, &pool));
  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, AcquireReleaseSingle) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/0, &pool));

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));
  ASSERT_NE(event, nullptr);
  EXPECT_EQ(event->pool, &pool);

  iree_async_event_pool_release(&pool, event);

  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, AcquireFromPreallocated) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/2, &pool));

  // Acquire should come from preallocated pool without growing.
  iree_async_event_t* event1 = nullptr;
  iree_async_event_t* event2 = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event1));
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event2));
  ASSERT_NE(event1, nullptr);
  ASSERT_NE(event2, nullptr);
  EXPECT_NE(event1, event2);

  iree_async_event_pool_release(&pool, event1);
  iree_async_event_pool_release(&pool, event2);

  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, GrowOnDemand) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/0, &pool));

  // Multiple acquires should trigger growth each time.
  std::vector<iree_async_event_t*> events;
  for (int i = 0; i < 5; ++i) {
    iree_async_event_t* event = nullptr;
    IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));
    ASSERT_NE(event, nullptr);
    events.push_back(event);
  }

  // All events should be unique.
  for (size_t i = 0; i < events.size(); ++i) {
    for (size_t j = i + 1; j < events.size(); ++j) {
      EXPECT_NE(events[i], events[j]);
    }
  }

  for (auto* event : events) {
    iree_async_event_pool_release(&pool, event);
  }

  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, ReuseAfterRelease) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/0, &pool));

  // Acquire, release, acquire again.
  iree_async_event_t* event1 = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event1));
  ASSERT_NE(event1, nullptr);

  iree_async_event_pool_release(&pool, event1);

  iree_async_event_t* event2 = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event2));
  ASSERT_NE(event2, nullptr);

  // Should get the same event back (LIFO from return stack, then migration).
  EXPECT_EQ(event1, event2);

  iree_async_event_pool_release(&pool, event2);

  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, EventSignalAfterAcquire) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/1, &pool));

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));

  // Event should be usable for signaling.
  IREE_ASSERT_OK(iree_async_event_set(event));

  iree_async_event_pool_release(&pool, event);

  // Acquire again - reset should have been called, event should be fresh.
  iree_async_event_t* event2 = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event2));
  EXPECT_EQ(event, event2);

  // Can signal again without issues.
  IREE_ASSERT_OK(iree_async_event_set(event2));

  iree_async_event_pool_release(&pool, event2);
  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, ConcurrentAcquire) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/0, &pool));

  constexpr int kThreadCount = 4;
  constexpr int kAcquiresPerThread = 10;

  std::vector<std::thread> threads;
  std::vector<std::vector<iree_async_event_t*>> thread_events(kThreadCount);

  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&pool, &thread_events, t, kAcquiresPerThread]() {
      for (int i = 0; i < kAcquiresPerThread; ++i) {
        iree_async_event_t* event = nullptr;
        iree_status_t status = iree_async_event_pool_acquire(&pool, &event);
        if (iree_status_is_ok(status) && event) {
          thread_events[t].push_back(event);
        }
        iree_status_ignore(status);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all acquired events are unique.
  std::vector<iree_async_event_t*> all_events;
  for (const auto& events : thread_events) {
    for (auto* event : events) {
      ASSERT_NE(event, nullptr);
      all_events.push_back(event);
    }
  }

  for (size_t i = 0; i < all_events.size(); ++i) {
    for (size_t j = i + 1; j < all_events.size(); ++j) {
      EXPECT_NE(all_events[i], all_events[j])
          << "Duplicate event at indices " << i << " and " << j;
    }
  }

  // Release all events.
  for (auto* event : all_events) {
    iree_async_event_pool_release(&pool, event);
  }

  iree_async_event_pool_deinitialize(&pool);
}

TEST_F(EventPoolTest, MixedAcquireRelease) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/2, &pool));

  // Simulate realistic usage: acquire some, release some, acquire more.
  iree_async_event_t* e1 = nullptr;
  iree_async_event_t* e2 = nullptr;
  iree_async_event_t* e3 = nullptr;

  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &e1));
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &e2));

  // Release e1, acquire e3 (should get e1 back via migration).
  iree_async_event_pool_release(&pool, e1);

  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &e3));
  EXPECT_EQ(e3, e1);  // Should reuse released event.

  // Release all.
  iree_async_event_pool_release(&pool, e2);
  iree_async_event_pool_release(&pool, e3);

  iree_async_event_pool_deinitialize(&pool);
}

// Test concurrent acquire and release from multiple threads.
// This exercises the lock-free Treiber stack (release) under contention.
TEST_F(EventPoolTest, ConcurrentAcquireRelease) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/8, &pool));

  constexpr int kThreadCount = 4;
  constexpr int kIterationsPerThread = 100;
  std::atomic<bool> start_flag{false};

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&pool, &start_flag, kIterationsPerThread]() {
      // Wait for all threads to be ready.
      while (!start_flag.load(std::memory_order_acquire)) {
      }

      for (int i = 0; i < kIterationsPerThread; ++i) {
        iree_async_event_t* event = nullptr;
        iree_status_t status = iree_async_event_pool_acquire(&pool, &event);
        if (iree_status_is_ok(status) && event) {
          // Briefly hold the event, then release.
          iree_async_event_pool_release(&pool, event);
        }
        iree_status_ignore(status);
      }
    });
  }

  // Start all threads simultaneously.
  start_flag.store(true, std::memory_order_release);

  for (auto& thread : threads) {
    thread.join();
  }

  iree_async_event_pool_deinitialize(&pool);
}

// Test that pool correctly tracks all events for cleanup even after many
// acquire/release cycles.
TEST_F(EventPoolTest, AllEventsTrackedAfterChurn) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/0, &pool));

  // Acquire many events to trigger growth.
  constexpr int kEventCount = 20;
  std::vector<iree_async_event_t*> events;
  for (int i = 0; i < kEventCount; ++i) {
    iree_async_event_t* event = nullptr;
    IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));
    ASSERT_NE(event, nullptr);
    events.push_back(event);
  }

  // Release all events.
  for (auto* event : events) {
    iree_async_event_pool_release(&pool, event);
  }
  events.clear();

  // Acquire again - should reuse existing events.
  for (int i = 0; i < kEventCount; ++i) {
    iree_async_event_t* event = nullptr;
    IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));
    ASSERT_NE(event, nullptr);
    events.push_back(event);
  }

  // Release all events again.
  for (auto* event : events) {
    iree_async_event_pool_release(&pool, event);
  }

  // Deinitialize should clean up all events without leaks.
  // (ASAN will catch leaks if pool_all_next linkage is broken)
  iree_async_event_pool_deinitialize(&pool);
}

// Test that events can be signaled multiple times between acquire/release
// without issues (tests eventfd accumulation behavior).
TEST_F(EventPoolTest, MultipleSignalsBeforeRelease) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/1, &pool));

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));
  ASSERT_NE(event, nullptr);

  // Signal the event multiple times.
  IREE_ASSERT_OK(iree_async_event_set(event));
  IREE_ASSERT_OK(iree_async_event_set(event));
  IREE_ASSERT_OK(iree_async_event_set(event));

  iree_async_event_pool_release(&pool, event);
  iree_async_event_pool_deinitialize(&pool);
}

// Test rapid acquire/release cycles to stress the migration path.
TEST_F(EventPoolTest, RapidAcquireReleaseCycles) {
  iree_async_event_pool_t pool;
  IREE_ASSERT_OK(iree_async_event_pool_initialize(
      proactor_, iree_allocator_system(), /*initial_capacity=*/1, &pool));

  // Rapid cycles should trigger migration from return_stack to acquire_stack.
  for (int i = 0; i < 100; ++i) {
    iree_async_event_t* event = nullptr;
    IREE_ASSERT_OK(iree_async_event_pool_acquire(&pool, &event));
    ASSERT_NE(event, nullptr);
    iree_async_event_pool_release(&pool, event);
  }

  iree_async_event_pool_deinitialize(&pool);
}

}  // namespace
