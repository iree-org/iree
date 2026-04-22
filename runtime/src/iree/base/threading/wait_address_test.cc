// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/wait_address.h"

#include <atomic>
#include <thread>
#include <vector>

#include "iree/base/threading/thread.h"
#include "iree/testing/gtest.h"

namespace {

TEST(WaitAddressTest, WakeNoWaiters) {
  iree_atomic_int32_t value;
  iree_atomic_store(&value, 0, iree_memory_order_release);
  iree_wait_address_wake_all(&value);
}

TEST(WaitAddressTest, AlreadyChangedReturnsOk) {
  iree_atomic_int32_t value;
  iree_atomic_store(&value, 1, iree_memory_order_release);
  EXPECT_EQ(iree_wait_address_wait_int32(&value, 0, IREE_TIME_INFINITE_FUTURE),
            IREE_STATUS_OK);
}

TEST(WaitAddressTest, ImmediateDeadlineReturnsDeadlineExceeded) {
  iree_atomic_int32_t value;
  iree_atomic_store(&value, 0, iree_memory_order_release);
  EXPECT_EQ(iree_wait_address_wait_int32(&value, 0, IREE_TIME_INFINITE_PAST),
            IREE_STATUS_DEADLINE_EXCEEDED);
}

TEST(WaitAddressTest, WakeAllReleasesWaiters) {
  iree_atomic_int32_t value;
  iree_atomic_store(&value, 0, iree_memory_order_release);
  std::atomic<int32_t> waiters_started{0};
  std::atomic<int32_t> waiters_completed{0};
  constexpr int32_t kWaiterCount = 4;

  std::vector<std::thread> waiters;
  for (int32_t i = 0; i < kWaiterCount; ++i) {
    waiters.emplace_back([&]() {
      waiters_started.fetch_add(1, std::memory_order_acq_rel);
      while (iree_atomic_load(&value, iree_memory_order_acquire) == 0) {
        iree_wait_address_wait_int32(&value, 0, IREE_TIME_INFINITE_FUTURE);
      }
      waiters_completed.fetch_add(1, std::memory_order_acq_rel);
    });
  }

  while (waiters_started.load(std::memory_order_acquire) < kWaiterCount) {
    iree_thread_yield();
  }
  iree_atomic_store(&value, 1, iree_memory_order_release);
  iree_wait_address_wake_all(&value);

  for (std::thread& waiter : waiters) {
    waiter.join();
  }
  EXPECT_EQ(waiters_completed.load(std::memory_order_acquire), kWaiterCount);
}

}  // namespace
