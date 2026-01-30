// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

namespace {

//==============================================================================
// Lifecycle benchmarks
//==============================================================================

void BM_NotificationCreateDelete(benchmark::State& state) {
  for (auto _ : state) {
    iree_notification_t notification;
    iree_notification_initialize(&notification);
    benchmark::DoNotOptimize(notification);
    iree_notification_deinitialize(&notification);
  }
}
BENCHMARK(BM_NotificationCreateDelete);

//==============================================================================
// Post benchmarks
//==============================================================================

void BM_NotificationPostNoWaiters(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  for (auto _ : state) {
    iree_notification_post(&notification, IREE_ALL_WAITERS);
  }
  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationPostNoWaiters);

void BM_NotificationPostSingleNoWaiters(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  for (auto _ : state) {
    iree_notification_post(&notification, 1);
  }
  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationPostSingleNoWaiters);

//==============================================================================
// Await benchmarks
//==============================================================================

void BM_NotificationAwaitImmediate(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  for (auto _ : state) {
    // Condition always true - should return immediately without waiting.
    iree_notification_await(
        &notification, +[](void*) { return true; }, nullptr,
        iree_immediate_timeout());
  }
  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationAwaitImmediate);

void BM_NotificationAwaitTimeoutImmediate(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  for (auto _ : state) {
    // Condition always false with immediate timeout - should return
    // immediately.
    iree_notification_await(
        &notification, +[](void*) { return false; }, nullptr,
        iree_immediate_timeout());
  }
  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationAwaitTimeoutImmediate);

//==============================================================================
// Prepare/commit benchmarks
//==============================================================================

void BM_NotificationPrepareCancel(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  for (auto _ : state) {
    iree_wait_token_t token = iree_notification_prepare_wait(&notification);
    benchmark::DoNotOptimize(token);
    iree_notification_cancel_wait(&notification);
  }
  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationPrepareCancel);

void BM_NotificationPreparePostCommit(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  for (auto _ : state) {
    iree_wait_token_t token = iree_notification_prepare_wait(&notification);
    iree_notification_post(&notification, IREE_ALL_WAITERS);
    bool result =
        iree_notification_commit_wait(&notification, token, 0, iree_time_now());
    benchmark::DoNotOptimize(result);
  }
  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationPreparePostCommit);

//==============================================================================
// Wake latency benchmark
//==============================================================================

// Measures the latency of a single post waking a single waiter.
void BM_NotificationWakeLatency(benchmark::State& state) {
  iree_notification_t notification;
  iree_notification_initialize(&notification);
  std::atomic<bool> ready{false};
  std::atomic<bool> stop{false};

  std::thread waiter([&]() {
    while (!stop.load(std::memory_order_acquire)) {
      iree_notification_await(
          &notification,
          +[](void* arg) {
            auto* ready = static_cast<std::atomic<bool>*>(arg);
            return ready->load(std::memory_order_acquire);
          },
          &ready, iree_make_timeout_ms(100));
      ready.store(false, std::memory_order_release);
    }
  });

  for (auto _ : state) {
    ready.store(true, std::memory_order_release);
    iree_notification_post(&notification, IREE_ALL_WAITERS);
    // Wait for waiter to consume.
    while (ready.load(std::memory_order_acquire)) {
      iree_thread_yield();
    }
  }

  stop.store(true, std::memory_order_release);
  ready.store(true, std::memory_order_release);
  iree_notification_post(&notification, IREE_ALL_WAITERS);
  waiter.join();

  iree_notification_deinitialize(&notification);
}
BENCHMARK(BM_NotificationWakeLatency)->UseRealTime();

}  // namespace
