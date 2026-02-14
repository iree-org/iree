// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <thread>

#include "benchmark/benchmark.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

namespace {

//==============================================================================
// Thread creation benchmarks
//==============================================================================

void BM_ThreadCreate(benchmark::State& state) {
  for (auto _ : state) {
    std::atomic<bool> completed{false};

    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));

    iree_thread_t* thread = nullptr;
    iree_thread_create(
        +[](void* arg) -> int {
          auto* completed = static_cast<std::atomic<bool>*>(arg);
          completed->store(true, std::memory_order_release);
          return 0;
        },
        &completed, params, iree_allocator_system(), &thread);

    // Wait for thread to complete.
    while (!completed.load(std::memory_order_acquire)) {
      iree_thread_yield();
    }

    iree_thread_release(thread);
  }
}
BENCHMARK(BM_ThreadCreate)->UseRealTime();

void BM_ThreadCreateRelease(benchmark::State& state) {
  // Measures thread create + release (which internally joins).
  for (auto _ : state) {
    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));

    iree_thread_t* thread = nullptr;
    iree_thread_create(
        +[](void*) -> int { return 0; }, nullptr, params,
        iree_allocator_system(), &thread);

    // Release internally joins/waits for completion.
    iree_thread_release(thread);
  }
}
BENCHMARK(BM_ThreadCreateRelease)->UseRealTime();

void BM_ThreadCreateSuspendedResume(benchmark::State& state) {
  for (auto _ : state) {
    std::atomic<bool> completed{false};

    iree_thread_create_params_t params;
    memset(&params, 0, sizeof(params));
    params.create_suspended = true;

    iree_thread_t* thread = nullptr;
    iree_thread_create(
        +[](void* arg) -> int {
          auto* completed = static_cast<std::atomic<bool>*>(arg);
          completed->store(true, std::memory_order_release);
          return 0;
        },
        &completed, params, iree_allocator_system(), &thread);

    iree_thread_resume(thread);

    while (!completed.load(std::memory_order_acquire)) {
      iree_thread_yield();
    }

    iree_thread_release(thread);
  }
}
BENCHMARK(BM_ThreadCreateSuspendedResume)->UseRealTime();

//==============================================================================
// std::thread comparison benchmarks
//==============================================================================

void BM_StdThreadCreate(benchmark::State& state) {
  for (auto _ : state) {
    std::thread t([]() {});
    t.join();
  }
}
BENCHMARK(BM_StdThreadCreate)->UseRealTime();

void BM_StdThreadCreateWithWork(benchmark::State& state) {
  for (auto _ : state) {
    std::atomic<bool> completed{false};
    std::thread t(
        [&completed]() { completed.store(true, std::memory_order_release); });
    while (!completed.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    t.join();
  }
}
BENCHMARK(BM_StdThreadCreateWithWork)->UseRealTime();

//==============================================================================
// Thread yield benchmark
//==============================================================================

void BM_ThreadYield(benchmark::State& state) {
  for (auto _ : state) {
    iree_thread_yield();
  }
}
BENCHMARK(BM_ThreadYield);

void BM_StdThreadYield(benchmark::State& state) {
  for (auto _ : state) {
    std::this_thread::yield();
  }
}
BENCHMARK(BM_StdThreadYield);

//==============================================================================
// Priority override benchmarks
//==============================================================================

void BM_ThreadPriorityOverride(benchmark::State& state) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  std::atomic<bool> keep_running{true};

  iree_thread_t* thread = nullptr;
  iree_thread_create(
      +[](void* arg) -> int {
        auto* keep_running = static_cast<std::atomic<bool>*>(arg);
        while (keep_running->load(std::memory_order_acquire)) {
          iree_thread_yield();
        }
        return 0;
      },
      &keep_running, params, iree_allocator_system(), &thread);

  for (auto _ : state) {
    iree_thread_override_t* override_token =
        iree_thread_priority_class_override_begin(
            thread, IREE_THREAD_PRIORITY_CLASS_HIGH);
    iree_thread_override_end(override_token);
  }

  keep_running.store(false, std::memory_order_release);
  // Release internally joins/waits for thread completion.
  iree_thread_release(thread);
}
BENCHMARK(BM_ThreadPriorityOverride)->UseRealTime();

//==============================================================================
// Affinity request benchmark
//==============================================================================

void BM_ThreadRequestAffinity(benchmark::State& state) {
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  std::atomic<bool> keep_running{true};

  iree_thread_t* thread = nullptr;
  iree_thread_create(
      +[](void* arg) -> int {
        auto* keep_running = static_cast<std::atomic<bool>*>(arg);
        while (keep_running->load(std::memory_order_acquire)) {
          iree_thread_yield();
        }
        return 0;
      },
      &keep_running, params, iree_allocator_system(), &thread);

  iree_thread_affinity_t affinity;
  iree_thread_affinity_set_group_any(0, &affinity);

  for (auto _ : state) {
    iree_thread_request_affinity(thread, affinity);
  }

  keep_running.store(false, std::memory_order_release);
  // Release internally joins/waits for thread completion.
  iree_thread_release(thread);
}
BENCHMARK(BM_ThreadRequestAffinity)->UseRealTime();

}  // namespace
