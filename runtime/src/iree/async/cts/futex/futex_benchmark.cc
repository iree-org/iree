// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for async futex operations (FUTEX_WAIT/FUTEX_WAKE).
//
// Benchmarks:
//   - FutexWakeNoWaiters: Async FUTEX_WAKE with no actual waiters. Measures
//     pure proactor submission + kernel roundtrip overhead.
//   - FutexCrossThread: Submit FUTEX_WAIT via proactor, background thread
//     wakes with raw syscall. Measures full async wait/wake roundtrip.

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/operations/futex.h"
#include "iree/base/threading/futex.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Futex operation helpers
//===----------------------------------------------------------------------===//

// Initializes a FUTEX_WAIT operation.
static void InitFutexWaitOp(iree_async_futex_wait_operation_t* operation,
                            void* address, uint32_t expected,
                            iree_async_completion_fn_t callback,
                            void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT;
  operation->base.completion_fn = callback;
  operation->base.user_data = user_data;
  operation->futex_address = address;
  operation->expected_value = expected;
  operation->futex_flags =
      IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
}

// Initializes a FUTEX_WAKE operation.
static void InitFutexWakeOp(iree_async_futex_wake_operation_t* operation,
                            void* address, int32_t count,
                            iree_async_completion_fn_t callback,
                            void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE;
  operation->base.completion_fn = callback;
  operation->base.user_data = user_data;
  operation->futex_address = address;
  operation->wake_count = count;
  operation->futex_flags =
      IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
}

//===----------------------------------------------------------------------===//
// Benchmark implementations
//===----------------------------------------------------------------------===//

#if defined(IREE_RUNTIME_USE_FUTEX)

// Async FUTEX_WAKE with no actual waiters.
// Measures pure proactor submission + kernel roundtrip overhead.
// This is the baseline cost of the io_uring futex path.
static void BM_WakeNoWaiters(::benchmark::State& state,
                             const ProactorFactory& factory) {
  auto* context = CreateBenchmarkContext(factory, state);
  if (!context) return;

  if (!RequireCapability(
          context, IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS, state)) {
    DestroyBenchmarkContext(context);
    return;
  }

  std::atomic<uint32_t> futex_word{0};
  iree_async_futex_wake_operation_t wake_op;

  for (auto _ : state) {
    context->Reset();

    InitFutexWakeOp(&wake_op, &futex_word, 1, BenchmarkContext::Callback,
                    context);

    auto start = std::chrono::high_resolution_clock::now();

    iree_status_t status =
        iree_async_proactor_submit_one(context->proactor, &wake_op.base);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Submit failed");
      iree_status_ignore(status);
      break;
    }

    if (!context->SpinPollUntilComplete(1)) {
      state.SkipWithError("Poll timeout");
      break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  DestroyBenchmarkContext(context);
}

// Submit FUTEX_WAIT via proactor, background thread wakes with raw syscall.
// Measures full async wait/wake roundtrip latency.
static void BM_CrossThread(::benchmark::State& state,
                           const ProactorFactory& factory) {
  auto* context = CreateBenchmarkContext(factory, state);
  if (!context) return;

  if (!RequireCapability(
          context, IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS, state)) {
    DestroyBenchmarkContext(context);
    return;
  }

  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> stop_signaler{false};
  std::atomic<uint32_t> wake_signal{0};

  // Background thread that wakes the futex when signaled.
  // Uses futex-based waiting for low-latency signaling.
  std::thread signaler([&]() {
    uint32_t last_signal = 0;
    while (!stop_signaler.load(std::memory_order_acquire)) {
      uint32_t signal = wake_signal.load(std::memory_order_acquire);
      if (signal > last_signal) {
        last_signal = signal;
        futex_word.store(signal, std::memory_order_release);
        iree_futex_wake(&futex_word, 1);
      } else {
        // Use futex wait with short timeout for responsive signaling.
        iree_futex_wait(&wake_signal, signal,
                        iree_time_now() + 1000000);  // 1ms timeout.
      }
    }
  });

  iree_async_futex_wait_operation_t wait_op;
  uint32_t iteration = 0;

  for (auto _ : state) {
    context->Reset();
    ++iteration;

    InitFutexWaitOp(&wait_op, &futex_word, iteration - 1,
                    BenchmarkContext::Callback, context);

    auto start = std::chrono::high_resolution_clock::now();

    iree_status_t status =
        iree_async_proactor_submit_one(context->proactor, &wait_op.base);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Submit failed");
      iree_status_ignore(status);
      break;
    }

    // Signal the background thread to wake us.
    wake_signal.store(iteration, std::memory_order_release);
    iree_futex_wake(&wake_signal, 1);

    // Wait for completion.
    if (!context->SpinPollUntilComplete(1)) {
      state.SkipWithError("Poll timeout");
      break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  stop_signaler.store(true, std::memory_order_release);
  iree_futex_wake(&wake_signal, 1);
  signaler.join();

  DestroyBenchmarkContext(context);
}

#else  // !IREE_RUNTIME_USE_FUTEX

// Placeholder benchmarks when futex is not available.
static void BM_WakeNoWaiters(::benchmark::State& state,
                             const ProactorFactory& factory) {
  state.SkipWithError("Futex not available on this platform");
}

static void BM_CrossThread(::benchmark::State& state,
                           const ProactorFactory& factory) {
  state.SkipWithError("Futex not available on this platform");
}

#endif  // IREE_RUNTIME_USE_FUTEX

//===----------------------------------------------------------------------===//
// Benchmark suite class
//===----------------------------------------------------------------------===//

class FutexBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Baseline: async wake with no waiters (io_uring roundtrip cost).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/FutexWakeNoWaiters").c_str(),
        [factory](::benchmark::State& state) {
          BM_WakeNoWaiters(state, factory);
        })
        ->UseManualTime()
        ->Unit(::benchmark::kNanosecond);

    // Cross-thread: async wait + raw wake roundtrip.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/FutexCrossThread").c_str(),
        [factory](::benchmark::State& state) {
          BM_CrossThread(state, factory);
        })
        ->UseManualTime()
        ->Unit(::benchmark::kNanosecond);
  }
};

CTS_REGISTER_BENCHMARK_SUITE(FutexBenchmarks);

}  // namespace iree::async::cts
