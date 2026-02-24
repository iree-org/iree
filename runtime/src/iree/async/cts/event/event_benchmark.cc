// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for event signal/wait operations.
//
// Measures:
//   - Event pool acquire/release cycle time (nanosecond scale)
//   - Event wait latency: pre-signaled and signal-then-wait patterns
//   - Batch signal/wait throughput
//   - Cross-thread signaling latency
//
// All benchmarks follow IREE style:
//   - Setup outside the benchmark loop
//   - Tight measurement loop with no error paths
//   - Explicit cleanup after the loop
//   - No iree_status_ignore() in benchmark code (only in shared infrastructure)

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/event.h"
#include "iree/async/event_pool.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Event benchmark context (RAII wrapper)
//===----------------------------------------------------------------------===//

// Extends BenchmarkContext with event pool management.
// Uses RAII to ensure proper cleanup even on early returns.
class EventBenchmarkContext {
 public:
  ~EventBenchmarkContext() {
    if (pool_initialized_) {
      iree_async_event_pool_deinitialize(&pool_);
    }
    iree_async_proactor_release(proactor_);
  }

  // Factory method - returns nullptr and sets error on failure.
  static std::unique_ptr<EventBenchmarkContext> Create(
      const ProactorFactory& factory, size_t pool_capacity,
      ::benchmark::State& state) {
    std::unique_ptr<EventBenchmarkContext> ctx(new EventBenchmarkContext());

    auto result = factory();
    if (!result.ok()) {
      if (result.status().code() == iree::StatusCode::kUnavailable) {
        state.SkipWithError("Backend unavailable on this system");
      } else {
        state.SkipWithError("Proactor creation failed");
      }
      return nullptr;
    }
    ctx->proactor_ = result.value();

    iree_status_t status = iree_async_event_pool_initialize(
        ctx->proactor_, iree_allocator_system(), pool_capacity, &ctx->pool_);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Event pool initialization failed");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      return nullptr;
    }
    ctx->pool_initialized_ = true;

    return ctx;
  }

  iree_async_proactor_t* proactor() { return proactor_; }
  iree_async_event_pool_t* pool() { return &pool_; }

  // Completion tracking.
  std::atomic<int> completions{0};
  iree_status_code_t last_status_code = IREE_STATUS_OK;

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<EventBenchmarkContext*>(user_data);
    ctx->last_status_code = iree_status_code(status);
    ctx->completions.fetch_add(1, std::memory_order_release);
    // Note: We must consume the status. The callback signature doesn't allow
    // returning it, so we store the code above and free/ignore here.
    iree_status_free(status);
  }

  void Reset() {
    completions.store(0, std::memory_order_release);
    last_status_code = IREE_STATUS_OK;
  }

  // Poll until expected completions, with timeout.
  bool PollUntilComplete(int expected,
                         iree_duration_t budget_ns = 5000000000LL) {
    iree_time_t deadline = iree_time_now() + budget_ns;
    while (completions.load(std::memory_order_acquire) < expected) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t count = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(100), &count);
      // DEADLINE_EXCEEDED is expected when nothing is ready yet.
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_free(status);
        return false;
      }
      iree_status_free(status);
    }
    return last_status_code == IREE_STATUS_OK;
  }

 private:
  EventBenchmarkContext() = default;

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_event_pool_t pool_ = {};
  bool pool_initialized_ = false;
};

//===----------------------------------------------------------------------===//
// Pool microbenchmarks
//===----------------------------------------------------------------------===//

// Measures event pool acquire + release cycle time.
// Target: <500ns acquire, <20ns release (per event_pool.h documentation).
static void BM_EventPoolAcquireRelease(::benchmark::State& state,
                                       const ProactorFactory& factory) {
  auto ctx = EventBenchmarkContext::Create(factory, /*pool_capacity=*/1, state);
  if (!ctx) return;

  for (auto _ : state) {
    iree_async_event_t* event = nullptr;
    iree_status_t status = iree_async_event_pool_acquire(ctx->pool(), &event);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.SkipWithError("Acquire failed");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      break;
    }
    iree_async_event_pool_release(ctx->pool(), event);
  }
}

// Measures batch acquire/release to observe amortization effects.
static void BM_EventPoolBatchAcquireRelease(::benchmark::State& state,
                                            const ProactorFactory& factory,
                                            size_t batch_size) {
  auto ctx = EventBenchmarkContext::Create(factory,
                                           /*pool_capacity=*/batch_size, state);
  if (!ctx) return;

  std::vector<iree_async_event_t*> events(batch_size, nullptr);

  for (auto _ : state) {
    // Acquire batch.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_status_t status =
          iree_async_event_pool_acquire(ctx->pool(), &events[i]);
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        state.SkipWithError("Batch acquire failed");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        // Release any already acquired.
        for (size_t j = 0; j < i; ++j) {
          iree_async_event_pool_release(ctx->pool(), events[j]);
        }
        return;
      }
    }
    // Release batch.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_async_event_pool_release(ctx->pool(), events[i]);
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
}

//===----------------------------------------------------------------------===//
// Signal/wait latency benchmarks
//===----------------------------------------------------------------------===//

// Measures minimum latency path: event is pre-signaled before wait is
// submitted. This tests the io_uring POLL_ADD immediate-completion path.
static void BM_EventWaitPreSignaled(::benchmark::State& state,
                                    const ProactorFactory& factory) {
  auto ctx = EventBenchmarkContext::Create(factory, /*pool_capacity=*/1, state);
  if (!ctx) return;

  // Acquire event once, reuse across iterations.
  iree_async_event_t* event = nullptr;
  iree_status_t status = iree_async_event_pool_acquire(ctx->pool(), &event);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Event acquire failed");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return;
  }

  // Pre-build wait operation structure (reused each iteration).
  iree_async_event_wait_operation_t wait_op;

  for (auto _ : state) {
    // Pre-signal the event before submitting wait.
    iree_async_event_set(event);

    // Submit wait - should complete immediately since already signaled.
    memset(&wait_op, 0, sizeof(wait_op));
    wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_op.base.completion_fn = EventBenchmarkContext::Callback;
    wait_op.base.user_data = ctx.get();
    wait_op.event = event;

    ctx->Reset();
    status = iree_async_proactor_submit_one(ctx->proactor(), &wait_op.base);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.SkipWithError("Submit failed");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      break;
    }

    // Poll until complete. The linked POLL_ADD->READ auto-drains the eventfd.
    if (!ctx->PollUntilComplete(1)) {
      state.SkipWithError("Wait timeout");
      break;
    }
  }

  iree_async_event_pool_release(ctx->pool(), event);
}

// Measures typical producer pattern: submit wait, then signal, then poll.
static void BM_EventSignalWait(::benchmark::State& state,
                               const ProactorFactory& factory) {
  auto ctx = EventBenchmarkContext::Create(factory, /*pool_capacity=*/1, state);
  if (!ctx) return;

  iree_async_event_t* event = nullptr;
  iree_status_t status = iree_async_event_pool_acquire(ctx->pool(), &event);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Event acquire failed");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return;
  }

  iree_async_event_wait_operation_t wait_op;

  for (auto _ : state) {
    // Submit wait first.
    memset(&wait_op, 0, sizeof(wait_op));
    wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_op.base.completion_fn = EventBenchmarkContext::Callback;
    wait_op.base.user_data = ctx.get();
    wait_op.event = event;

    ctx->Reset();
    status = iree_async_proactor_submit_one(ctx->proactor(), &wait_op.base);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.SkipWithError("Submit failed");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      break;
    }

    // Signal the event.
    iree_async_event_set(event);

    // Poll until complete.
    if (!ctx->PollUntilComplete(1)) {
      state.SkipWithError("Wait timeout");
      break;
    }
  }

  iree_async_event_pool_release(ctx->pool(), event);
}

//===----------------------------------------------------------------------===//
// Batch operation benchmarks
//===----------------------------------------------------------------------===//

// Measures batch throughput: submit N waits, signal all, poll all.
// Note: wait_ops are pre-allocated outside the loop.
static void BM_EventBatchSignalWait(::benchmark::State& state,
                                    const ProactorFactory& factory,
                                    size_t batch_size) {
  auto ctx = EventBenchmarkContext::Create(factory,
                                           /*pool_capacity=*/batch_size, state);
  if (!ctx) return;

  // Acquire all events upfront.
  std::vector<iree_async_event_t*> events(batch_size, nullptr);
  for (size_t i = 0; i < batch_size; ++i) {
    iree_status_t status =
        iree_async_event_pool_acquire(ctx->pool(), &events[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Event acquire failed");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      for (size_t j = 0; j < i; ++j) {
        iree_async_event_pool_release(ctx->pool(), events[j]);
      }
      return;
    }
  }

  // Pre-allocate wait operation structures.
  std::vector<iree_async_event_wait_operation_t> wait_ops(batch_size);

  bool error_occurred = false;
  for (auto _ : state) {
    if (error_occurred) break;
    ctx->Reset();

    // Initialize and submit all waits.
    bool submit_failed = false;
    for (size_t i = 0; i < batch_size; ++i) {
      memset(&wait_ops[i], 0, sizeof(wait_ops[i]));
      wait_ops[i].base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
      wait_ops[i].base.completion_fn = EventBenchmarkContext::Callback;
      wait_ops[i].base.user_data = ctx.get();
      wait_ops[i].event = events[i];

      iree_status_t status =
          iree_async_proactor_submit_one(ctx->proactor(), &wait_ops[i].base);
      if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
        state.SkipWithError("Submit failed");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        submit_failed = true;
        error_occurred = true;
        break;
      }
    }
    if (submit_failed) continue;

    // Signal all events.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_async_event_set(events[i]);
    }

    // Poll until all complete.
    if (!ctx->PollUntilComplete(static_cast<int>(batch_size))) {
      state.SkipWithError("Batch wait timeout");
      error_occurred = true;
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    if (events[i]) {
      iree_async_event_pool_release(ctx->pool(), events[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// Cross-thread benchmarks
//===----------------------------------------------------------------------===//

// Cross-thread synchronization helper.
struct CrossThreadSync {
  std::mutex mutex;
  std::condition_variable cv;
  bool signal_requested = false;
  bool stop_requested = false;

  void RequestSignal() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      signal_requested = true;
    }
    cv.notify_one();
  }

  void RequestStop() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop_requested = true;
    }
    cv.notify_one();
  }

  // Returns true if should signal, false if should stop.
  bool WaitForRequest() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return signal_requested || stop_requested; });
    if (stop_requested) return false;
    signal_requested = false;
    return true;
  }
};

// Measures cross-thread signal latency.
// A background thread signals events while the main thread waits.
static void BM_EventCrossThreadSignal(::benchmark::State& state,
                                      const ProactorFactory& factory) {
  auto ctx = EventBenchmarkContext::Create(factory, /*pool_capacity=*/1, state);
  if (!ctx) return;

  iree_async_event_t* event = nullptr;
  iree_status_t status = iree_async_event_pool_acquire(ctx->pool(), &event);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Event acquire failed");
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return;
  }

  CrossThreadSync sync;

  // Background signaler thread.
  std::thread signaler([&]() {
    while (sync.WaitForRequest()) {
      iree_async_event_set(event);
    }
  });

  iree_async_event_wait_operation_t wait_op;

  for (auto _ : state) {
    memset(&wait_op, 0, sizeof(wait_op));
    wait_op.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
    wait_op.base.completion_fn = EventBenchmarkContext::Callback;
    wait_op.base.user_data = ctx.get();
    wait_op.event = event;

    ctx->Reset();
    status = iree_async_proactor_submit_one(ctx->proactor(), &wait_op.base);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      state.SkipWithError("Submit failed");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      break;
    }

    // Wake signaler thread.
    sync.RequestSignal();

    // Poll until complete.
    if (!ctx->PollUntilComplete(1)) {
      state.SkipWithError("Cross-thread wait timeout");
      break;
    }
  }

  sync.RequestStop();
  signaler.join();

  iree_async_event_pool_release(ctx->pool(), event);
}

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

class EventBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Pool microbenchmarks.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/EventPoolAcquireRelease").c_str(),
        [factory](::benchmark::State& state) {
          BM_EventPoolAcquireRelease(state, factory);
        })
        ->Unit(::benchmark::kNanosecond);

    for (size_t batch : {4, 16, 64}) {
      std::string name = std::string(prefix) +
                         "/EventPoolBatchAcquireRelease/" +
                         std::to_string(batch);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, batch](::benchmark::State& state) {
            BM_EventPoolBatchAcquireRelease(state, factory, batch);
          })
          ->Unit(::benchmark::kNanosecond);
    }

    // Signal/wait latency.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/EventWaitPreSignaled").c_str(),
        [factory](::benchmark::State& state) {
          BM_EventWaitPreSignaled(state, factory);
        })
        ->Unit(::benchmark::kMicrosecond);

    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/EventSignalWait").c_str(),
        [factory](::benchmark::State& state) {
          BM_EventSignalWait(state, factory);
        })
        ->Unit(::benchmark::kMicrosecond);

    // Batch throughput.
    for (size_t batch : {4, 16, 64}) {
      std::string name = std::string(prefix) + "/EventBatchSignalWait/" +
                         std::to_string(batch);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, batch](::benchmark::State& state) {
            BM_EventBatchSignalWait(state, factory, batch);
          })
          ->Unit(::benchmark::kMicrosecond);
    }

    // Cross-thread.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/EventCrossThreadSignal").c_str(),
        [factory](::benchmark::State& state) {
          BM_EventCrossThreadSignal(state, factory);
        })
        ->Unit(::benchmark::kMicrosecond);
  }
};

CTS_REGISTER_BENCHMARK_SUITE(EventBenchmarks);

}  // namespace iree::async::cts
