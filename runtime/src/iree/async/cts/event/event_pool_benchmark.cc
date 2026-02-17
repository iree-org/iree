// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for event pool operations.
//
// Measures the performance of event pool acquire/release operations.
// These are user-space operations that don't involve kernel I/O paths.

#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/event_pool.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Benchmark context
//===----------------------------------------------------------------------===//

struct EventPoolContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_event_pool_t pool = {};
  bool pool_initialized = false;
};

static EventPoolContext* CreateEventPoolContext(
    const ProactorFactory& factory, iree_host_size_t initial_capacity,
    ::benchmark::State& state) {
  auto* ctx = new EventPoolContext();

  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  iree_status_t status = iree_async_event_pool_initialize(
      ctx->proactor, iree_allocator_system(), initial_capacity, &ctx->pool);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Pool initialization failed");
    iree_status_ignore(status);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }
  ctx->pool_initialized = true;

  return ctx;
}

static void DestroyEventPoolContext(EventPoolContext* ctx) {
  if (!ctx) return;
  if (ctx->pool_initialized) {
    iree_async_event_pool_deinitialize(&ctx->pool);
  }
  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

//===----------------------------------------------------------------------===//
// Benchmark implementations
//===----------------------------------------------------------------------===//

// Benchmark acquire/release cycle from a pre-populated pool.
// This is the hot path when events are available.
// Target: <20ns per cycle.
static void BM_AcquireRelease(::benchmark::State& state,
                              const ProactorFactory& factory) {
  auto* ctx = CreateEventPoolContext(factory, /*initial_capacity=*/1, state);
  if (!ctx) return;

  // Pre-acquire one event to ensure the pool has it.
  iree_async_event_t* event = nullptr;
  iree_status_t status = iree_async_event_pool_acquire(&ctx->pool, &event);
  if (!iree_status_is_ok(status) || !event) {
    state.SkipWithError("Initial acquire failed");
    iree_status_ignore(status);
    DestroyEventPoolContext(ctx);
    return;
  }

  // Return it so it's in the return stack.
  iree_async_event_pool_release(&ctx->pool, event);

  for (auto _ : state) {
    // Acquire triggers migration from return stack if needed.
    status = iree_async_event_pool_acquire(&ctx->pool, &event);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }
    // Release puts it back on return stack.
    iree_async_event_pool_release(&ctx->pool, event);
  }

  DestroyEventPoolContext(ctx);
}

// Benchmark acquire from a pool that needs to grow (eventfd creation).
// This is the cold path that triggers syscalls.
// Target: ~2us (eventfd creation cost).
static void BM_AcquireGrow(::benchmark::State& state,
                           const ProactorFactory& factory) {
  for (auto _ : state) {
    // Create a fresh pool with no capacity each iteration.
    auto result = factory();
    if (!result.ok()) {
      state.SkipWithError("Proactor creation failed");
      return;
    }
    iree_async_proactor_t* proactor = result.value();

    iree_async_event_pool_t pool;
    iree_status_t status =
        iree_async_event_pool_initialize(proactor, iree_allocator_system(),
                                         /*initial_capacity=*/0, &pool);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Pool initialization failed");
      iree_status_ignore(status);
      iree_async_proactor_release(proactor);
      return;
    }

    // This acquire triggers growth (eventfd creation).
    iree_async_event_t* event = nullptr;
    status = iree_async_event_pool_acquire(&pool, &event);
    iree_status_ignore(status);

    // Clean up.
    if (event) {
      iree_async_event_pool_release(&pool, event);
    }
    iree_async_event_pool_deinitialize(&pool);
    iree_async_proactor_release(proactor);
  }
}

// Benchmark batch acquire: acquire N events, then release all.
// Tests pool performance under burst allocation patterns.
static void BM_BatchAcquire(::benchmark::State& state,
                            const ProactorFactory& factory, size_t batch_size) {
  auto* ctx = CreateEventPoolContext(factory, batch_size, state);
  if (!ctx) return;

  std::vector<iree_async_event_t*> events(batch_size, nullptr);

  for (auto _ : state) {
    // Acquire batch.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_status_t status =
          iree_async_event_pool_acquire(&ctx->pool, &events[i]);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        // Release what we got.
        for (size_t j = 0; j < i; ++j) {
          iree_async_event_pool_release(&ctx->pool, events[j]);
        }
        state.SkipWithError("Batch acquire failed");
        DestroyEventPoolContext(ctx);
        return;
      }
    }

    // Release batch.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_async_event_pool_release(&ctx->pool, events[i]);
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
  DestroyEventPoolContext(ctx);
}

//===----------------------------------------------------------------------===//
// Benchmark suite class
//===----------------------------------------------------------------------===//

class EventPoolBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Basic acquire/release cycle.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/EventPoolAcquireRelease").c_str(),
        [factory](::benchmark::State& state) {
          BM_AcquireRelease(state, factory);
        })
        ->Unit(::benchmark::kNanosecond);

    // Acquire with growth (cold path).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/EventPoolAcquireGrow").c_str(),
        [factory](::benchmark::State& state) {
          BM_AcquireGrow(state, factory);
        })
        ->Unit(::benchmark::kNanosecond);

    // Batch acquire at various sizes.
    for (size_t batch : {4, 16, 64}) {
      std::string name = std::string(prefix) + "/EventPoolBatchAcquire/" +
                         std::to_string(batch);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, batch](::benchmark::State& state) {
            BM_BatchAcquire(state, factory, batch);
          })
          ->Unit(::benchmark::kNanosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(EventPoolBenchmarks);

}  // namespace iree::async::cts
