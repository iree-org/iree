// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for buffer pool operations.
//
// Measures the performance of buffer pool acquire/release operations.
// The release path is particularly interesting because it involves
// lock-free freelist operations and (for recv pools) atomic stores to
// kernel-visible buffer ring tails.

#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/buffer_pool.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/proactor.h"
#include "iree/async/slab.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Benchmark context
//===----------------------------------------------------------------------===//

struct BufferPoolContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_slab_t* slab = nullptr;
  iree_async_region_t* region = nullptr;
  iree_async_buffer_pool_t* pool = nullptr;
  iree_host_size_t buffer_count = 0;
};

static BufferPoolContext* CreateBufferPoolContext(
    const ProactorFactory& factory, iree_host_size_t buffer_size,
    iree_host_size_t buffer_count, ::benchmark::State& state) {
  auto* ctx = new BufferPoolContext();
  ctx->buffer_count = buffer_count;

  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create slab with NUMA-aware allocation.
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = buffer_size;
  slab_options.buffer_count = buffer_count;
  iree_status_t status =
      iree_async_slab_create(slab_options, iree_allocator_system(), &ctx->slab);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Slab allocation failed");
    iree_status_ignore(status);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Register slab with proactor for zero-copy I/O.
  status = iree_async_proactor_register_slab(ctx->proactor, ctx->slab,
                                             IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
                                             &ctx->region);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Slab registration failed");
    iree_status_ignore(status);
    iree_async_slab_release(ctx->slab);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Create pool over registered region.
  status = iree_async_buffer_pool_allocate(ctx->region, iree_allocator_system(),
                                           &ctx->pool);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Pool allocation failed");
    iree_status_ignore(status);
    iree_async_region_release(ctx->region);
    iree_async_slab_release(ctx->slab);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  return ctx;
}

static void DestroyBufferPoolContext(BufferPoolContext* ctx) {
  if (!ctx) return;
  iree_async_buffer_pool_free(ctx->pool);
  iree_async_region_release(ctx->region);
  iree_async_slab_release(ctx->slab);
  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

//===----------------------------------------------------------------------===//
// Benchmark implementations
//===----------------------------------------------------------------------===//

// Benchmark full acquire/release cycle.
// This is the hot path for per-packet buffer management.
// Measures: freelist pop + freelist push.
static void BM_AcquireRelease(::benchmark::State& state,
                              const ProactorFactory& factory) {
  // Use 16 buffers of 4KB (typical network buffer size).
  auto* ctx = CreateBufferPoolContext(factory, /*buffer_size=*/4096,
                                      /*buffer_count=*/16, state);
  if (!ctx) return;

  iree_async_buffer_lease_t lease;
  for (auto _ : state) {
    iree_status_t status = iree_async_buffer_pool_acquire(ctx->pool, &lease);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      state.SkipWithError("Acquire failed");
      break;
    }
    iree_async_buffer_lease_release(&lease);
  }

  DestroyBufferPoolContext(ctx);
}

// Benchmark acquire-only (freelist pop).
// Isolates the cost of the user-space freelist operation.
// Requires pre-releasing buffers between iterations.
static void BM_AcquireOnly(::benchmark::State& state,
                           const ProactorFactory& factory) {
  // Large pool so we don't exhaust during warmup.
  auto* ctx = CreateBufferPoolContext(factory, /*buffer_size=*/4096,
                                      /*buffer_count=*/1024, state);
  if (!ctx) return;

  // Acquire all, then release all to set up steady state.
  std::vector<iree_async_buffer_lease_t> leases(ctx->buffer_count);
  for (size_t i = 0; i < ctx->buffer_count; ++i) {
    iree_status_t status =
        iree_async_buffer_pool_acquire(ctx->pool, &leases[i]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      state.SkipWithError("Setup acquire failed");
      DestroyBufferPoolContext(ctx);
      return;
    }
  }
  for (size_t i = 0; i < ctx->buffer_count; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }

  // Now benchmark just acquire (pool is full).
  size_t idx = 0;
  for (auto _ : state) {
    iree_status_t status =
        iree_async_buffer_pool_acquire(ctx->pool, &leases[idx]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      // Pool exhausted - release all and restart.
      for (size_t i = 0; i < idx; ++i) {
        iree_async_buffer_lease_release(&leases[i]);
      }
      idx = 0;
      continue;
    }
    ++idx;
    if (idx >= ctx->buffer_count) {
      // Release all to refill.
      for (size_t i = 0; i < idx; ++i) {
        iree_async_buffer_lease_release(&leases[i]);
      }
      idx = 0;
    }
  }

  // Cleanup remaining.
  for (size_t i = 0; i < idx; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }

  DestroyBufferPoolContext(ctx);
}

// Benchmark release-only (freelist push).
// Isolates the cost of returning buffers to the freelist.
static void BM_ReleaseOnly(::benchmark::State& state,
                           const ProactorFactory& factory) {
  // Large pool for sustained benchmarking.
  auto* ctx = CreateBufferPoolContext(factory, /*buffer_size=*/4096,
                                      /*buffer_count=*/1024, state);
  if (!ctx) return;

  // Pre-acquire all buffers.
  std::vector<iree_async_buffer_lease_t> leases(ctx->buffer_count);
  for (size_t i = 0; i < ctx->buffer_count; ++i) {
    iree_status_t status =
        iree_async_buffer_pool_acquire(ctx->pool, &leases[i]);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      state.SkipWithError("Setup acquire failed");
      DestroyBufferPoolContext(ctx);
      return;
    }
  }

  // Benchmark release (pool is empty, all buffers leased).
  size_t idx = 0;
  for (auto _ : state) {
    iree_async_buffer_lease_release(&leases[idx]);
    ++idx;
    if (idx >= ctx->buffer_count) {
      // Re-acquire all to continue.
      for (size_t i = 0; i < ctx->buffer_count; ++i) {
        iree_status_t status =
            iree_async_buffer_pool_acquire(ctx->pool, &leases[i]);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          state.SkipWithError("Re-acquire failed");
          DestroyBufferPoolContext(ctx);
          return;
        }
      }
      idx = 0;
    }
  }

  // Cleanup remaining.
  for (size_t i = idx; i < ctx->buffer_count; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }

  DestroyBufferPoolContext(ctx);
}

// Benchmark batch acquire/release: acquire N buffers, then release all.
// Tests pool performance under burst allocation patterns typical of
// scatter-gather I/O or multiple concurrent receives.
static void BM_BatchAcquireRelease(::benchmark::State& state,
                                   const ProactorFactory& factory,
                                   size_t batch_size) {
  // Pool size must be at least batch_size and power of 2 for io_uring.
  size_t pool_size = batch_size;
  if (pool_size < 16) pool_size = 16;
  // Round up to power of 2.
  size_t po2 = 1;
  while (po2 < pool_size) po2 <<= 1;
  pool_size = po2;

  auto* ctx =
      CreateBufferPoolContext(factory, /*buffer_size=*/4096, pool_size, state);
  if (!ctx) return;

  std::vector<iree_async_buffer_lease_t> leases(batch_size);

  for (auto _ : state) {
    // Acquire batch.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_status_t status =
          iree_async_buffer_pool_acquire(ctx->pool, &leases[i]);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        // Release what we got.
        for (size_t j = 0; j < i; ++j) {
          iree_async_buffer_lease_release(&leases[j]);
        }
        state.SkipWithError("Batch acquire failed");
        DestroyBufferPoolContext(ctx);
        return;
      }
    }

    // Release batch.
    for (size_t i = 0; i < batch_size; ++i) {
      iree_async_buffer_lease_release(&leases[i]);
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
  DestroyBufferPoolContext(ctx);
}

// Benchmark with varying buffer sizes to check for size-dependent effects.
// Larger buffers might have different cache/TLB behavior.
static void BM_AcquireReleaseSize(::benchmark::State& state,
                                  const ProactorFactory& factory,
                                  iree_host_size_t buffer_size) {
  auto* ctx = CreateBufferPoolContext(factory, buffer_size,
                                      /*buffer_count=*/64, state);
  if (!ctx) return;

  iree_async_buffer_lease_t lease;
  for (auto _ : state) {
    iree_status_t status = iree_async_buffer_pool_acquire(ctx->pool, &lease);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      state.SkipWithError("Acquire failed");
      break;
    }
    iree_async_buffer_lease_release(&lease);
  }

  state.SetBytesProcessed(state.iterations() * buffer_size);
  DestroyBufferPoolContext(ctx);
}

//===----------------------------------------------------------------------===//
// Benchmark suite class
//===----------------------------------------------------------------------===//

class PoolBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Full acquire/release cycle (main hot path).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/BufferPoolAcquireRelease").c_str(),
        [factory](::benchmark::State& state) {
          BM_AcquireRelease(state, factory);
        })
        ->Unit(::benchmark::kNanosecond);

    // Acquire-only (freelist pop).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/BufferPoolAcquireOnly").c_str(),
        [factory](::benchmark::State& state) {
          BM_AcquireOnly(state, factory);
        })
        ->Unit(::benchmark::kNanosecond);

    // Release-only (freelist push).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/BufferPoolReleaseOnly").c_str(),
        [factory](::benchmark::State& state) {
          BM_ReleaseOnly(state, factory);
        })
        ->Unit(::benchmark::kNanosecond);

    // Batch acquire/release at various sizes.
    for (size_t batch : {4, 16, 64}) {
      std::string name = std::string(prefix) +
                         "/BufferPoolBatchAcquireRelease/" +
                         std::to_string(batch);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, batch](::benchmark::State& state) {
            BM_BatchAcquireRelease(state, factory, batch);
          })
          ->Unit(::benchmark::kNanosecond);
    }

    // Different buffer sizes (64B to 64KB).
    for (iree_host_size_t size : {64, 512, 4096, 65536}) {
      std::string name = std::string(prefix) + "/BufferPoolAcquireRelease/" +
                         std::to_string(size) + "B";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, size](::benchmark::State& state) {
            BM_AcquireReleaseSize(state, factory, size);
          })
          ->Unit(::benchmark::kNanosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(PoolBenchmarks);

}  // namespace iree::async::cts
