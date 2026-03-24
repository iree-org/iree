// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/memory/block_pool.h"

namespace {

static iree_hal_memory_block_pool_options_t BenchOptions(int block_count) {
  iree_hal_memory_block_pool_options_t options = {};
  options.block_size = 4096;
  options.block_count = static_cast<uint32_t>(block_count);
  options.frontier_capacity = 4;
  return options;
}

// Steady-state acquire+release cycle: acquire one block, immediately release
// it. This is the hot path for block pool usage (e.g., KV cache block
// recycling).
static void BM_AcquireRelease(benchmark::State& state) {
  iree_hal_memory_block_pool_t* pool = NULL;
  IREE_CHECK_OK(iree_hal_memory_block_pool_allocate(
      BenchOptions(1024), iree_allocator_system(), &pool));

  for (auto _ : state) {
    iree_hal_memory_block_pool_allocation_t alloc;
    iree_status_t status = iree_hal_memory_block_pool_acquire(pool, &alloc);
    if (iree_status_is_ok(status)) {
      iree_hal_memory_block_pool_release(pool, alloc.block_index, NULL);
    } else {
      iree_status_ignore(status);
    }
  }

  iree_hal_memory_block_pool_free(pool);
}
BENCHMARK(BM_AcquireRelease);

// Acquire-only until exhaustion, then release all and repeat.
// Measures acquisition throughput without interleaved releases.
static void BM_AcquireBurst(benchmark::State& state) {
  const int block_count = static_cast<int>(state.range(0));
  iree_hal_memory_block_pool_t* pool = NULL;
  IREE_CHECK_OK(iree_hal_memory_block_pool_allocate(
      BenchOptions(block_count), iree_allocator_system(), &pool));

  std::vector<uint32_t> indices(block_count);
  for (auto _ : state) {
    // Acquire all.
    for (int i = 0; i < block_count; ++i) {
      iree_hal_memory_block_pool_allocation_t alloc;
      iree_status_t status = iree_hal_memory_block_pool_acquire(pool, &alloc);
      if (iree_status_is_ok(status)) {
        indices[i] = alloc.block_index;
      } else {
        iree_status_ignore(status);
        state.SkipWithError("unexpected exhaustion");
        break;
      }
    }
    // Release all.
    for (int i = 0; i < block_count; ++i) {
      iree_hal_memory_block_pool_release(pool, indices[i], NULL);
    }
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          block_count * 2);

  iree_hal_memory_block_pool_free(pool);
}
BENCHMARK(BM_AcquireBurst)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

// Contended acquire+release from multiple threads.
static void BM_ContendedAcquireRelease(benchmark::State& state) {
  static constexpr int kBlockCount = 1024;
  static iree_hal_memory_block_pool_t* shared_pool = nullptr;

  if (state.thread_index() == 0) {
    IREE_CHECK_OK(iree_hal_memory_block_pool_allocate(
        BenchOptions(kBlockCount), iree_allocator_system(), &shared_pool));
  }

  for (auto _ : state) {
    iree_hal_memory_block_pool_allocation_t alloc;
    iree_status_t status =
        iree_hal_memory_block_pool_acquire(shared_pool, &alloc);
    if (iree_status_is_ok(status)) {
      iree_hal_memory_block_pool_release(shared_pool, alloc.block_index, NULL);
    } else {
      iree_status_ignore(status);
    }
  }

  if (state.thread_index() == 0) {
    iree_hal_memory_block_pool_free(shared_pool);
    shared_pool = nullptr;
  }
}
BENCHMARK(BM_ContendedAcquireRelease)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);

}  // namespace
