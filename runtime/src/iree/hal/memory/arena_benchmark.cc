// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>

#include "benchmark/benchmark.h"
#include "iree/hal/memory/arena.h"

namespace {

static iree_hal_memory_arena_options_t BenchOptions() {
  iree_hal_memory_arena_options_t options = {};
  options.capacity = 64 * 1024 * 1024;  // 64 MB
  options.frontier_capacity = 4;
  return options;
}

// Steady-state acquire+release cycle: acquire one region, immediately release.
static void BM_AcquireRelease(benchmark::State& state) {
  iree_hal_memory_arena_t* arena = NULL;
  IREE_CHECK_OK(iree_hal_memory_arena_allocate(
      BenchOptions(), iree_allocator_system(), &arena));

  for (auto _ : state) {
    iree_hal_memory_arena_allocation_t alloc;
    IREE_CHECK_OK(iree_hal_memory_arena_acquire(arena, 256, 16, &alloc));
    iree_hal_memory_arena_release(arena, NULL);
  }

  iree_hal_memory_arena_free(arena);
}
BENCHMARK(BM_AcquireRelease);

// Batch pattern: acquire N regions, then release N. Matches real scratch usage
// (allocate during command recording, free when submission completes).
static void BM_AcquireBurst(benchmark::State& state) {
  const int batch_size = static_cast<int>(state.range(0));
  iree_hal_memory_arena_t* arena = NULL;
  IREE_CHECK_OK(iree_hal_memory_arena_allocate(
      BenchOptions(), iree_allocator_system(), &arena));

  for (auto _ : state) {
    int acquired_count = 0;
    for (int i = 0; i < batch_size; ++i) {
      iree_hal_memory_arena_allocation_t alloc;
      iree_status_t status =
          iree_hal_memory_arena_acquire(arena, 256, 16, &alloc);
      if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        state.SkipWithError("unexpected exhaustion");
        break;
      }
      ++acquired_count;
    }
    for (int i = 0; i < acquired_count; ++i) {
      iree_hal_memory_arena_release(arena, NULL);
    }
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          batch_size * 2);

  iree_hal_memory_arena_free(arena);
}
BENCHMARK(BM_AcquireBurst)->Arg(8)->Arg(32)->Arg(128)->Arg(512);

// Acquire with varying alignments to measure alignment overhead.
static void BM_AcquireAligned(benchmark::State& state) {
  const iree_device_size_t alignment =
      static_cast<iree_device_size_t>(state.range(0));
  iree_hal_memory_arena_t* arena = NULL;
  IREE_CHECK_OK(iree_hal_memory_arena_allocate(
      BenchOptions(), iree_allocator_system(), &arena));

  for (auto _ : state) {
    iree_hal_memory_arena_allocation_t alloc;
    IREE_CHECK_OK(iree_hal_memory_arena_acquire(arena, 64, alignment, &alloc));
    iree_hal_memory_arena_release(arena, NULL);
  }

  iree_hal_memory_arena_free(arena);
}
BENCHMARK(BM_AcquireAligned)->Arg(1)->Arg(16)->Arg(256)->Arg(4096);

}  // namespace
