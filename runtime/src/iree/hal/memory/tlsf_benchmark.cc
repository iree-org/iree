// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/memory/tlsf.h"

namespace {

static iree_hal_memory_tlsf_options_t BenchOptions() {
  iree_hal_memory_tlsf_options_t options = {};
  // 256MB is large enough that the synthetic fragmented workload should not
  // fail due to total arena capacity.
  options.range_length = 256 * 1024 * 1024;
  options.alignment = 16;
  options.initial_block_capacity = 4096;
  options.frontier_capacity = 8;
  return options;
}

// Benchmark steady-state allocation (no coalescing, no splits after warmup).
static void BM_Allocate(benchmark::State& state) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_CHECK_OK(iree_hal_memory_tlsf_initialize(
      BenchOptions(), iree_allocator_system(), &tlsf));

  iree_device_size_t size = static_cast<iree_device_size_t>(state.range(0));

  for (auto _ : state) {
    iree_hal_memory_tlsf_allocation_t alloc;
    IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &alloc));
    iree_hal_memory_tlsf_free(&tlsf, alloc.block_index, NULL);
  }

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}
BENCHMARK(BM_Allocate)->Arg(64)->Arg(256)->Arg(4096)->Arg(65536);

// Benchmark free with no coalescing (neighbors are allocated).
static void BM_FreeNoCoalesce(benchmark::State& state) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_CHECK_OK(iree_hal_memory_tlsf_initialize(
      BenchOptions(), iree_allocator_system(), &tlsf));

  iree_device_size_t size = static_cast<iree_device_size_t>(state.range(0));

  // Pre-allocate three blocks. Free and re-allocate the middle one each
  // iteration. The left and right neighbors remain allocated, preventing
  // coalescing.
  iree_hal_memory_tlsf_allocation_t left, right;
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &left));
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &right));

  // We need a gap between left and right, so allocate the middle too.
  // Actually, allocate left, middle, right to ensure adjacency.
  iree_hal_memory_tlsf_allocation_t middle;
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &middle));
  // Free right - but then middle's right neighbor would be free. Fix:
  // keep right allocated and use a 4-block pattern.
  // Simpler: just allocate two guards and benchmark free of a block between.
  iree_hal_memory_tlsf_free(&tlsf, middle.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, right.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, left.block_index, NULL);

  // New approach: fill enough to make 3 adjacent allocated blocks.
  iree_hal_memory_tlsf_allocation_t guard_left, target, guard_right;
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &guard_left));
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &target));
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &guard_right));

  for (auto _ : state) {
    iree_hal_memory_tlsf_free(&tlsf, target.block_index, NULL);
    IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &target));
  }

  iree_hal_memory_tlsf_free(&tlsf, guard_right.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, target.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, guard_left.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}
BENCHMARK(BM_FreeNoCoalesce)->Arg(64)->Arg(256)->Arg(4096);

// Benchmark free with both-side coalescing.
static void BM_FreeCoalesceBoth(benchmark::State& state) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_CHECK_OK(iree_hal_memory_tlsf_initialize(
      BenchOptions(), iree_allocator_system(), &tlsf));

  iree_device_size_t size = static_cast<iree_device_size_t>(state.range(0));

  // Allocate three blocks, free outer two, then repeatedly free+alloc the
  // middle (which coalesces with both neighbors each time).
  iree_hal_memory_tlsf_allocation_t a, b, c;
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &a));
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &b));
  IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &c));

  for (auto _ : state) {
    // Free outer blocks (they coalesce with the rest of the range).
    iree_hal_memory_tlsf_free(&tlsf, a.block_index, NULL);
    iree_hal_memory_tlsf_free(&tlsf, c.block_index, NULL);
    // Free middle - coalesces both directions.
    iree_hal_memory_tlsf_free(&tlsf, b.block_index, NULL);
    // Re-allocate all three.
    IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &a));
    IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &b));
    IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &c));
  }

  iree_hal_memory_tlsf_free(&tlsf, c.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, b.block_index, NULL);
  iree_hal_memory_tlsf_free(&tlsf, a.block_index, NULL);
  iree_hal_memory_tlsf_deinitialize(&tlsf);
}
BENCHMARK(BM_FreeCoalesceBoth)->Arg(64)->Arg(256)->Arg(4096);

// Benchmark a fragmented workload: many small allocations, interleaved frees.
static void BM_FragmentedWorkload(benchmark::State& state) {
  iree_hal_memory_tlsf_t tlsf;
  IREE_CHECK_OK(iree_hal_memory_tlsf_initialize(
      BenchOptions(), iree_allocator_system(), &tlsf));

  const int batch_size = static_cast<int>(state.range(0));
  std::vector<iree_hal_memory_tlsf_allocation_t> allocs;
  allocs.reserve(batch_size * 2);

  for (auto _ : state) {
    // Allocate a batch.
    for (int i = 0; i < batch_size; ++i) {
      iree_hal_memory_tlsf_allocation_t alloc;
      iree_device_size_t size = 16 * (1 + (i * 7) % 32);
      IREE_CHECK_OK(iree_hal_memory_tlsf_allocate(&tlsf, size, &alloc));
      allocs.push_back(alloc);
    }
    // Free every other block.
    for (size_t i = 0; i < allocs.size(); i += 2) {
      iree_hal_memory_tlsf_free(&tlsf, allocs[i].block_index, NULL);
      allocs[i].block_index = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
    }
    // Free remaining.
    for (auto& a : allocs) {
      if (a.block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
        iree_hal_memory_tlsf_free(&tlsf, a.block_index, NULL);
      }
    }
    allocs.clear();
  }

  iree_hal_memory_tlsf_deinitialize(&tlsf);
}
BENCHMARK(BM_FragmentedWorkload)->Arg(10)->Arg(100)->Arg(1000);

}  // namespace
