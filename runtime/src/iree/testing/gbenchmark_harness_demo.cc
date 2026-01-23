// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demo of iree/testing/gbenchmark_harness.h usage.
// Can be used with iree-bazel-try for one-shot benchmarking.

#include "iree/testing/gbenchmark_harness.h"

static void BM_Noop(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(state.iterations());
  }
}
BENCHMARK(BM_Noop);

static void BM_SystemAllocator(benchmark::State& state) {
  iree_allocator_t allocator = iree_allocator_system();
  for (auto _ : state) {
    void* ptr = NULL;
    iree_allocator_malloc(allocator, 1024, &ptr);
    benchmark::DoNotOptimize(ptr);
    iree_allocator_free(allocator, ptr);
  }
}
BENCHMARK(BM_SystemAllocator);

static void BM_StatusCreation(benchmark::State& state) {
  for (auto _ : state) {
    iree_status_t status = iree_make_status(IREE_STATUS_OK);
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK(BM_StatusCreation);
