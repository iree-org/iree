// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal harness for one-shot Google Benchmark experiments via iree-bazel-try.
// Uses the standard Google Benchmark C++ API with IREE types available.
//
// Basic benchmark:
//   iree-bazel-try -e '
//   #include "iree/testing/gbenchmark_harness.h"
//   void BM_Noop(benchmark::State& state) {
//     for (auto _ : state) {}
//   }
//   BENCHMARK(BM_Noop);
//   '
//
// With IREE allocator:
//   iree-bazel-try -e '
//   #include "iree/testing/gbenchmark_harness.h"
//   void BM_Alloc(benchmark::State& state) {
//     for (auto _ : state) {
//       void* p = NULL;
//       iree_allocator_malloc(iree_allocator_system(), 1024, &p);
//       DoNotOptimize(p);
//       iree_allocator_free(iree_allocator_system(), p);
//     }
//   }
//   BENCHMARK(BM_Alloc);
//   '
//
// The harness handles:
// - Google Benchmark inclusion and main() via benchmark_main
// - IREE base types (iree_allocator_t, iree_status_t, etc.)
// - Common benchmark utilities (DoNotOptimize, ClobberMemory)

#ifndef IREE_TESTING_GBENCHMARK_HARNESS_H_
#define IREE_TESTING_GBENCHMARK_HARNESS_H_

#include "benchmark/benchmark.h"
#include "iree/base/api.h"

// Bring common benchmark utilities into scope for convenience.
using ::benchmark::ClobberMemory;
using ::benchmark::DoNotOptimize;
using ::benchmark::State;

#endif  // IREE_TESTING_GBENCHMARK_HARNESS_H_
