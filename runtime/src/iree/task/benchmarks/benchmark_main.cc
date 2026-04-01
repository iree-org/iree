// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry point for task system benchmarks.
//
// Benchmark suites register themselves at static init time via
// ::benchmark::RegisterBenchmark(). This main() just initializes
// Google Benchmark and runs everything.
//
// Usage:
//   iree-bazel-run //runtime/src/iree/task/benchmarks:dispatch_benchmarks
//
//   # Run specific benchmark:
//   iree-bazel-run //runtime/src/iree/task/benchmarks:dispatch_benchmarks -- \
//     --benchmark_filter="DispatchChain/Noop/64x128/8w"
//
//   # JSON output for analysis:
//   iree-bazel-run //runtime/src/iree/task/benchmarks:dispatch_benchmarks -- \
//     --benchmark_format=json > results.json

#include "benchmark/benchmark.h"

BENCHMARK_MAIN();
