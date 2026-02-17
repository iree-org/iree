// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry point for CTS benchmark binaries.
//
// Link-time composition: benchmark suites and backends register at static init
// time. This main() calls InstantiateAll() to pair suites with backends and
// register benchmarks with Google Benchmark.
//
// Usage:
//   cc_binary_benchmark(
//       name = "buffer_benchmarks",
//       srcs = ["//runtime/src/iree/async/cts:benchmark_main.cc"],
//       deps = [
//           ":backends",
//           "//runtime/src/iree/async/cts/buffer:all_benchmarks",
//           ...
//       ],
//   )

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/registry.h"

int main(int argc, char** argv) {
  // Instantiate benchmark suites for all registered backends.
  // This must happen before benchmark::Initialize() so that benchmarks
  // are registered before the framework parses command-line filters.
  ::iree::async::cts::CtsRegistry::InstantiateAll();

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
