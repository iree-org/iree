// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry point for carrier CTS benchmark binaries.
//
// Link-time composition: benchmark suites and backends register at static init.
// This main() calls InstantiateAll() to pair suites with backends.

#include "benchmark/benchmark.h"
#include "iree/net/carrier/cts/util/registry.h"

int main(int argc, char** argv) {
  // Instantiate benchmark suites for all registered backends.
  ::iree::net::carrier::cts::CtsRegistry::InstantiateAll();

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
