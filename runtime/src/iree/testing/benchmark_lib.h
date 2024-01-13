// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This header is a trampoline for including the backing C++ benchmark.h from
// the native benchmark library. We are trying to move away from this but we
// still have some benchmarks that use it. Those should include this header
// instead of benchmark/benchmark.h so we keep the build graph layering clean
// and can issue an error when it is not available.

#ifndef IREE_TESTING_BENCHMARK_LIB_FULL_H_

#if IREE_HAS_NOP_BENCHMARK_LIB
#error \
    "IREE was compiled without threading or benchmark support. Guard this include with IREE_HAS_NOP_BENCHMARK_LIB"
#else
#include "benchmark/benchmark.h"
#endif  // IREE_HAS_NOP_BENCHMARK_LIB

#endif  // IREE_TESTING_BENCHMARK_LIB_FULL_H_