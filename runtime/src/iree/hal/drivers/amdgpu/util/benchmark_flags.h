// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_BENCHMARK_FLAGS_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_BENCHMARK_FLAGS_H_

#include <benchmark/benchmark.h>

#include "iree/async/operation.h"

// Returns the async wait strategy selected by --completion_wait_flags.
iree_async_wait_flags_t iree_hal_amdgpu_benchmark_completion_wait_flags(void);

// Adds completion wait mode labels to |state| so benchmark rows remain
// self-describing when active/yield/blocking wait results are compared.
void iree_hal_amdgpu_benchmark_set_completion_wait_counters(
    benchmark::State& state);

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_BENCHMARK_FLAGS_H_
