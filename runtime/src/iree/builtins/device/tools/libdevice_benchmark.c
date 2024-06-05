// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/device/device.h"
#include "iree/testing/benchmark.h"

// Example flag; not really useful:
IREE_FLAG(int32_t, batch_count, 64, "Ops to run per benchmark iteration.");

static iree_status_t iree_h2f_ieee_benchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/FLAG_batch_count)) {
    for (int i = 0; i < FLAG_batch_count; ++i) {
      iree_optimization_barrier(iree_h2f_ieee(0x3400 + i));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_f2h_ieee_benchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/FLAG_batch_count)) {
    for (int i = 0; i < FLAG_batch_count; ++i) {
      iree_optimization_barrier(iree_f2h_ieee(0.25f + i));
    }
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "libdevice_benchmark",
      "Benchmarks the libdevice implementation of the target machine.\n"
      "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);

  {
    static const iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_h2f_ieee_benchmark,
        .user_data = NULL,
    };
    iree_benchmark_register(IREE_SV("iree_h2f_ieee"), &benchmark_def);
  }

  {
    static const iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_f2h_ieee_benchmark,
        .user_data = NULL,
    };
    iree_benchmark_register(IREE_SV("iree_f2h_ieee"), &benchmark_def);
  }

  iree_benchmark_run_specified();
  return 0;
}
