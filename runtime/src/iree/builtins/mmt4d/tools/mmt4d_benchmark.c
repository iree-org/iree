// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/mmt4d/mmt4d.h"
#include "iree/testing/benchmark.h"

// Example flag; not really useful:
IREE_FLAG(int32_t, batch_count, 64, "Ops to run per benchmark iteration.");

static iree_status_t iree_mmt4d_example_matmul_f32_benchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/FLAG_batch_count)) {
    for (int i = 0; i < FLAG_batch_count; ++i) {
      // Example: this just ensure things link.
      iree_mmt4d_example_matmul_f32(NULL, 0, NULL, 0, NULL, 0, 0, 0, 0,
                                    (float)i, 0.0f);
    }
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "mmt4d_benchmark",
      "Benchmarks the libmmt4d implementation of the target machine.\n"
      "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);

  // TODO: always add _generic variants to have a baseline vs reference?

  {
    static const iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_mmt4d_example_matmul_f32_benchmark,
        .user_data = NULL,
    };
    iree_benchmark_register(IREE_SV("iree_mmt4d_example_matmul_f32"),
                            &benchmark_def);
  }

  iree_benchmark_run_specified();
  return 0;
}
