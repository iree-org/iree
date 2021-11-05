// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/libmmt4d/mmt4d.h"
#include "iree/testing/benchmark.h"

IREE_FLAG(int32_t, batch_count, 64,
          "MMT4D ops to run per benchmark iteration.");

typedef struct {
  iree_string_view_t name;
  int m0;
  int k0;
  int n0;
  size_t lhs_size;
  size_t rhs_size;
  size_t dst_size;
  void (*fn)(int k_size, const int8_t* lhs, const int8_t* rhs,
             int32_t* restrict dst);
} mmt4d_benchmark_t;

static void fill_rand(void* ptr, size_t length, uint32_t mod) {
  static uint32_t state = 0;
  for (size_t i = 0; i < length; ++i) {
    state = (state * 123 + 456) % 321;
    ((uint8_t*)ptr)[i] = state % mod;
  }
}

static iree_status_t mmt4d_benchmark(const iree_benchmark_def_t* benchmark_def,
                                     iree_benchmark_state_t* benchmark_state) {
  const mmt4d_benchmark_t* benchmark =
      (const mmt4d_benchmark_t*)benchmark_def->user_data;

  const int k_size = 4 * benchmark->k0;
  const size_t lhs_length = k_size * benchmark->m0 * benchmark->lhs_size;
  void* lhs = malloc(lhs_length);
  fill_rand(lhs, lhs_length, 5);
  const size_t rhs_length = k_size * benchmark->n0 * benchmark->rhs_size;
  void* rhs = malloc(rhs_length);
  fill_rand(rhs, rhs_length, 6);
  size_t dst_length = benchmark->m0 * benchmark->n0 * benchmark->dst_size;
  void* dst = calloc(1, dst_length);

  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/FLAG_batch_count)) {
    for (int i = 0; i < FLAG_batch_count; ++i) {
      benchmark->fn(k_size, lhs, rhs, dst);
    }
  }

  free(lhs);
  free(rhs);
  free(dst);

  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "mmt4d_benchmark",
      "Benchmarks the MMT4D implementation of the target machine.\n"
      "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);

  static const mmt4d_benchmark_t benchmarks[] = {
      {
          .name = IREE_SVL("mmt4d_8x4x8_i8i8i32"),
          .m0 = 8,
          .k0 = 4,
          .n0 = 8,
          .lhs_size = sizeof(int8_t),
          .rhs_size = sizeof(int8_t),
          .dst_size = sizeof(int32_t),
          .fn = mmt4d_8x4x8_i8i8i32,
      },
  };

  for (size_t i = 0; i < IREE_ARRAYSIZE(benchmarks); ++i) {
    iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = mmt4d_benchmark,
        .user_data = &benchmarks[i],
    };
    iree_benchmark_register(benchmarks[i].name, &benchmark_def);
  }

  iree_benchmark_run_specified();
  return 0;
}
