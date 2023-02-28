// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/flags.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/tools/ukernel_test_utils.h"
#include "iree/testing/benchmark.h"

IREE_FLAG(int32_t, batch_count, 1000, "Ops to run per benchmark iteration.");
IREE_FLAG(int32_t, m_size, 1,
          "M-dimension of mmt4d ops. The overall number of rows of the "
          "accumulator is that times the M0 tile size.");
IREE_FLAG(int32_t, n_size, 1,
          "N-dimension of mmt4d ops. The overall number of columns of the "
          "accumulator is that times the N0 tile size.");
IREE_FLAG(
    int32_t, k_size, 256,
    "K-dimension of mmt4d ops. That's the number of iterations of the inner "
    "loop. The overall accumulation depth is that times the K0 tile size.");
IREE_FLAG(bool, accumulate, false,
          "Whether the kernel should accumulate into the existing accumulator "
          "tile values, or zero the accumulator tile.");

struct iree_mmt4d_benchmark_user_data_t {
  iree_uk_mmt4d_type_t type;
  int M0;
  int N0;
  int K0;
  const iree_uk_uint64_t* cpu_data;
};

typedef struct iree_mmt4d_benchmark_user_data_t
    iree_mmt4d_benchmark_user_data_t;

static iree_status_t iree_mmt4d_benchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_mmt4d_benchmark_user_data_t* user_data = benchmark_def->user_data;
  iree_uk_mmt4d_params_t params;
  memset(&params, 0, sizeof params);
  params.type = user_data->type;
  params.flags = FLAG_accumulate ? IREE_UK_FLAG_ACCUMULATE : 0;
  params.M = FLAG_m_size;
  params.N = FLAG_n_size;
  params.K = FLAG_k_size;
  params.M0 = user_data->M0;
  params.N0 = user_data->N0;
  params.K0 = user_data->K0;
  params.cpu_data = user_data->cpu_data;
  params.lhs_stride = params.K * params.M0 * params.K0;
  params.rhs_stride = params.K * params.N0 * params.K0;
  params.out_stride = params.N * params.M0 * params.N0;
  iree_uk_type_t lhs_type = iree_uk_mmt4d_lhs_type(params.type);
  iree_uk_type_t rhs_type = iree_uk_mmt4d_rhs_type(params.type);
  iree_uk_type_t out_type = iree_uk_mmt4d_out_type(params.type);
  iree_uk_ssize_t lhs_buffer_size =
      iree_uk_test_2d_buffer_length(lhs_type, params.M, params.lhs_stride);
  iree_uk_ssize_t rhs_buffer_size =
      iree_uk_test_2d_buffer_length(rhs_type, params.N, params.rhs_stride);
  iree_uk_ssize_t out_buffer_size =
      iree_uk_test_2d_buffer_length(out_type, params.M, params.out_stride);
  void* lhs_buffer = malloc(lhs_buffer_size);
  void* rhs_buffer = malloc(rhs_buffer_size);
  void* out_buffer = malloc(out_buffer_size);
  iree_uk_test_random_engine_t* engine = iree_uk_test_random_engine_create();
  // It's just about plausible that on some platform, for some number type,
  // performance might be different on zero buffers vs random buffers. But it
  // shouldn't matter that we recreate the random engine every time, getting
  // the same random values again.
  iree_uk_test_write_random_buffer(lhs_buffer, lhs_buffer_size, lhs_type,
                                   engine);
  iree_uk_test_write_random_buffer(rhs_buffer, rhs_buffer_size, rhs_type,
                                   engine);
  iree_uk_test_write_random_buffer(out_buffer, out_buffer_size, out_type,
                                   engine);
  iree_uk_test_random_engine_destroy(engine);
  params.lhs_buffer = lhs_buffer;
  params.rhs_buffer = rhs_buffer;
  params.out_buffer = out_buffer;
  iree_uk_int64_t total_iterations = 0;
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/FLAG_batch_count)) {
    for (int i = 0; i < FLAG_batch_count; ++i) {
      iree_uk_mmt4d(&params);
    }
    total_iterations += FLAG_batch_count;
  }
  iree_benchmark_set_items_processed(
      benchmark_state, total_iterations * 2 * params.M * params.N * params.K *
                           params.M0 * params.N0 * params.K0);
  free(lhs_buffer);
  free(rhs_buffer);
  free(out_buffer);
  return iree_ok_status();
}

static void iree_mmt4d_benchmark_register(
    const iree_mmt4d_benchmark_user_data_t* user_data, const char* name) {
  // Does this benchmark require an optional CPU feature?
  if (user_data->cpu_data[0]) {
    if ((iree_cpu_data_field(0) & user_data->cpu_data[0]) !=
        user_data->cpu_data[0]) {
      // The CPU does not meet this benchmark's requirements. The builtin
      // would crash.
      return;
    }
  }

  // benchmark_def does not need to be static, it will be cloned.
  const iree_benchmark_def_t benchmark_def = {
      .flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME,
      .time_unit = IREE_BENCHMARK_UNIT_MICROSECOND,
      .minimum_duration_ns = 0,
      .iteration_count = 0,
      .run = iree_mmt4d_benchmark,
      .user_data = user_data,
  };
  iree_benchmark_register(IREE_SV(name), &benchmark_def);
}

#define MMT4D_BENCHMARK_REGISTER(_type, _m0, _n0, _k0, _cpu_data_field_0,      \
                                 _label)                                       \
  do {                                                                         \
    static const iree_uk_uint64_t local_cpu_data[IREE_CPU_DATA_FIELD_COUNT] =  \
        {_cpu_data_field_0};                                                   \
    static const iree_mmt4d_benchmark_user_data_t user_data = {                \
        .type = iree_uk_mmt4d_type_##_type,                                    \
        .M0 = _m0,                                                             \
        .N0 = _n0,                                                             \
        .K0 = _k0,                                                             \
        .cpu_data = local_cpu_data,                                            \
    };                                                                         \
    iree_mmt4d_benchmark_register(&user_data, "iree_uk_mmt4d_" #_type "_" #_m0 \
                                              "x" #_n0 "x" #_k0 "_" #_label);  \
  } while (0)

#define MMT4D_BENCHMARK_REGISTER_GENERIC(_type, _m0, _n0, _k0) \
  MMT4D_BENCHMARK_REGISTER(_type, _m0, _n0, _k0, 0, generic)

#define MMT4D_BENCHMARK_REGISTER_ARM_64(_type, _m0, _n0, _k0) \
  MMT4D_BENCHMARK_REGISTER(_type, _m0, _n0, _k0, 0, arm_64)

#define MMT4D_BENCHMARK_REGISTER_ARM_64_WITH_CPU_FEATURE(_type, _m0, _n0, _k0, \
                                                         _cpu_feature)         \
  MMT4D_BENCHMARK_REGISTER(_type, _m0, _n0, _k0,                               \
                           IREE_CPU_DATA0_ARM_64_##_cpu_feature,               \
                           arm_64_##_cpu_feature)

int main(int argc, char** argv) {
  iree_flags_set_usage("mmt4d_benchmark",
                       "Benchmarks the mmt4d microkernel.\n"
                       "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);
  iree_cpu_initialize(iree_allocator_system());

  // Generic code paths, not actually used, but interesting to get a sense
  // of how slow generic code goes vs decent SIMD kernels. Interesting also to
  // compare generic float vs int arithmetic.
  MMT4D_BENCHMARK_REGISTER_GENERIC(f32f32f32, 4, 4, 1);
  MMT4D_BENCHMARK_REGISTER_GENERIC(i8i8i32, 4, 4, 1);

// ARM_64 benchmarks.
#if defined(IREE_UK_ARCH_ARM_64)

  MMT4D_BENCHMARK_REGISTER_ARM_64(f32f32f32, 8, 8, 1);
  MMT4D_BENCHMARK_REGISTER_ARM_64(i8i8i32, 8, 8, 1);
  MMT4D_BENCHMARK_REGISTER_ARM_64_WITH_CPU_FEATURE(i8i8i32, 8, 8, 4, DOTPROD);
  MMT4D_BENCHMARK_REGISTER_ARM_64_WITH_CPU_FEATURE(i8i8i32, 8, 8, 8, I8MM);

#endif  // defined(IREE_UK_ARCH_ARM_64)

  iree_benchmark_run_specified();
  return 0;
}
