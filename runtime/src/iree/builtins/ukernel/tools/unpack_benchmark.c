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

IREE_FLAG(int64_t, batch_min_traversal_size, 1000000000,
          "Minimum number of bytes to be traversed in each batch.");

IREE_FLAG(
    int64_t, working_set_size, 1000000,
    "Number of bytes to be traversed by the benchmark workload (input and "
    "output buffers together). Matrix shapes are computed accordingly.");
IREE_FLAG(
    int32_t, padding_size, 0,
    "Padding size (same value used for both dimensions, 0 means no padding)");

typedef struct iree_unpack_benchmark_user_data_t {
  iree_uk_unpack_type_t type;
  int size2;
  int size3;
  iree_uk_uint32_t flags;
  const iree_uk_uint64_t* cpu_data;
} iree_unpack_benchmark_user_data_t;

IREE_UK_ATTRIBUTE_NOINLINE static void iree_memcpy_noinline(
    void* restrict dst, const void* restrict src, size_t size) {
  memcpy(dst, src, size);
}

static iree_status_t iree_memcpy_benchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_uk_int64_t total_iterations = 0;
  iree_uk_int64_t batch_count =
      (FLAG_batch_min_traversal_size + FLAG_working_set_size - 1) /
      FLAG_working_set_size;
  iree_uk_ssize_t buffer_size = FLAG_working_set_size / 2;
  uint8_t* in_buffer = malloc(buffer_size);
  uint8_t* out_buffer = malloc(buffer_size);
  for (iree_uk_ssize_t i = 0; i < buffer_size; ++i) in_buffer[i] = (i & 0xFF);
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_memcpy_noinline(out_buffer, in_buffer, buffer_size);
    }
    total_iterations += batch_count;
  }
  // Report bytes per second, so that can be easily compared to known memory
  // system performance metrics (e.g. RAM bandwidth, to tell whether this is
  // memory-bound).
  iree_benchmark_set_items_processed(benchmark_state,
                                     total_iterations * buffer_size);
  assert(!memcmp(in_buffer, out_buffer, buffer_size));
  free(in_buffer);
  free(out_buffer);
  return iree_ok_status();
}

static iree_status_t iree_unpack_benchmark(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_unpack_benchmark_user_data_t* user_data = benchmark_def->user_data;
  iree_uk_type_t in_type = iree_uk_unpack_in_type(user_data->type);
  iree_uk_type_t out_type = iree_uk_unpack_out_type(user_data->type);
  iree_uk_ssize_t in_type_size = iree_uk_type_size(in_type);
  iree_uk_ssize_t out_type_size = iree_uk_type_size(out_type);

  // The inner dims 2, 3 are given to us as part of the benchmark user_data.
  // The outer dims 0, 1 are to be determined based on FLAG_working_set_size.
  iree_uk_ssize_t in_size0 = 1;
  iree_uk_ssize_t in_size1 = 1;
  iree_uk_ssize_t in_size2 = user_data->size2;
  iree_uk_ssize_t in_size3 = user_data->size3;
  int target_matrix_size_in_elems =
      FLAG_working_set_size / (in_type_size + out_type_size);
  int target_product_of_outer_sizes_0_1 =
      target_matrix_size_in_elems / (in_size2 * in_size3);
  while (target_product_of_outer_sizes_0_1 >= 4) {
    target_product_of_outer_sizes_0_1 /= 4;
    in_size0 *= 2;
    in_size1 *= 2;
  }
  in_size1 *= target_product_of_outer_sizes_0_1;

  iree_uk_unpack_params_t params;
  memset(&params, 0, sizeof params);
  params.type = user_data->type;
  params.flags = user_data->flags;
  params.in_size0 = in_size0;
  params.in_size1 = in_size1;
  params.in_size2 = in_size2;
  params.in_size3 = in_size3;
  if (params.flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&in_size0, &in_size1);
  }
  if (params.flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&in_size2, &in_size3);
  }
  params.out_size0 = iree_max(0, in_size0 * in_size2 - FLAG_padding_size);
  params.out_size1 = iree_max(0, in_size1 * in_size3 - FLAG_padding_size);
  params.out_stride0 = params.out_size1;
  params.in_stride0 = params.in_size1 * params.in_size2 * params.in_size3;
  iree_uk_ssize_t in_buffer_size = iree_uk_test_2d_buffer_length(
      in_type, params.in_size0, params.in_stride0);
  iree_uk_ssize_t out_buffer_size = iree_uk_test_2d_buffer_length(
      out_type, params.out_size0, params.out_stride0);
  void* in_buffer = malloc(in_buffer_size);
  void* out_buffer = malloc(out_buffer_size);
  iree_uk_test_random_engine_t* engine = iree_uk_test_random_engine_create();
  // It's just about plausible that on some platform, for some number type,
  // performance might be different on zero buffers vs random buffers. But it
  // shouldn't matter that we recreate the random engine every time, getting
  // the same random values again.
  iree_uk_test_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  iree_uk_test_write_random_buffer(out_buffer, out_buffer_size, out_type,
                                   engine);
  iree_uk_test_random_engine_destroy(engine);
  params.in_buffer = in_buffer;
  params.out_buffer = out_buffer;
  iree_uk_int64_t total_iterations = 0;
  iree_uk_int64_t batch_count =
      (FLAG_batch_min_traversal_size + FLAG_working_set_size - 1) /
      FLAG_working_set_size;
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/batch_count)) {
    for (int i = 0; i < batch_count; ++i) {
      iree_uk_unpack(&params);
    }
    total_iterations += batch_count;
  }
  // Report bytes per second, so that can be easily compared to known memory
  // system performance metrics (e.g. RAM bandwidth, to tell whether this is
  // memory-bound).
  iree_benchmark_set_items_processed(benchmark_state,
                                     total_iterations * out_buffer_size);
  free(in_buffer);
  free(out_buffer);
  return iree_ok_status();
}

static void iree_unpack_benchmark_register(
    const iree_unpack_benchmark_user_data_t* user_data, const char* name) {
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
      .run = iree_unpack_benchmark,
      .user_data = user_data,
  };
  iree_benchmark_register(IREE_SV(name), &benchmark_def);
}

#define UNPACK_BENCHMARK_REGISTER_WITH_FLAGS(                                  \
    _flags, _flags_suffix, _type, _size2, _size3, _cpu_data_field_0, _label)   \
  do {                                                                         \
    static const iree_uk_uint64_t local_cpu_data[IREE_CPU_DATA_FIELD_COUNT] =  \
        {_cpu_data_field_0};                                                   \
    static const iree_unpack_benchmark_user_data_t user_data = {               \
        .type = iree_uk_unpack_type_##_type,                                   \
        .size2 = _size2,                                                       \
        .size3 = _size3,                                                       \
        .flags = _flags,                                                       \
        .cpu_data = local_cpu_data,                                            \
    };                                                                         \
    iree_unpack_benchmark_register(&user_data,                                 \
                                   "iree_uk_unpack_" #_type "_" #_size2        \
                                   "x" #_size3 "_" _flags_suffix "_" #_label); \
  } while (0)

#define UNPACK_BENCHMARK_REGISTER(...)                                      \
  UNPACK_BENCHMARK_REGISTER_WITH_FLAGS(0, "TRANSPOSE_NONE", __VA_ARGS__);   \
  UNPACK_BENCHMARK_REGISTER_WITH_FLAGS(IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER, \
                                       "TRANSPOSE_INNER", __VA_ARGS__);     \
  UNPACK_BENCHMARK_REGISTER_WITH_FLAGS(IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER, \
                                       "TRANSPOSE_OUTER", __VA_ARGS__);     \
  UNPACK_BENCHMARK_REGISTER_WITH_FLAGS(                                     \
      IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER |                                 \
          IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER,                              \
      "TRANSPOSE_BOTH", __VA_ARGS__);

#define UNPACK_BENCHMARK_REGISTER_GENERIC(_type, _size2, _size3) \
  UNPACK_BENCHMARK_REGISTER(_type, _size2, _size3, 0, generic)

#define UNPACK_BENCHMARK_REGISTER_ARM_64(_type, _size2, _size3) \
  UNPACK_BENCHMARK_REGISTER(_type, _size2, _size3, 0, arm_64)

#define UNPACK_BENCHMARK_REGISTER_ARM_64_WITH_CPU_FEATURE(        \
    _type, _size2, _size3, _cpu_feature)                          \
  UNPACK_BENCHMARK_REGISTER(_type, _size2, _size3,                \
                            IREE_CPU_DATA0_ARM_64_##_cpu_feature, \
                            arm_64_##_cpu_feature)

int main(int argc, char** argv) {
  iree_flags_set_usage("unpack_benchmark",
                       "Benchmarks the pack microkernel.\n"
                       "\n");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK, &argc, &argv);
  iree_benchmark_initialize(&argc, argv);
  iree_cpu_initialize(iree_allocator_system());

  const iree_benchmark_def_t memcpy_benchmark_def = {
      .flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME,
      .time_unit = IREE_BENCHMARK_UNIT_MICROSECOND,
      .minimum_duration_ns = 0,
      .iteration_count = 0,
      .run = iree_memcpy_benchmark,
      .user_data = 0,
  };
  iree_benchmark_register(IREE_SV("memcpy"), &memcpy_benchmark_def);

  // Generic code paths, not actually used, but interesting to get a sense
  // of how slow generic code goes vs decent SIMD kernels.
  UNPACK_BENCHMARK_REGISTER_GENERIC(f32f32, 4, 4);

// ARM_64 benchmarks.
#if defined(IREE_UK_ARCH_ARM_64)

  UNPACK_BENCHMARK_REGISTER_ARM_64(f32f32, 8, 1);
  UNPACK_BENCHMARK_REGISTER_ARM_64(i8i8, 8, 1);
  UNPACK_BENCHMARK_REGISTER_ARM_64(i8i8, 8, 4);
  UNPACK_BENCHMARK_REGISTER_ARM_64(i8i8, 8, 8);

#endif  // defined(IREE_UK_ARCH_ARM_64)

  iree_benchmark_run_specified();
  return 0;
}
