// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/memcpy_benchmark.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/builtins/ukernel/tools/benchmark.h"

IREE_UK_ATTRIBUTE_NOINLINE static void iree_memcpy_noinline(
    void* restrict dst, const void* restrict src, size_t size) {
  memcpy(dst, src, size);
}

typedef struct iree_uk_benchmark_memcpy_user_data_t {
  int64_t working_set_size;
  int64_t batch_min_traversal_size;
} iree_uk_benchmark_memcpy_user_data_t;

static iree_status_t iree_uk_benchmark_memcpy(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_uk_benchmark_memcpy_user_data_t* user_data =
      benchmark_def->user_data;

  int64_t total_iterations = 0;
  int64_t batch_count =
      (user_data->batch_min_traversal_size + user_data->working_set_size - 1) /
      user_data->working_set_size;
  iree_uk_ssize_t buffer_size = user_data->working_set_size / 2;
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

void iree_uk_benchmark_register_memcpy(int64_t working_set_size,
                                       int64_t batch_min_traversal_size) {
  iree_uk_benchmark_memcpy_user_data_t* user_data =
      iree_uk_benchmark_static_alloc(
          sizeof(iree_uk_benchmark_memcpy_user_data_t));
  user_data->working_set_size = working_set_size;
  user_data->batch_min_traversal_size = batch_min_traversal_size;

  const iree_benchmark_def_t memcpy_benchmark_def = {
      .flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME,
      .time_unit = IREE_BENCHMARK_UNIT_MICROSECOND,
      .minimum_duration_ns = 0,
      .iteration_count = 0,
      .run = iree_uk_benchmark_memcpy,
      .user_data = user_data,
  };
  char name[128];
  snprintf(name, sizeof name, "memcpy_wss_%" PRIi64, working_set_size);
  iree_benchmark_register(IREE_SV(name), &memcpy_benchmark_def);
}
