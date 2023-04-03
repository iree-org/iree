// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/tools/benchmark.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/schemas/cpu_data.h"

struct iree_uk_benchmark_user_data_t {
  const void* params;
  iree_uk_uint64_t* cpu_data;
  iree_uk_random_engine_t random_engine;
};

const void* iree_uk_benchmark_params(
    const iree_uk_benchmark_user_data_t* user_data) {
  return user_data->params;
}

const iree_uk_uint64_t* iree_uk_benchmark_cpu_data(
    const iree_uk_benchmark_user_data_t* user_data) {
  return user_data->cpu_data;
}

iree_uk_random_engine_t* iree_uk_benchmark_random_engine(
    const iree_uk_benchmark_user_data_t* user_data) {
  // Cast constness away, i.e. consider random engine state mutation as not
  // really a benchmark state mutation.
  return (iree_uk_random_engine_t*)&user_data->random_engine;
}

static int s_iree_uk_benchmark_static_alloc_count;
static int s_iree_uk_benchmark_static_alloc_max;
static void** s_iree_uk_benchmark_static_alloc_ptrs;

void* iree_uk_benchmark_static_alloc(size_t size) {
  IREE_UK_ASSERT(s_iree_uk_benchmark_static_alloc_count <
                 s_iree_uk_benchmark_static_alloc_max);
  void* ptr = malloc(size);
  s_iree_uk_benchmark_static_alloc_ptrs
      [s_iree_uk_benchmark_static_alloc_count++] = ptr;
  return ptr;
}

void iree_uk_benchmark_initialize(int* argc, char** argv) {
  // Maximum number of benchmarks that can be registered.
  int max_benchmarks = 256;
  // Maximum number of calls to iree_uk_benchmark_static_alloc in
  // iree_uk_benchmark_register
  int max_static_allocs_per_benchmark = 3;

  s_iree_uk_benchmark_static_alloc_max =
      max_static_allocs_per_benchmark * max_benchmarks;
  s_iree_uk_benchmark_static_alloc_ptrs =
      malloc(s_iree_uk_benchmark_static_alloc_max * sizeof(void*));

  iree_benchmark_initialize(argc, argv);
}

void iree_uk_benchmark_run_and_cleanup(void) {
  iree_benchmark_run_specified();
  for (int i = 0; i < s_iree_uk_benchmark_static_alloc_count; ++i) {
    free(s_iree_uk_benchmark_static_alloc_ptrs[i]);
  }
  free(s_iree_uk_benchmark_static_alloc_ptrs);
}

void iree_uk_benchmark_register(
    const char* name,
    iree_status_t (*benchmark_func)(const iree_benchmark_def_t*,
                                    iree_benchmark_state_t*),
    const void* params, size_t params_size, const char* cpu_features) {
  // Does this benchmark require an optional CPU feature?
  iree_uk_uint64_t cpu_data_local[IREE_CPU_DATA_FIELD_COUNT] = {0};
  if (cpu_features) {
    iree_uk_initialize_cpu_once();
    iree_uk_make_cpu_data_for_features(cpu_features, cpu_data_local);
    if (!iree_uk_cpu_supports(cpu_data_local)) {
      return;
    }
  }
  iree_uk_benchmark_user_data_t* user_data =
      iree_uk_benchmark_static_alloc(sizeof(iree_uk_benchmark_user_data_t));
  user_data->params = iree_uk_benchmark_static_alloc(params_size);
  memcpy((void*)user_data->params, params, params_size);
  user_data->cpu_data = iree_uk_benchmark_static_alloc(sizeof cpu_data_local);
  memcpy((void*)user_data->cpu_data, cpu_data_local, sizeof cpu_data_local);
  // benchmark_def does not need to be static, it will be cloned.
  const iree_benchmark_def_t benchmark_def = {
      .flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME,
      .time_unit = IREE_BENCHMARK_UNIT_MICROSECOND,
      .minimum_duration_ns = 0,
      .iteration_count = 0,
      .run = benchmark_func,
      .user_data = user_data,
  };
  iree_string_builder_t full_name;
  iree_string_builder_initialize(iree_allocator_system(), &full_name);
  IREE_CHECK_OK(iree_string_builder_append_cstring(&full_name, name));
  if (cpu_features) {
    IREE_CHECK_OK(iree_string_builder_append_cstring(&full_name, "_"));
    IREE_CHECK_OK(iree_string_builder_append_cstring(&full_name, cpu_features));
  }
  iree_benchmark_register(iree_string_builder_view(&full_name), &benchmark_def);
  iree_string_builder_deinitialize(&full_name);
}
