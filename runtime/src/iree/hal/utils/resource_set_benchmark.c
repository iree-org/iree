// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/prng.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/testing/benchmark.h"

typedef struct iree_hal_test_resource_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_test_resource_t;

typedef struct iree_hal_test_resource_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_test_resource_t* resource);
} iree_hal_test_resource_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_test_resource_vtable_t);

static const iree_hal_test_resource_vtable_t iree_hal_test_resource_vtable;

static iree_status_t iree_hal_test_resource_create(
    iree_allocator_t host_allocator, iree_hal_resource_t** out_resource) {
  iree_hal_test_resource_t* test_resource = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*test_resource), (void**)&test_resource));
  iree_hal_resource_initialize(&iree_hal_test_resource_vtable,
                               &test_resource->resource);
  test_resource->host_allocator = host_allocator;
  *out_resource = (iree_hal_resource_t*)test_resource;
  return iree_ok_status();
}

static void iree_hal_test_resource_destroy(iree_hal_test_resource_t* resource) {
  iree_allocator_t host_allocator = resource->host_allocator;
  iree_allocator_free(host_allocator, resource);
}

static const iree_hal_test_resource_vtable_t iree_hal_test_resource_vtable = {
    /*.destroy=*/iree_hal_test_resource_destroy,
};

// Tests init/deinit performance when 0+ resources are in the set.
// This is our worst-case with unique resources that never match the MRU.
//
// user_data is a count of elements to insert into each set.
static iree_status_t iree_hal_resource_set_benchmark_lifecycle_n(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t host_allocator = benchmark_state->host_allocator;

  // Initialize the block pool we'll be serving from.
  // Sized like we usually do it in the runtime for ~512-1024 elements.
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, host_allocator, &block_pool);

  // Allocate the resources we'll be using - we keep them live so that we are
  // measuring just the retain/release and set times instead of the timing of
  // resource creation/deletion.
  uint32_t count = (uint32_t)(uintptr_t)benchmark_def->user_data;
  iree_hal_resource_t** resources = NULL;
  if (count > 0) {
    IREE_CHECK_OK(iree_allocator_malloc(host_allocator,
                                        sizeof(iree_hal_resource_t*) * count,
                                        (void**)&resources));
  }
  for (uint32_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_test_resource_create(host_allocator, &resources[i]));
  }

  // Create/insert/delete lifecycle.
  int64_t batch_count;
  while (iree_benchmark_keep_running(benchmark_state, &batch_count)) {
    for (int64_t i = 0; i < batch_count; ++i) {
      iree_hal_resource_set_t* set = NULL;
      IREE_CHECK_OK(iree_hal_resource_set_allocate(&block_pool, &set));
      IREE_CHECK_OK(iree_hal_resource_set_insert(set, count, resources));
      iree_hal_resource_set_free(set);
    }
  }

  // Cleanup.
  for (uint32_t i = 0; i < count; ++i) {
    iree_hal_resource_release(resources[i]);
  }
  iree_allocator_free(host_allocator, resources);
  iree_arena_block_pool_deinitialize(&block_pool);

  return iree_ok_status();
}

// Tests insertion performance when either the MRU is used (n < MRU size) or
// the worst-case performance when all resources are unique and guaranteed to
// miss the MRU. Expect to see a cliff where we spill the MRU.
//
// user_data is a count of unique elements to insert.
static iree_status_t iree_hal_resource_set_benchmark_insert_n(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t host_allocator = benchmark_state->host_allocator;

  // Initialize the block pool we'll be serving from.
  // Sized like we usually do it in the runtime for ~512-1024 elements.
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, host_allocator, &block_pool);

  // Create the empty set using the block pool for additional memory.
  iree_hal_resource_set_t* set = NULL;
  IREE_CHECK_OK(iree_hal_resource_set_allocate(&block_pool, &set));

  // Allocate the resources we'll be using - we keep them live so that we are
  // measuring just the retain/release and set times instead of the timing of
  // resource creation/deletion.
  uint32_t count = (uint32_t)(uintptr_t)benchmark_def->user_data;
  iree_hal_resource_t** resources = NULL;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator,
                                      sizeof(iree_hal_resource_t*) * count,
                                      (void**)&resources));
  for (uint32_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_test_resource_create(host_allocator, &resources[i]));
  }

  // Insert the resources. After the first iteration these should all be hits.
  int64_t batch_count;
  while (iree_benchmark_keep_running(benchmark_state, &batch_count)) {
    for (int64_t i = 0; i < batch_count; ++i) {
      IREE_CHECK_OK(iree_hal_resource_set_insert(set, count, resources));
    }
  }

  // Cleanup.
  for (uint32_t i = 0; i < count; ++i) {
    iree_hal_resource_release(resources[i]);
  }
  iree_hal_resource_set_free(set);
  iree_allocator_free(host_allocator, resources);
  iree_arena_block_pool_deinitialize(&block_pool);

  return iree_ok_status();
}

// Tests insertion into the set in a randomized order.
// This lets us get a somewhat reasonable approximation of average performance.
// In reality what the compiler spits out is non-random and often just
// alternating A/B/C/B/A/C/A/B/C etc kind of sequences.
//
// This is the most important benchmark: if this is fast then we are :thumbsup:.
//
// user_data is a count of unique element pool to insert N times. The higher
// the pool size the more likely we are to miss the MRU.
static iree_status_t iree_hal_resource_set_benchmark_randomized_n(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t host_allocator = benchmark_state->host_allocator;

  // Initialize the block pool we'll be serving from.
  // Sized like we usually do it in the runtime for ~512-1024 elements.
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, host_allocator, &block_pool);

  // Allocate the resources we'll be using - we keep them live so that we are
  // measuring just the retain/release and set times instead of the timing of
  // resource creation/deletion.
  uint32_t count = (uint32_t)(uintptr_t)benchmark_def->user_data;
  iree_hal_resource_t** resources = NULL;
  IREE_CHECK_OK(iree_allocator_malloc(host_allocator,
                                      sizeof(iree_hal_resource_t*) * count,
                                      (void**)&resources));
  for (uint32_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_test_resource_create(host_allocator, &resources[i]));
  }

  // The same set is maintained; we'll eventually have all resources in the set
  // and be testing the MRU hit %.
  iree_hal_resource_set_t* set = NULL;
  IREE_CHECK_OK(iree_hal_resource_set_allocate(&block_pool, &set));

  // The PRNG we use to select the elements.
  iree_prng_xoroshiro128_state_t prng = {0};
  iree_prng_xoroshiro128_initialize(123ull, &prng);

  // Insert N random resources into the set.
  int64_t batch_count;
  while (iree_benchmark_keep_running(benchmark_state, &batch_count)) {
    for (int64_t i = 0; i < batch_count; ++i) {
      uint32_t resource_idx =
          iree_prng_xoroshiro128plus_next_uint32(&prng) % count;
      iree_hal_resource_t* resource = resources[resource_idx];
      IREE_CHECK_OK(iree_hal_resource_set_insert(set, 1, &resource));
    }
  }

  // Cleanup.
  iree_hal_resource_set_free(set);
  for (uint32_t i = 0; i < count; ++i) {
    iree_hal_resource_release(resources[i]);
  }
  iree_allocator_free(host_allocator, resources);
  iree_arena_block_pool_deinitialize(&block_pool);

  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_benchmark_initialize(&argc, argv);

  // iree_hal_resource_set_benchmark_lifecycle_n
  {
    iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_hal_resource_set_benchmark_lifecycle_n,
    };
    benchmark_def.user_data = (void*)0u;
    iree_benchmark_register(iree_make_cstring_view("lifecycle_0"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)1u;
    iree_benchmark_register(iree_make_cstring_view("lifecycle_1"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)256u;
    iree_benchmark_register(iree_make_cstring_view("lifecycle_256"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)1024u;
    iree_benchmark_register(iree_make_cstring_view("lifecycle_1024"),
                            &benchmark_def);
  }

  // iree_hal_resource_set_benchmark_insert_n
  {
    iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_hal_resource_set_benchmark_insert_n,
    };
    benchmark_def.user_data = (void*)1u;
    iree_benchmark_register(iree_make_cstring_view("insert_1"), &benchmark_def);
    benchmark_def.user_data = (void*)5u;
    iree_benchmark_register(iree_make_cstring_view("insert_5"), &benchmark_def);
    benchmark_def.user_data = (void*)32u;
    iree_benchmark_register(iree_make_cstring_view("insert_32"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)64u;
    iree_benchmark_register(iree_make_cstring_view("insert_64"),
                            &benchmark_def);
  }

  // iree_hal_resource_set_benchmark_randomized_n
  {
    iree_benchmark_def_t benchmark_def = {
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_NANOSECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_hal_resource_set_benchmark_randomized_n,
    };
    benchmark_def.user_data = (void*)1u;
    iree_benchmark_register(iree_make_cstring_view("randomized_1"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)4u;
    iree_benchmark_register(iree_make_cstring_view("randomized_4"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)8u;
    iree_benchmark_register(iree_make_cstring_view("randomized_8"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)32u;
    iree_benchmark_register(iree_make_cstring_view("randomized_32"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)256u;
    iree_benchmark_register(iree_make_cstring_view("randomized_256"),
                            &benchmark_def);
    benchmark_def.user_data = (void*)4096u;
    iree_benchmark_register(iree_make_cstring_view("randomized_4096"),
                            &benchmark_def);
  }

  iree_benchmark_run_specified();
  return 0;
}
