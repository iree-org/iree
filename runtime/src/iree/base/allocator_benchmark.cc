// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for iree_allocator_t operations.
// Build/run: iree-bazel-run //runtime/src/iree/base:allocator_benchmark
//
// These benchmarks measure the overhead of IREE's allocator abstraction layer
// and various allocation patterns. Useful for comparing allocator
// implementations and evaluating the cost of checked arithmetic helpers.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/testing/benchmark.h"

//===----------------------------------------------------------------------===//
// Benchmark configuration
//===----------------------------------------------------------------------===//

// Allocation sizes for benchmarks.
#define ALLOC_SIZE_SMALL 64
#define ALLOC_SIZE_MEDIUM 4096
#define ALLOC_SIZE_LARGE (64 * 1024)

// Array allocation parameters.
#define ARRAY_COUNT 100
#define ARRAY_ELEMENT_SIZE 32

// Cache line alignment constant (matches IREE_HAL_HEAP_BUFFER_ALIGNMENT).
#define CACHE_LINE_ALIGNMENT 64

//===----------------------------------------------------------------------===//
// Test structures for struct layout benchmarks
//===----------------------------------------------------------------------===//

typedef struct {
  void* allocator;
  uint16_t capacity;
  uint16_t count;
  void* user_handles;
  void* native_handles;
} test_wait_set_t;

typedef struct {
  uint64_t dummy[4];
} test_wait_handle_t;

typedef void* test_native_handle_t;

typedef struct {
  void* resource_vtable;
  void* allocator;
  uint64_t flags;
  uint64_t size;
} test_buffer_header_t;

//===----------------------------------------------------------------------===//
// Basic allocation benchmarks
//===----------------------------------------------------------------------===//

static iree_status_t BM_MallocFree_Small(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_MallocFree_Medium(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_MEDIUM, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_MallocFree_Large(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_LARGE, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_MallocUninitializedFree_Medium(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        allocator, ALLOC_SIZE_MEDIUM, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Realloc benchmarks
//===----------------------------------------------------------------------===//

static iree_status_t BM_ReallocGrow(const iree_benchmark_def_t* benchmark_def,
                                    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr));
    IREE_RETURN_IF_ERROR(
        iree_allocator_realloc(allocator, ALLOC_SIZE_MEDIUM, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_ReallocShrink(const iree_benchmark_def_t* benchmark_def,
                                      iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_MEDIUM, &ptr));
    IREE_RETURN_IF_ERROR(
        iree_allocator_realloc(allocator, ALLOC_SIZE_SMALL, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Clone benchmark
//===----------------------------------------------------------------------===//

static iree_status_t BM_Clone(const iree_benchmark_def_t* benchmark_def,
                              iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  // Source data to clone.
  uint8_t source_data[ALLOC_SIZE_MEDIUM];
  memset(source_data, 0xAB, sizeof(source_data));
  iree_const_byte_span_t source =
      iree_make_const_byte_span(source_data, sizeof(source_data));

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_clone(allocator, source, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Array allocation benchmarks
//===----------------------------------------------------------------------===//

static iree_status_t BM_MallocArray(const iree_benchmark_def_t* benchmark_def,
                                    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, ARRAY_COUNT,
                                                     ARRAY_ELEMENT_SIZE, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_MallocArrayUninitialized(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        allocator, ARRAY_COUNT, ARRAY_ELEMENT_SIZE, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_ReallocArray(const iree_benchmark_def_t* benchmark_def,
                                     iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, ARRAY_COUNT,
                                                     ARRAY_ELEMENT_SIZE, &ptr));
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        allocator, ARRAY_COUNT * 2, ARRAY_ELEMENT_SIZE, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Struct allocation benchmarks
//===----------------------------------------------------------------------===//

static iree_status_t BM_MallocWithTrailing(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_with_trailing(
        allocator, sizeof(test_wait_set_t), ALLOC_SIZE_MEDIUM, &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

static iree_status_t BM_MallocStructArray(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
        allocator, sizeof(test_wait_set_t), ARRAY_COUNT, ARRAY_ELEMENT_SIZE,
        &ptr));
    iree_allocator_free(allocator, ptr);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Inline arena benchmarks
//===----------------------------------------------------------------------===//

static iree_status_t BM_InlineArenaMalloc(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    IREE_ALLOCATOR_INLINE_STORAGE(storage, 8192);
    iree_allocator_t allocator = iree_allocator_inline_arena(&storage.header);
    void* ptr1 = NULL;
    void* ptr2 = NULL;
    void* ptr3 = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr1));
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr2));
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr3));
    // Arena frees are no-ops, storage cleaned up automatically.
    iree_benchmark_use_ptr((char const volatile*)ptr1);
    iree_benchmark_use_ptr((char const volatile*)ptr2);
    iree_benchmark_use_ptr((char const volatile*)ptr3);
  }
  return iree_ok_status();
}

static iree_status_t BM_SystemMallocMultiple(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  iree_allocator_t allocator = iree_allocator_system();
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    void* ptr1 = NULL;
    void* ptr2 = NULL;
    void* ptr3 = NULL;
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr1));
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr2));
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(allocator, ALLOC_SIZE_SMALL, &ptr3));
    iree_allocator_free(allocator, ptr3);
    iree_allocator_free(allocator, ptr2);
    iree_allocator_free(allocator, ptr1);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Struct layout calculation benchmarks
//===----------------------------------------------------------------------===//
// These measure the overhead of IREE_STRUCT_LAYOUT vs manual checked
// arithmetic. With IREE_ATTRIBUTE_ALWAYS_INLINE, both should have identical
// performance when counts are compile-time constants (complete constant
// folding).

static iree_status_t BM_StructLayoutManual(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    for (int i = 0; i < 1000; ++i) {
      iree_host_size_t capacity = 64;

      iree_host_size_t user_handle_list_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              capacity, sizeof(test_wait_handle_t), &user_handle_list_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }
      iree_host_size_t native_handle_list_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              capacity, sizeof(test_native_handle_t),
              &native_handle_list_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }
      iree_host_size_t total_size = 0;
      if (IREE_UNLIKELY(
              !iree_host_size_checked_add(sizeof(test_wait_set_t),
                                          user_handle_list_size, &total_size) ||
              !iree_host_size_checked_add(total_size, native_handle_list_size,
                                          &total_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }

      iree_benchmark_use_ptr((char const volatile*)&total_size);
      iree_benchmark_use_ptr((char const volatile*)&user_handle_list_size);
    }
  }
  return iree_ok_status();
}

static iree_status_t BM_StructLayoutMacro(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    for (int i = 0; i < 1000; ++i) {
      iree_host_size_t capacity = 64;
      iree_host_size_t total = 0;
      iree_host_size_t user_offset = 0;
      iree_host_size_t native_offset = 0;

      iree_status_t status = IREE_STRUCT_LAYOUT(
          sizeof(test_wait_set_t), &total,
          IREE_STRUCT_FIELD(capacity, test_wait_handle_t, &user_offset),
          IREE_STRUCT_FIELD(capacity, test_native_handle_t, &native_offset));
      if (!iree_status_is_ok(status)) {
        return status;
      }

      iree_benchmark_use_ptr((char const volatile*)&total);
      iree_benchmark_use_ptr((char const volatile*)&user_offset);
    }
  }
  return iree_ok_status();
}

// Prevent constant propagation - volatile forces runtime evaluation.
static iree_host_size_t get_dynamic_value(iree_host_size_t value) {
  volatile iree_host_size_t v = value;
  return v;
}

static iree_status_t BM_StructLayoutManual_Dynamic(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    for (int i = 0; i < 1000; ++i) {
      iree_host_size_t capacity = get_dynamic_value(64);

      iree_host_size_t user_handle_list_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              capacity, sizeof(test_wait_handle_t), &user_handle_list_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }
      iree_host_size_t native_handle_list_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              capacity, sizeof(test_native_handle_t),
              &native_handle_list_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }
      iree_host_size_t total_size = 0;
      if (IREE_UNLIKELY(
              !iree_host_size_checked_add(sizeof(test_wait_set_t),
                                          user_handle_list_size, &total_size) ||
              !iree_host_size_checked_add(total_size, native_handle_list_size,
                                          &total_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }

      iree_benchmark_use_ptr((char const volatile*)&total_size);
      iree_benchmark_use_ptr((char const volatile*)&user_handle_list_size);
    }
  }
  return iree_ok_status();
}

static iree_status_t BM_StructLayoutMacro_Dynamic(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    for (int i = 0; i < 1000; ++i) {
      iree_host_size_t capacity = get_dynamic_value(64);
      iree_host_size_t total = 0;
      iree_host_size_t user_offset = 0;
      iree_host_size_t native_offset = 0;

      iree_status_t status = IREE_STRUCT_LAYOUT(
          sizeof(test_wait_set_t), &total,
          IREE_STRUCT_FIELD(capacity, test_wait_handle_t, &user_offset),
          IREE_STRUCT_FIELD(capacity, test_native_handle_t, &native_offset));
      if (!iree_status_is_ok(status)) {
        return status;
      }

      iree_benchmark_use_ptr((char const volatile*)&total);
      iree_benchmark_use_ptr((char const volatile*)&user_offset);
    }
  }
  return iree_ok_status();
}

static iree_status_t BM_StructLayoutManual_Aligned(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    for (int i = 0; i < 1000; ++i) {
      iree_host_size_t data_size = 4096;

      // Header aligned to max_align_t.
      iree_host_size_t header_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_align(
              sizeof(test_buffer_header_t), iree_max_align_t, &header_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }

      // Data offset aligned to cache line.
      iree_host_size_t data_offset = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_align(
              header_size, CACHE_LINE_ALIGNMENT, &data_offset))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }

      // Total size.
      iree_host_size_t total_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_add(data_offset, data_size,
                                                    &total_size))) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "overflow");
      }

      iree_benchmark_use_ptr((char const volatile*)&total_size);
      iree_benchmark_use_ptr((char const volatile*)&data_offset);
    }
  }
  return iree_ok_status();
}

static iree_status_t BM_StructLayoutMacro_Aligned(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    for (int i = 0; i < 1000; ++i) {
      iree_host_size_t data_size = 4096;
      iree_host_size_t total = 0;
      iree_host_size_t header_offset = 0;
      iree_host_size_t data_offset = 0;

      iree_status_t status = IREE_STRUCT_LAYOUT(
          0, &total,
          IREE_STRUCT_FIELD_ALIGNED(1, test_buffer_header_t, iree_max_align_t,
                                    &header_offset),
          IREE_STRUCT_FIELD_ALIGNED(data_size, uint8_t, CACHE_LINE_ALIGNMENT,
                                    &data_offset));
      if (!iree_status_is_ok(status)) {
        return status;
      }

      iree_benchmark_use_ptr((char const volatile*)&total);
      iree_benchmark_use_ptr((char const volatile*)&data_offset);
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

// Basic allocation.
IREE_BENCHMARK_REGISTER(BM_MallocFree_Small);
IREE_BENCHMARK_REGISTER(BM_MallocFree_Medium);
IREE_BENCHMARK_REGISTER(BM_MallocFree_Large);
IREE_BENCHMARK_REGISTER(BM_MallocUninitializedFree_Medium);

// Realloc.
IREE_BENCHMARK_REGISTER(BM_ReallocGrow);
IREE_BENCHMARK_REGISTER(BM_ReallocShrink);

// Clone.
IREE_BENCHMARK_REGISTER(BM_Clone);

// Array allocation.
IREE_BENCHMARK_REGISTER(BM_MallocArray);
IREE_BENCHMARK_REGISTER(BM_MallocArrayUninitialized);
IREE_BENCHMARK_REGISTER(BM_ReallocArray);

// Struct allocation.
IREE_BENCHMARK_REGISTER(BM_MallocWithTrailing);
IREE_BENCHMARK_REGISTER(BM_MallocStructArray);

// Inline arena vs system allocator.
IREE_BENCHMARK_REGISTER(BM_InlineArenaMalloc);
IREE_BENCHMARK_REGISTER(BM_SystemMallocMultiple);

// Struct layout calculation (constant counts - should constant-fold).
IREE_BENCHMARK_REGISTER(BM_StructLayoutManual);
IREE_BENCHMARK_REGISTER(BM_StructLayoutMacro);

// Struct layout calculation (dynamic counts - runtime evaluation).
IREE_BENCHMARK_REGISTER(BM_StructLayoutManual_Dynamic);
IREE_BENCHMARK_REGISTER(BM_StructLayoutMacro_Dynamic);

// Struct layout calculation with alignment.
IREE_BENCHMARK_REGISTER(BM_StructLayoutManual_Aligned);
IREE_BENCHMARK_REGISTER(BM_StructLayoutMacro_Aligned);
