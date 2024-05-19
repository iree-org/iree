// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_BENCHMARK_H_
#define IREE_TESTING_BENCHMARK_H_

// This is a C API shim for a benchmark-like interface.
// The intent is that we can write benchmarks that are portable to bare-metal
// systems and use some simple tooling while also allowing them to run on
// the full benchmark library with all its useful reporting and statistics.

#include <math.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Benchmarking tools
//===----------------------------------------------------------------------===//

void iree_benchmark_use_ptr(char const volatile* x);

#if !defined(IREE_BENCHMARK_HAS_INLINE_ASSEMBLY)
#if defined(IREE_COMPILER_MSVC) || defined(IREE_PLATFORM_EMSCRIPTEN)
#define IREE_BENCHMARK_HAS_INLINE_ASSEMBLY 0
#elif defined(IREE_COMPILER_CLANG) || defined(IREE_COMPILER_GCC)
#define IREE_BENCHMARK_HAS_INLINE_ASSEMBLY 1
#else
#define IREE_BENCHMARK_HAS_INLINE_ASSEMBLY 0
#endif  // non-asm-targets
#endif  // !IREE_BENCHMARK_HAS_INLINE_ASSEMBLY

#if IREE_BENCHMARK_HAS_INLINE_ASSEMBLY == 0

#if defined(IREE_COMPILER_MSVC)
#define iree_benchmark_clobber() _ReadWriteBarrier()
#else
#define iree_benchmark_clobber()
#endif  // IREE_COMPILER_MSVC

#if defined(__cplusplus)
}  // extern "C"
template <typename T>
inline IREE_ATTRIBUTE_ALWAYS_INLINE void iree_optimization_barrier(T&& value) {
  iree_benchmark_use_ptr(&reinterpret_cast<char const volatile&>(value));
  iree_benchmark_clobber();
}
extern "C" {
#else
// TODO: a C-compatible optimization barrier.
#define iree_optimization_barrier(x)
#endif  // __cplusplus

#elif defined(IREE_COMPILER_CLANG)

#if defined(__cplusplus)
}  // extern "C"
inline IREE_ATTRIBUTE_ALWAYS_INLINE void iree_benchmark_clobber() {
  asm volatile("" : : : "memory");
}
template <typename T>
inline IREE_ATTRIBUTE_ALWAYS_INLINE void iree_optimization_barrier(T&& value) {
  asm volatile("" : "+r,m"(value) : : "memory");
}
extern "C" {
#else
// TODO: a C-compatible optimization barrier.
#define iree_optimization_barrier(x)
#endif  // __cplusplus

#elif defined(IREE_COMPILER_GCC)

#if defined(__cplusplus)
}  // extern "C"
inline IREE_ATTRIBUTE_ALWAYS_INLINE void iree_benchmark_clobber() {
  asm volatile("" : : : "memory");
}
template <typename T>
inline IREE_ATTRIBUTE_ALWAYS_INLINE
    typename std::enable_if<std::is_trivially_copyable<T>::value &&
                            (sizeof(T) <= sizeof(T*))>::type
    iree_optimization_barrier(T& value) {
  asm volatile("" : "+m,r"(value) : : "memory");
}
template <typename T>
inline IREE_ATTRIBUTE_ALWAYS_INLINE
    typename std::enable_if<!std::is_trivially_copyable<T>::value ||
                            (sizeof(T) > sizeof(T*))>::type
    iree_optimization_barrier(T& value) {
  asm volatile("" : "+m"(value) : : "memory");
}
template <typename T>
inline IREE_ATTRIBUTE_ALWAYS_INLINE
    typename std::enable_if<std::is_trivially_copyable<T>::value &&
                            (sizeof(T) <= sizeof(T*))>::type
    iree_optimization_barrier(T&& value) {
  asm volatile("" : "+m,r"(value) : : "memory");
}
template <typename T>
inline IREE_ATTRIBUTE_ALWAYS_INLINE
    typename std::enable_if<!std::is_trivially_copyable<T>::value ||
                            (sizeof(T) > sizeof(T*))>::type
    iree_optimization_barrier(T&& value) {
  asm volatile("" : "+m"(value) : : "memory");
}
extern "C" {
#else
// TODO: a C-compatible optimization barrier.
#define iree_optimization_barrier(x)
#endif  // __cplusplus

#endif  // IREE_BENCHMARK_HAS_INLINE_ASSEMBLY

//===----------------------------------------------------------------------===//
// iree_benchmark_state_t
//===----------------------------------------------------------------------===//

// Benchmark state manipulator.
// Passed to each benchmark during execution to control the benchmark state
// or append information beyond just timing.
typedef struct iree_benchmark_state_t {
  // Internal implementation handle.
  void* impl;

  // Allocator that can be used for host allocations required during benchmark
  // execution.
  iree_allocator_t host_allocator;
} iree_benchmark_state_t;

// Returns a range argument with the given ordinal.
int64_t iree_benchmark_get_range(iree_benchmark_state_t* state,
                                 iree_host_size_t ordinal);

// Returns true while the benchmark should keep running its step loop.
//
// Usage:
//  while (iree_benchmark_keep_running(state, 1000)) {
//    // process 1000 elements
//  }
bool iree_benchmark_keep_running(iree_benchmark_state_t* state,
                                 uint64_t batch_count);

// Reports that the currently executing benchmark cannot be run.
// Callers should return after calling as further benchmark-related calls may
// fail.
void iree_benchmark_skip(iree_benchmark_state_t* state, const char* message);

// Suspends the benchmark timer until iree_benchmark_resume_timing is called.
// This can be used to guard per-step code that is required to initialze the
// work but not something that needs to be accounted for in the benchmark
// timing. Introduces non-trivial overhead: only use this ~once per step when
// then going on to perform large amounts of batch work in the step.
void iree_benchmark_pause_timing(iree_benchmark_state_t* state);

// Resumes the benchmark timer after a prior iree_benchmark_suspend_timing.
void iree_benchmark_resume_timing(iree_benchmark_state_t* state);

// Sets a label string that will be displayed alongside the report line from the
// currently executing benchmark.
void iree_benchmark_set_label(iree_benchmark_state_t* state, const char* label);

// Adds a 'bytes/s' label with the given value.
//
// REQUIRES: must only be called outside of the benchmark step loop.
void iree_benchmark_set_bytes_processed(iree_benchmark_state_t* state,
                                        int64_t bytes);

// Adds an `items/s` label with the given value.
//
// REQUIRES: must only be called outside of the benchmark step loop.
void iree_benchmark_set_items_processed(iree_benchmark_state_t* state,
                                        int64_t items);

//===----------------------------------------------------------------------===//
// iree_benchmark_def_t
//===----------------------------------------------------------------------===//

enum iree_benchmark_flag_bits_t {
  IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME = 1u << 0,

  IREE_BENCHMARK_FLAG_USE_REAL_TIME = 1u << 1,
  IREE_BENCHMARK_FLAG_USE_MANUAL_TIME = 1u << 2,
};
typedef uint32_t iree_benchmark_flags_t;

typedef enum iree_benchmark_unit_e {
  IREE_BENCHMARK_UNIT_MILLISECOND = 0,
  IREE_BENCHMARK_UNIT_MICROSECOND,
  IREE_BENCHMARK_UNIT_NANOSECOND,
} iree_benchmark_unit_t;

typedef struct iree_benchmark_def_t iree_benchmark_def_t;

typedef iree_status_t(IREE_API_PTR* iree_benchmark_fn_t)(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state);

// A benchmark case definition.
struct iree_benchmark_def_t {
  // IREE_BENCHMARK_FLAG_* bitmask controlling benchmark behavior and reporting.
  iree_benchmark_flags_t flags;

  // Time unit used in display.
  iree_benchmark_unit_t time_unit;  // MILLISECOND by default

  // Optional minimum duration the benchmark should run for in nanoseconds.
  iree_duration_t minimum_duration_ns;  // 0 if unspecified to autodetect
  // Optional iteration count the benchmark should run for.
  uint64_t iteration_count;  // 0 if unspecified to autodetect

  // TODO(benvanik): add range arguments.

  // Runs the benchmark to completion.
  // Implementations must call iree_benchmark_keep_running in a loop until it
  // returns false.
  iree_benchmark_fn_t run;

  // User-defined data accessible in the run function.
  const void* user_data;
};

// Registers a benchmark with the given definition.
const iree_benchmark_def_t* iree_benchmark_register(
    iree_string_view_t name, const iree_benchmark_def_t* benchmark_def);

//===----------------------------------------------------------------------===//
// Benchmark registration utilities
//===----------------------------------------------------------------------===//

#define IREE_BENCHMARK_IMPL_NAME_(name) \
  IREE_BENCHMARK_IMPL_CONCAT_(iree_benchmark_, __COUNTER__, name)
#define IREE_BENCHMARK_IMPL_CONCAT_(a, b, c) \
  IREE_BENCHMARK_IMPL_CONCAT2_(a, b, c)
#define IREE_BENCHMARK_IMPL_CONCAT2_(a, b, c) a##b##c

#define IREE_BENCHMARK_FN(name)                                        \
  static iree_status_t name(const iree_benchmark_def_t* benchmark_def, \
                            iree_benchmark_state_t* benchmark_state)

// Allocates a benchmark definition for the given function and returns it.
// The returned pointer is safe to store in a static variable.
// TODO(benvanik): allow optionally passing flags with variadic macros.
iree_benchmark_def_t* iree_make_function_benchmark(iree_benchmark_fn_t fn);

// TODO(benvanik): find a way to make this C-compatible.
// Today this requires C++ in order to initialize the benchmark via the function
// and C disallows this. We can probably use some tricky attributes to run
// functions instead.
//
// Defines a benchmark of a function with default parameters.
//
// Example:
//  IREE_BENCHMARK_FN(my_benchmark) {
//    while (iree_benchmark_keep_running(benchmark_state, 1000)) {
//      // process 1000 elements
//    }
//    return iree_ok_status();
//  }
//  IREE_BENCHMARK_REGISTER(my_benchmark);
#define IREE_BENCHMARK_REGISTER(name)                                         \
  static const iree_benchmark_def_t* IREE_BENCHMARK_IMPL_NAME_(name)          \
      IREE_ATTRIBUTE_UNUSED = (iree_benchmark_def_t*)iree_benchmark_register( \
          iree_make_cstring_view(#name), iree_make_function_benchmark(name))

//===----------------------------------------------------------------------===//
// Benchmark infra management
//===----------------------------------------------------------------------===//

// Initializes the benchmark framework.
// Must be called before any other iree_benchmark_* functions.
void iree_benchmark_initialize(int* argc, char** argv);

// Runs all registered benchmarks specified by the command line flags.
// Must be called after iree_benchmark_initialize and zero or more benchmarks
// have been registered with iree_benchmark_register.
void iree_benchmark_run_specified(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TESTING_BENCHMARK_H_
