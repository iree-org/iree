// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/debugging.h"
#include "iree/testing/benchmark.h"

//===----------------------------------------------------------------------===//
// Benchmarking tools
//===----------------------------------------------------------------------===//

// NOTE: LTO will kill this (most likely) - google benchmark warns about this
// as well but has been letting it ride for years :shrug:
void iree_benchmark_use_ptr(char const volatile* x) {}

//===----------------------------------------------------------------------===//
// iree_benchmark_state_t
//===----------------------------------------------------------------------===//

benchmark::State& GetBenchmarkState(iree_benchmark_state_t* state) {
  return *(benchmark::State*)state->impl;
}

int64_t iree_benchmark_get_range(iree_benchmark_state_t* state,
                                 iree_host_size_t ordinal) {
  auto& s = GetBenchmarkState(state);
  return s.range(ordinal);
}

bool iree_benchmark_keep_running(iree_benchmark_state_t* state,
                                 uint64_t batch_count) {
  auto& s = GetBenchmarkState(state);
  return s.KeepRunningBatch(batch_count);
}

void iree_benchmark_skip(iree_benchmark_state_t* state, const char* message) {
  auto& s = GetBenchmarkState(state);
  s.SkipWithError(message);
}

void iree_benchmark_pause_timing(iree_benchmark_state_t* state) {
  auto& s = GetBenchmarkState(state);
  s.PauseTiming();
}

void iree_benchmark_resume_timing(iree_benchmark_state_t* state) {
  auto& s = GetBenchmarkState(state);
  s.ResumeTiming();
}

void iree_benchmark_set_label(iree_benchmark_state_t* state,
                              const char* label) {
  auto& s = GetBenchmarkState(state);
  s.SetLabel(label);
}

void iree_benchmark_set_bytes_processed(iree_benchmark_state_t* state,
                                        int64_t bytes) {
  auto& s = GetBenchmarkState(state);
  s.SetBytesProcessed(bytes);
}

void iree_benchmark_set_items_processed(iree_benchmark_state_t* state,
                                        int64_t items) {
  auto& s = GetBenchmarkState(state);
  s.SetItemsProcessed(items);
}

//===----------------------------------------------------------------------===//
// iree_benchmark_def_t
//===----------------------------------------------------------------------===//

static std::string StatusToString(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return "OK";
  }
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return "<!>";
  }
  std::string result(buffer_length, '\0');
  if (IREE_UNLIKELY(!iree_status_format(status, result.size() + 1,
                                        const_cast<char*>(result.data()),
                                        &buffer_length))) {
    return "<!>";
  }
  return result;
}

static void iree_benchmark_run(const char* benchmark_name,
                               const iree_benchmark_def_t* benchmark_def,
                               benchmark::State& benchmark_state) {
  IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, benchmark_name,
                                      strlen(benchmark_name));
  IREE_TRACE_FRAME_MARK();

  iree_benchmark_state_t state;
  memset(&state, 0, sizeof(state));
  state.impl = &benchmark_state;
  state.host_allocator = iree_allocator_system();

  iree_status_t status = benchmark_def->run(benchmark_def, &state);
  if (!iree_status_is_ok(status)) {
    auto status_str = StatusToString(status);
    iree_status_ignore(status);
    benchmark_state.SkipWithError(status_str.c_str());
  }

  IREE_TRACE_ZONE_END(z0);
}

const iree_benchmark_def_t* iree_benchmark_register(
    iree_string_view_t name, const iree_benchmark_def_t* benchmark_def) {
  std::string name_str(name.data, name.size);
  std::string prefixed_str = !iree_string_view_starts_with(name, IREE_SV("BM_"))
                                 ? "BM_" + name_str
                                 : name_str;
  iree_benchmark_def_t cloned_def = *benchmark_def;
  auto* instance = benchmark::RegisterBenchmark(
      prefixed_str.c_str(),
      [name_str, cloned_def](benchmark::State& state) -> void {
        iree_benchmark_run(name_str.c_str(), &cloned_def, state);
      });

  if (iree_all_bits_set(benchmark_def->flags,
                        IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME)) {
    instance->MeasureProcessCPUTime();
  }
  if (iree_all_bits_set(benchmark_def->flags,
                        IREE_BENCHMARK_FLAG_USE_REAL_TIME)) {
    instance->UseRealTime();
  }
  if (iree_all_bits_set(benchmark_def->flags,
                        IREE_BENCHMARK_FLAG_USE_MANUAL_TIME)) {
    instance->UseManualTime();
  }

  if (benchmark_def->minimum_duration_ns != 0) {
    instance->MinTime((double)benchmark_def->minimum_duration_ns * 1e-9);
  } else if (benchmark_def->iteration_count != 0) {
    instance->Iterations(benchmark_def->iteration_count);
  }

  switch (benchmark_def->time_unit) {
    default:
    case IREE_BENCHMARK_UNIT_MILLISECOND:
      instance->Unit(benchmark::kMillisecond);
      break;
    case IREE_BENCHMARK_UNIT_MICROSECOND:
      instance->Unit(benchmark::kMicrosecond);
      break;
    case IREE_BENCHMARK_UNIT_NANOSECOND:
      instance->Unit(benchmark::kNanosecond);
      break;
  }

  return benchmark_def;
}

iree_benchmark_def_t* iree_make_function_benchmark(iree_benchmark_fn_t fn) {
  // Go straight to malloc for this implementation as benchmark does the same
  // thing. This also has the benefit of hiding the startup allocation from
  // our stats >_>
  iree_benchmark_def_t* def = NULL;
  IREE_LEAK_CHECK_DISABLE_PUSH();
  def = (iree_benchmark_def_t*)calloc(1, sizeof(*def));
  IREE_LEAK_CHECK_DISABLE_POP();

  def->run = fn;

  // Return with no expectation of it ever being freed.
  return def;
}

//===----------------------------------------------------------------------===//
// Benchmark infra management
//===----------------------------------------------------------------------===//

void iree_benchmark_initialize(int* argc, char** argv) {
  benchmark::Initialize(argc, argv);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  // clang-format off
  fprintf(stderr,
"\x1b[31m"
"===----------------------------------------------------------------------===\n"
"\n"
"         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████\n"
"         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██\n"
"         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███\n"
"         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██\n"
"          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████\n"
"\n"
"===----------------------------------------------------------------------===\n"
"\n"
"Tracing is enabled and will skew your results!\n"
"The timings involved here can be an order of magnitude off due to the\n"
"tracing time sampling, recording, and instrumentation overhead.\n"
"Disable tracing with IREE_ENABLE_RUNTIME_TRACING=OFF and rebuild.\n"
"\x1b[0m"
"\n"
  );
  fflush(stderr);
  // clang-format on
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
}

void iree_benchmark_run_specified(void) { benchmark::RunSpecifiedBenchmarks(); }
