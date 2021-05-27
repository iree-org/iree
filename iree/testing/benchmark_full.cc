// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "benchmark/benchmark.h"
#include "iree/base/tracing.h"
#include "iree/testing/benchmark.h"

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
  IREE_TRACE_SCOPE_DYNAMIC(benchmark_name);
  IREE_TRACE_FRAME_MARK();

  iree_benchmark_state_t state;
  memset(&state, 0, sizeof(state));
  state.impl = &benchmark_state;
  state.host_allocator = iree_allocator_system();

  iree_status_t status = benchmark_def->run(&state);
  if (!iree_status_is_ok(status)) {
    auto status_str = StatusToString(status);
    iree_status_ignore(status);
    benchmark_state.SkipWithError(status_str.c_str());
  }
}

void iree_benchmark_register(iree_string_view_t name,
                             const iree_benchmark_def_t* benchmark_def) {
  std::string name_str(name.data, name.size);
  std::string prefixed_str = "BM_" + name_str;
  auto* instance = benchmark::RegisterBenchmark(
      prefixed_str.c_str(),
      [name_str, benchmark_def](benchmark::State& state) -> void {
        iree_benchmark_run(name_str.c_str(), benchmark_def, state);
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
    instance->MinTime((double)benchmark_def->minimum_duration_ns / 1e-9);
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
}

//===----------------------------------------------------------------------===//
// Benchmark infra management
//===----------------------------------------------------------------------===//

void iree_benchmark_initialize(int* argc, char** argv) {
  benchmark::Initialize(argc, argv);
}

void iree_benchmark_run_specified(void) { benchmark::RunSpecifiedBenchmarks(); }
