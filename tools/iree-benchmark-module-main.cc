// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// iree-benchmark-module: benchmarks public functions in an IREE VM module
//===----------------------------------------------------------------------===//
//
// This runs exported functions using flags specified on the command line.
// Each function is measured independently and the numbers reported will be for
// the full end-to-end CPU and wall times.
//
// From an ML perspective this is an integration benchmark for measuring total
// user-visible latency of model entry points. It is *not* a microbenchmarking
// tool for individual device-side dispatch functions (aka ops aka kernels).
// --dispatch_statistics adds a breakdown by dispatch function, collected with a
// lightweight HAL profiling session in a separate untimed pass that runs the
// program again; it shows where time goes rather than timing a kernel.
// If interested in the precise time of a particular dispatch then tracy,
// executable_library_benchmark, and platform/vendor tooling (nsight, perf, etc)
// are to be used instead and attaching them to this tool is often useful in
// order to get a large sample set.
//
// By default all functions taking no inputs will be benchmarked. If a function
// takes inputs then the user will need to specify them using --input=
// flags. Depending on the input program the -iree-flow-export-benchmark-funcs
// flag can be passed to the compiler to attempt to wrap each function with
// dummy inputs however this will fail in programs with dynamically shaped
// inputs. The workaround for avoiding the need for flags is to provide the
// input program in a form with no inputs from the start.
//
// It's important to remember that IREE is not a BLAS library and is meant to
// run entire programs. It's not generally appropriate to benchmark a model with
// a single matmul, for example, as that's just treating IREE as a BLAS library.
// Note also that user-level ops in a frontend environment don't map to the
// dispatches that IREE executes: IREE is a compiler like any other and does not
// guarantee a source line of code translates into an atomically divisible and
// independently measurable execution command. In other words don't expect to be
// able to benchmark the cost of a broadcasting elementwise tf.add op within a
// model: by the time we are running the program that's fused itself into a
// single machine instruction operating as part of some other ops.
//
// For coarse dispatch testing and triaging it can still be useful to remove
// some of the overheads introduced by whole-program execution and the compiler
// flag --iree-hal-benchmark-dispatch-repeat-count=N is provided to enable
// batching. Whatever N is chosen must then be passed to this tool via
// --batch_size=N so that the benchmark reporting properly reflects the
// batching. As an example --iree-hal-benchmark-dispatch-repeat-count=32 +
// --batch_size=32 will reduce the overheads by 32x. Think of this as a way to
// control the p value in Amdahl's law representing the amount of time spent in
// dispatches relative to the rest of the program. This isn't representative of
// how the full program will run, though, and YMMV. Always verify timings with
// an appropriate device-specific tool before trusting the more generic and
// higher-level numbers from this tool.

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/recorder.h"
#include "iree/hal/utils/statistics_sink.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/process_results.h"
#include "iree/vm/api.h"

constexpr char kNanosecondsUnitString[] = "ns";
constexpr char kMicrosecondsUnitString[] = "us";
constexpr char kMillisecondsUnitString[] = "ms";

// TODO(hanchung): Extract the batch size using
// iree_vm_function_lookup_attr_by_name.
IREE_FLAG(int32_t, batch_size, 1,
          "Number of invocations per iteration, which for dispatch benchmarks "
          "must match the --iree-hal-benchmark-dispatch-repeat-count value "
          "used during compilation.");
IREE_FLAG(int32_t, batch_concurrency, 1,
          "Number of invocations within a batch that should run concurrently.");
IREE_FLAG(bool, agents_md, false,
          "Prints AGENTS.md guidance for iree-benchmark-module replay capture "
          "and exits.");

IREE_FLAG(string, function, "",
          "Name of a function contained in the module specified by --module= "
          "to run. If this is not set, all the exported functions will be "
          "benchmarked and they are expected to not have input arguments.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

IREE_FLAG(
    bool, dispatch_statistics, false,
    "Adds a row per dispatch function under each benchmark result with the\n"
    "time spent in that function per benchmark iteration. The rows come from\n"
    "a separate untimed pass that reruns the benchmark with a lightweight HAL\n"
    "profiling session active after each measured repetition. Dispatch rows\n"
    "average all profiled repetitions and also report, in seconds, the mean,\n"
    "min and max duration of one call and the stddev of the per-call mean\n"
    "across profiled batches. Requires a backend whose lightweight\n"
    "profiling attributes dispatches in existing command buffers (currently\n"
    "local-task and local-sync); elsewhere the session fails to start or the\n"
    "rows are missing with a warning. Cannot combine with\n"
    "--device_profiling_mode, --print_device_statistics,\n"
    "--device_capture_tool, or --device_replay_output.");

IREE_FLAG_LIST(
    string, input,
    "An input value or buffer of the format:\n"
    "  [shape]xtype=[value]\n"
    "  --input=\"2x2xi32=1 2 3 4\"\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  --input=\"2x2xi32=[[1 2][3 4]]\"\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  --input=2x2xi32=@some/file.bin\n"
    "numpy npy files (from numpy.save) can be read to provide 1+ values:\n"
    "  --input=@some.npy\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

static iree_status_t parse_time_unit(iree_string_view_t flag_name,
                                     void* storage, iree_string_view_t value) {
  auto* unit = (std::pair<bool, benchmark::TimeUnit>*)storage;
  auto unit_string = std::string(value.data, value.size);
  if (unit_string == kMillisecondsUnitString) {
    *unit = {true, benchmark::kMillisecond};
    return iree_ok_status();
  } else if (unit_string == kMicrosecondsUnitString) {
    *unit = {true, benchmark::kMicrosecond};
    return iree_ok_status();
  } else if (unit_string == kNanosecondsUnitString) {
    *unit = {true, benchmark::kNanosecond};
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported time unit");
}
static void print_time_unit(iree_string_view_t flag_name, void* storage,
                            FILE* file) {
  auto* unit = (std::pair<bool, benchmark::TimeUnit>*)storage;
  if (!unit->first) {
    return;
  }
  std::string unit_string;
  switch (unit->second) {
    case benchmark::kMillisecond:
      unit_string = kMillisecondsUnitString;
      break;
    case benchmark::kMicrosecond:
      unit_string = kMicrosecondsUnitString;
      break;
    case benchmark::kNanosecond:
      unit_string = kNanosecondsUnitString;
      break;
    default:
      assert(false && "Unexpected time unit.");
  }
  fprintf(file, "--%.*s=\"%s\"\n", (int)flag_name.size, flag_name.data,
          unit_string.c_str());
}
// Time unit to be printed. If the first field is false, each place will use its
// default time unit.
static std::pair<bool, benchmark::TimeUnit> FLAG_time_unit = {
    false, benchmark::kNanosecond};
IREE_FLAG_CALLBACK(
    parse_time_unit, print_time_unit, &FLAG_time_unit, time_unit,
    "The time unit to be printed in the results. Can be 'ms', 'us', or 'ns'.");

IREE_FLAG(
    bool, enable_output_processing, false,
    "Enable keeping outputs of last benchmark iteration and processing "
    "those. This needs to be enabled for --output* options to be effective.");

namespace iree {
namespace {

static const char kIreeBenchmarkModuleUsage[] =
    "Benchmarks exported functions from a compiled IREE module.\n"
    "\n"
    "Replay capture wraps the resolved HAL device group after normal "
    "--device=\n"
    "selection. Use --device_replay_output=path.ireereplay to record the HAL\n"
    "work issued by the benchmark. The recorder is closed after benchmark\n"
    "execution so the output file is complete when the tool exits.\n"
    "\n"
    "Replay capture flags:\n"
    "  --device_replay_output=path.ireereplay\n"
    "      Writes a HAL replay stream.\n"
    "  --device_replay_file_policy=reference|capture-ranges|capture-all|fail\n"
    "      Controls imported fd-backed HAL files such as parameter archives.\n"
    "  --device_replay_file_validation=identity|digest\n"
    "      Validation for referenced fd-backed files.\n"
    "  --agents_md\n"
    "      Prints AGENTS.md guidance specific to iree-benchmark-module "
    "capture.\n"
    "      Use `iree-run-replay --agents_md` for the full replay tool "
    "playbook.\n";

static void PrintBenchmarkModuleAgentMarkdown(FILE* file) {
  fputs(
      "# iree-benchmark-module Replay Capture\n"
      "\n"
      "`iree-benchmark-module` can capture the HAL work issued by benchmark\n"
      "iterations with `--device_replay_output=/path/to/model.ireereplay`.\n"
      "Capture flags compose with normal Google Benchmark controls.\n"
      "\n"
      "Synchronous and dispatch benchmarks record replay `execute` scopes "
      "around\n"
      "the VM invoke/list reset body. Asynchronous benchmarks record the "
      "scope\n"
      "around the resumed timing interval and keep batch setup and cleanup "
      "outside\n"
      "the selected scope. Use `iree-benchmark-replay --replay_scope=execute` "
      "to\n"
      "time the same region later while replay still executes the full "
      "captured\n"
      "stream.\n"
      "\n"
      "Use `--device_replay_file_policy=reference` for large stable parameter\n"
      "archives and `--device_replay_file_validation=identity` unless the "
      "files\n"
      "will move across filesystems and need digest validation.\n"
      "\n"
      "For replay execution, executable substitution, file remapping, dump "
      "JSONL,\n"
      "and the shared replay failure contract, pipe `iree-run-replay "
      "--agents_md`\n"
      "into your AGENTS.md.\n",
      file);
}

// A workload owns the ABI and iteration accounting resolved during discovery.
// The runner uses the same workload for measurement and profiling.
enum class InvocationKind { kSync, kAsync, kDispatch };

// Running mean and variance of a series (Welford's method).
struct RunningStats {
  uint64_t count = 0;
  double mean = 0.0;
  double m2 = 0.0;

  void Add(double value) {
    ++count;
    const double delta = value - mean;
    mean += delta / count;
    m2 += delta * (value - mean);
  }
  // Combines the statistics of a disjoint series (Chan et al.).
  void Merge(const RunningStats& other) {
    if (!other.count) return;
    if (!count) {
      *this = other;
      return;
    }
    const double total = static_cast<double>(count + other.count);
    const double delta = other.mean - mean;
    mean += delta * other.count / total;
    m2 += other.m2 + delta * delta * count * other.count / total;
    count += other.count;
  }
  double stddev() const {
    return count > 1 ? std::sqrt(m2 / (count - 1)) : 0.0;
  }
};

struct DispatchRow {
  uint64_t calls = 0;
  uint64_t duration_ns = 0;
  uint64_t tile_duration_ns = 0;
  // Shortest and longest single call.
  uint64_t min_ns = UINT64_MAX;
  uint64_t max_ns = 0;
  // Per-call mean of each profiled batch (loop iteration of the benchmark).
  RunningStats batch_means;
};

// Per-dispatch results accumulated over all profiled repetitions.
struct DispatchProfile {
  std::map<std::string, DispatchRow> rows;
  int64_t iterations = 0;
  uint64_t dropped_records = 0;
  uint64_t unscaled_rows = 0;
};

struct Workload {
  std::string name;
  iree_vm_function_t function = {};
  std::string results_cconv;
  InvocationKind kind = InvocationKind::kSync;
  int32_t batch_size = 1;
  int32_t concurrency = 1;
  benchmark::TimeUnit unit = benchmark::kMillisecond;
  vm::ref<iree_vm_list_t> inputs;
  vm::ref<iree_vm_list_t> outputs;
  DispatchProfile profile;
};

struct Invocation {
  vm::ref<iree_vm_list_t> inputs;
  vm::ref<iree_vm_list_t> outputs;
};

// Owns one batch's arguments, outputs and synchronization. Only the async ABI
// requires host-side batching; dispatch wrappers repeat inside the VM call.
struct Batch {
  std::vector<Invocation> invocations;
  vm::ref<iree_hal_fence_t> completion;

  iree_status_t Prepare(const Workload& workload, iree_hal_device_t* device) {
    iree_allocator_t allocator = iree_allocator_system();
    const bool async = workload.kind == InvocationKind::kAsync;
    const int count = async ? workload.batch_size : 1;
    invocations.resize(count);
    std::vector<vm::ref<iree_hal_semaphore_t>> timelines;
    std::vector<vm::ref<iree_hal_fence_t>> previous;
    if (async) {
      timelines.resize(workload.concurrency);
      previous.resize(workload.concurrency);
      IREE_RETURN_IF_ERROR(
          iree_hal_fence_create(workload.concurrency, allocator, &completion));
      for (auto& timeline : timelines) {
        IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
            device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline));
        IREE_RETURN_IF_ERROR(iree_hal_fence_insert(
            completion.get(), timeline.get(), count / workload.concurrency));
      }
    }
    for (int i = 0; i < count; ++i) {
      auto& invocation = invocations[i];
      IREE_RETURN_IF_ERROR(
          iree_vm_list_create(iree_vm_make_undefined_type_def(), 16, allocator,
                              &invocation.outputs));
      if (!async) {
        invocation.inputs = vm::retain_ref(workload.inputs.get());
        continue;
      }
      const int track = i % workload.concurrency;
      vm::ref<iree_hal_fence_t> signal;
      IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
          timelines[track].get(), i / workload.concurrency + 1, allocator,
          &signal));
      IREE_RETURN_IF_ERROR(iree_vm_list_clone(workload.inputs.get(), allocator,
                                              &invocation.inputs));
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_move(invocation.inputs.get(), previous[track]));
      previous[track] = vm::retain_ref(signal);
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_move(invocation.inputs.get(), signal));
    }
    return iree_ok_status();
  }

  iree_status_t Execute(const Workload& workload, iree_vm_context_t* context) {
    for (auto& invocation : invocations) {
      // Preserve the measured lifetime of outputs: ordinary functions release
      // the preceding results before invoking; dispatch wrappers clear after.
      if (workload.kind == InvocationKind::kSync) {
        IREE_RETURN_IF_ERROR(iree_vm_list_resize(invocation.outputs.get(), 0));
      }
      IREE_RETURN_IF_ERROR(iree_vm_invoke(
          context, workload.function, IREE_VM_INVOCATION_FLAG_NONE, nullptr,
          invocation.inputs.get(), invocation.outputs.get(),
          iree_allocator_system()));
      if (workload.kind == InvocationKind::kDispatch) {
        IREE_RETURN_IF_ERROR(iree_vm_list_resize(invocation.outputs.get(), 0));
      }
    }
    return completion
               ? iree_hal_fence_wait(completion.get(), iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE)
               : iree_ok_status();
  }
};

// Forwards profile chunks to a statistics sink under a lock. The statistics
// sink does not synchronize readers with producers, which may write from their
// own threads, and the profiler reads its rows between batches.
struct LockedStatisticsSink {
  iree_hal_resource_t resource;
  iree_slim_mutex_t mutex;
  iree_hal_profile_statistics_sink_t* statistics = nullptr;

  static iree_status_t Create(LockedStatisticsSink** out_sink) {
    static const iree_hal_profile_sink_vtable_t vtable = {
        /*.destroy=*/Destroy,
        /*.begin_session=*/BeginSession,
        /*.write=*/Write,
        /*.end_session=*/EndSession,
    };
    auto sink = std::make_unique<LockedStatisticsSink>();
    IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_sink_create(
        iree_allocator_system(), &sink->statistics));
    iree_slim_mutex_initialize(&sink->mutex);
    iree_hal_resource_initialize(&vtable, &sink->resource);
    *out_sink = sink.release();
    return iree_ok_status();
  }

  iree_hal_profile_sink_t* base() {
    return reinterpret_cast<iree_hal_profile_sink_t*>(this);
  }
  iree_hal_profile_sink_t* forward() {
    return iree_hal_profile_statistics_sink_base(statistics);
  }
  static LockedStatisticsSink* Cast(iree_hal_profile_sink_t* sink) {
    return reinterpret_cast<LockedStatisticsSink*>(sink);
  }

 private:
  static void Destroy(iree_hal_profile_sink_t* base_sink) {
    LockedStatisticsSink* sink = Cast(base_sink);
    iree_hal_profile_statistics_sink_release(sink->statistics);
    iree_slim_mutex_deinitialize(&sink->mutex);
    delete sink;
  }
  static iree_status_t BeginSession(
      iree_hal_profile_sink_t* base_sink,
      const iree_hal_profile_chunk_metadata_t* metadata) {
    LockedStatisticsSink* sink = Cast(base_sink);
    iree_slim_mutex_lock(&sink->mutex);
    iree_status_t status =
        iree_hal_profile_sink_begin_session(sink->forward(), metadata);
    iree_slim_mutex_unlock(&sink->mutex);
    return status;
  }
  static iree_status_t Write(iree_hal_profile_sink_t* base_sink,
                             const iree_hal_profile_chunk_metadata_t* metadata,
                             iree_host_size_t iovec_count,
                             const iree_const_byte_span_t* iovecs) {
    LockedStatisticsSink* sink = Cast(base_sink);
    iree_slim_mutex_lock(&sink->mutex);
    iree_status_t status = iree_hal_profile_sink_write(
        sink->forward(), metadata, iovec_count, iovecs);
    iree_slim_mutex_unlock(&sink->mutex);
    return status;
  }
  static iree_status_t EndSession(
      iree_hal_profile_sink_t* base_sink,
      const iree_hal_profile_chunk_metadata_t* metadata,
      iree_status_code_t session_status_code) {
    LockedStatisticsSink* sink = Cast(base_sink);
    iree_slim_mutex_lock(&sink->mutex);
    iree_status_t status = iree_hal_profile_sink_end_session(
        sink->forward(), metadata, session_status_code);
    iree_slim_mutex_unlock(&sink->mutex);
    return status;
  }
};

// Runs a lightweight HAL statistics session on every device during Google
// Benchmark's untimed profiling pass, which reruns the registered benchmark
// after each measured repetition, and adds the per-dispatch rows to the
// workload that pass runs.
class DispatchProfiler final : public benchmark::ProfilerManager {
 public:
  explicit DispatchProfiler(iree_hal_device_list_t* devices)
      : devices_(devices) {}
  ~DispatchProfiler() {
    iree_status_ignore(End(/*workload=*/nullptr));
    iree_status_ignore(status_);
  }

  // Starts and ends a session so unsupported backends fail before measuring.
  // Attribution also depends on backend metadata; an empty result is diagnosed
  // after execution.
  iree_status_t Probe() {
    IREE_RETURN_IF_ERROR(Begin());
    return End(/*workload=*/nullptr);
  }

  // Selects the workload and state of the next benchmark function run.
  void Attach(Workload* workload, benchmark::State* state) {
    workload_ = workload;
    state_ = state;
    in_pass_ = false;
  }
  void Detach() { Attach(nullptr, nullptr); }

  // True from the start of a profiling pass until the next Attach.
  bool in_pass() const { return in_pass_; }
  // The session of the running profiling pass, or NULL.
  iree_hal_profiling_session_t* session() const { return session_; }
  bool failed() const { return !iree_status_is_ok(status_); }
  // Returns and clears the failures of all profiling passes.
  iree_status_t ConsumeStatus() {
    return std::exchange(status_, iree_ok_status());
  }

  // Adds the per-call mean of each function's calls since the previous batch.
  // Called after each flushed batch of the profiling pass.
  void SampleBatch() {
    std::map<std::string, DispatchRow> totals;
    if (!Check(Collect(&totals, /*unscaled_rows=*/nullptr))) return;
    for (const auto& [key, total] : totals) {
      DispatchRow& previous = batch_totals_[key];
      const uint64_t calls = total.calls - previous.calls;
      if (calls) {
        pass_batch_means_[key].Add(
            static_cast<double>(total.duration_ns - previous.duration_ns) /
            calls);
      }
      previous = total;
    }
  }

  void AfterSetupStart() override {
    in_pass_ = true;
    Check(Begin());
  }
  void BeforeTeardownStop() override {
    // A skipped pass may have stopped early, so its rows are dropped.
    const bool completed = !state_->skipped();
    if (Check(End(completed ? workload_ : nullptr)) && completed) {
      workload_->profile.iterations += state_->iterations();
    }
  }

 private:
  bool Check(iree_status_t status) {
    if (iree_status_is_ok(status)) return true;
    if (state_) {
      state_->SkipWithError(iree::Status(iree_status_clone(status)).ToString());
    }
    status_ = iree_status_join(status_, status);
    return false;
  }

  iree_status_t Begin() {
    batch_totals_.clear();
    pass_batch_means_.clear();
    IREE_RETURN_IF_ERROR(LockedStatisticsSink::Create(&sink_));
    iree_hal_device_profiling_options_t options = {};
    options.flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
    options.sink = sink_->base();
    return iree_hal_profiling_session_begin(
        devices_->count, devices_->devices, &options,
        /*external_options=*/nullptr, /*flush_interval_ms=*/0,
        iree_allocator_system(), &session_);
  }

  // Ends the session and adds its rows to |workload| unless it is NULL.
  iree_status_t End(Workload* workload) {
    iree_status_t status =
        iree_hal_profiling_session_end(std::exchange(session_, nullptr));
    if (!sink_) return status;
    if (workload && iree_status_is_ok(status)) {
      DispatchProfile& profile = workload->profile;
      profile.dropped_records +=
          iree_hal_profile_statistics_sink_dropped_record_count(
              sink_->statistics);
      std::map<std::string, DispatchRow> totals;
      status = Collect(&totals, &profile.unscaled_rows);
      for (const auto& [key, total] : totals) {
        DispatchRow& result = profile.rows[key];
        result.calls += total.calls;
        result.duration_ns += total.duration_ns;
        result.tile_duration_ns += total.tile_duration_ns;
        result.min_ns = std::min(result.min_ns, total.min_ns);
        result.max_ns = std::max(result.max_ns, total.max_ns);
        result.batch_means.Merge(pass_batch_means_[key]);
      }
    }
    iree_hal_profile_sink_release(std::exchange(sink_, nullptr)->base());
    return status;
  }

  struct Collector {
    iree_hal_profile_statistics_sink_t* sink;
    std::map<std::string, DispatchRow>* totals;
    uint64_t unscaled_rows;
  };

  // Sums the current function rows of all devices by function name.
  iree_status_t Collect(std::map<std::string, DispatchRow>* out_totals,
                        uint64_t* unscaled_rows) {
    Collector collector = {sink_->statistics, out_totals, 0};
    iree_hal_profile_statistics_row_callback_t callback = {CollectRow,
                                                           &collector};
    iree_slim_mutex_lock(&sink_->mutex);
    iree_status_t status = iree_hal_profile_statistics_sink_for_each_row(
        sink_->statistics, callback);
    iree_slim_mutex_unlock(&sink_->mutex);
    if (unscaled_rows) *unscaled_rows += collector.unscaled_rows;
    return status;
  }

  static iree_status_t CollectRow(
      void* data, const iree_hal_profile_statistics_row_t* row) {
    Collector& collector = *static_cast<Collector*>(data);
    if (row->row_type !=
            IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_FUNCTION &&
        row->row_type !=
            IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_FUNCTION) {
      return iree_ok_status();
    }
    uint64_t duration = 0, min_ns = 0, max_ns = 0;
    if (!iree_all_bits_set(row->flags,
                           IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING) ||
        !iree_hal_profile_statistics_sink_scale_duration_to_ns(
            collector.sink, row, row->total_duration, &duration) ||
        !iree_hal_profile_statistics_sink_scale_duration_to_ns(
            collector.sink, row, row->minimum_duration, &min_ns) ||
        !iree_hal_profile_statistics_sink_scale_duration_to_ns(
            collector.sink, row, row->maximum_duration, &max_ns)) {
      ++collector.unscaled_rows;
      return iree_ok_status();
    }
    iree_string_view_t name;
    std::string key;
    if (iree_hal_profile_statistics_sink_find_function_name(
            collector.sink, row->executable_id, row->function_ordinal, &name)) {
      key.assign(name.data, name.size);
    } else {
      key = "executable_" + std::to_string(row->executable_id) + "_function_" +
            std::to_string(row->function_ordinal);
    }
    DispatchRow& result = (*collector.totals)[key];
    result.calls += row->sample_count;
    result.duration_ns += duration;
    result.min_ns = std::min(result.min_ns, min_ns);
    result.max_ns = std::max(result.max_ns, max_ns);
    if (iree_all_bits_set(row->flags,
                          IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TILE_TOTALS)) {
      result.tile_duration_ns += row->tile_duration_sum_ns;
    }
    return iree_ok_status();
  }

  iree_hal_device_list_t* devices_ = nullptr;
  iree_hal_profiling_session_t* session_ = nullptr;
  LockedStatisticsSink* sink_ = nullptr;
  // Function totals at the previous batch of the running pass.
  std::map<std::string, DispatchRow> batch_totals_;
  // Per-call batch means of the running pass, merged if the pass completes.
  std::map<std::string, RunningStats> pass_batch_means_;
  Workload* workload_ = nullptr;
  benchmark::State* state_ = nullptr;
  bool in_pass_ = false;
  iree_status_t status_ = iree_ok_status();
};

class BenchmarkSession {
 public:
  BenchmarkSession() { iree_tooling_module_list_initialize(&modules_); }
  ~BenchmarkSession() { iree_status_ignore(Close()); }

  iree_status_t Open() {
    if (FLAG_batch_size <= 0 || FLAG_batch_concurrency <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "batch size and concurrency must be positive");
    }
    if (FLAG_dispatch_statistics) {
      // Reject conflicts before any capture or profiling session starts.
      bool profiling_requested = false;
      IREE_RETURN_IF_ERROR(
          iree_hal_profiling_from_flags_is_requested(&profiling_requested));
      if (profiling_requested) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--dispatch_statistics cannot be combined with "
                                "device profiling or capture flags");
      }
    }
    iree_allocator_t allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(iree_tooling_create_instance(allocator, &instance_));
    IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
        instance_.get(), allocator, &modules_));
    if (!modules_.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no user module specified; use --module=file.vmfb");
    }
    IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
        instance_.get(), modules_.count, modules_.values,
        iree_string_view_empty(), allocator, &context_, &devices_, &allocator_,
        &recorder_));
    const bool selected = strlen(FLAG_function) != 0;
    iree_vm_module_t* module = iree_tooling_module_list_back(&modules_);
    if (selected) {
      iree_vm_function_t function;
      IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_make_cstring_view(FLAG_function), &function));
      IREE_RETURN_IF_ERROR(AddWorkload(function, true));
    } else {
      iree_vm_module_signature_t signature = iree_vm_module_signature(module);
      for (iree_host_size_t i = 0; i < signature.export_function_count; ++i) {
        iree_vm_function_t function;
        IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
            module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));
        IREE_RETURN_IF_ERROR(AddWorkload(function, false));
      }
    }
    if (FLAG_enable_output_processing && workloads_.size() != 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "output processing requires one function; use --function");
    }
    if (!FLAG_dispatch_statistics) {
      return iree_hal_begin_device_list_profiling_from_flags(
          devices_ ? devices_->count : 0,
          devices_ ? devices_->devices : nullptr, allocator, &profiling_);
    }
    if (recorder_) {
      // The profiling pass would add untimed executions to the capture.
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--dispatch_statistics cannot be combined with "
                              "--device_replay_output");
    }
    if (!devices_ || !devices_->count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--dispatch_statistics requires a module that uses HAL devices");
    }
    profiler_ = std::make_unique<DispatchProfiler>(devices_);
    return profiler_->Probe();
  }

  void Register() {
    for (auto& workload : workloads_) {
      Workload* target = workload.get();
      benchmark::RegisterBenchmark(
          target->name.c_str(),
          [this, target](benchmark::State& state) { Run(*target, state); })
          ->MeasureProcessCPUTime()
          ->UseRealTime()
          ->Unit(target->unit);
    }
  }

  iree_status_t ProcessOutputs(int* exit_code) {
    if (!FLAG_enable_output_processing) return iree_ok_status();
    for (auto& workload : workloads_) {
      if (!workload->outputs) continue;  // Filtered/listed without execution.
      if (device()) {
        iree_hal_buffer_params_t params = {};
        params.usage =
            IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
        params.access = IREE_HAL_MEMORY_ACCESS_ALL;
        params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
        params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
        IREE_RETURN_IF_ERROR(iree_tooling_transfer_variants(
            workload->outputs.get(), device(), allocator_.get(), params,
            nullptr, nullptr));
      }
      IREE_RETURN_IF_ERROR(iree_tooling_process_results_and_print(
          device(),
          iree_make_string_view(workload->results_cconv.data(),
                                workload->results_cconv.size()),
          workload->outputs.get(), iree_allocator_system(), exit_code));
    }
    return iree_ok_status();
  }

  iree_status_t Close() {
    iree_status_t status =
        iree_hal_profiling_session_end(std::exchange(profiling_, nullptr));
    profiler_.reset();
    workloads_.clear();
    context_.reset();
    iree_tooling_module_list_reset(&modules_);
    instance_.reset();
    if (recorder_) {
      status =
          iree_status_join(status, iree_hal_replay_recorder_close(recorder_));
      iree_hal_replay_recorder_release(recorder_);
      recorder_ = nullptr;
    }
    if (FLAG_print_statistics && devices_) {
      for (iree_host_size_t i = 0; i < devices_->count; ++i) {
        iree_hal_device_t* device = devices_->devices[i];
        iree_string_view_t device_id = iree_hal_device_id(device);
        fprintf(stderr, "Device: %.*s\n", (int)device_id.size, device_id.data);
        status = iree_status_join(
            status, iree_hal_allocator_statistics_fprint(
                        stderr, iree_hal_device_allocator(device)));
        fprintf(stderr, "\n");
      }
    }
    allocator_.reset();
    iree_hal_device_list_free(devices_);
    devices_ = nullptr;
    return status;
  }

  iree_status_t FinishMeasurements() {
    iree_status_t status = std::exchange(error_, iree_ok_status());
    if (profiler_) {
      status = iree_status_join(status, profiler_->ConsumeStatus());
    }
    return status;
  }

  // Null unless --dispatch_statistics is set.
  DispatchProfiler* profiler() const { return profiler_.get(); }
  const std::vector<std::unique_ptr<Workload>>& workloads() const {
    return workloads_;
  }

 private:
  iree_hal_device_t* device() const {
    return devices_ && devices_->count ? devices_->devices[0] : nullptr;
  }

  iree_status_t AddWorkload(iree_vm_function_t function, bool selected) {
    iree_string_view_t name = iree_vm_function_name(&function);
    iree_string_view_t type = iree_vm_function_lookup_attr_by_name(
        &function, IREE_SV("iree.benchmark"));
    iree_string_view_t model = iree_vm_function_lookup_attr_by_name(
        &function, IREE_SV("iree.abi.model"));
    InvocationKind kind = InvocationKind::kSync;
    if (iree_string_view_equal(type, IREE_SV("dispatch"))) {
      kind = InvocationKind::kDispatch;
    } else if (iree_string_view_equal(model, IREE_SV("coarse-fences"))) {
      kind = InvocationKind::kAsync;
    }
    // Without --function, run benchmark wrappers and any other public function
    // whose arguments the tool can supply without --input.
    const bool discovered = !selected && kind != InvocationKind::kDispatch &&
                            !iree_string_view_equal(type, IREE_SV("entry"));
    if (discovered &&
        (iree_string_view_starts_with(name, IREE_SV("__")) ||
         iree_string_view_find_char(name, '$', 0) != IREE_STRING_VIEW_NPOS)) {
      return iree_ok_status();  // Internal or special function.
    }
    iree_vm_function_signature_t signature =
        iree_vm_function_signature(&function);
    if (discovered) {
      iree_host_size_t argument_count = 0, result_count = 0;
      IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
          &signature, &argument_count, &result_count));
      if (argument_count != (kind == InvocationKind::kAsync ? 2u : 0u)) {
        return iree_ok_status();
      }
    }
    iree_string_view_t arguments, results;
    IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
        &signature, &arguments, &results));
    auto workload = std::make_unique<Workload>();
    workload->name = "BM_" + std::string(name.data, name.size);
    workload->function = function;
    workload->kind = kind;
    workload->results_cconv.assign(results.data, results.size);
    workload->batch_size = FLAG_batch_size;
    workload->concurrency =
        kind == InvocationKind::kAsync ? FLAG_batch_concurrency : 1;
    // Concurrency need not be a power of two. Keep the logical iteration count
    // consistent with the actual number of invocations in the rounded batch.
    int64_t batch =
        ((int64_t)workload->batch_size + workload->concurrency - 1) /
        workload->concurrency * workload->concurrency;
    if (batch > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "rounded batch size is too large");
    }
    workload->batch_size = static_cast<int32_t>(batch);
    workload->unit = FLAG_time_unit.first ? FLAG_time_unit.second
                     : kind == InvocationKind::kDispatch
                         ? benchmark::kMicrosecond
                         : benchmark::kMillisecond;
    if (kind == InvocationKind::kDispatch) {
      if (selected && FLAG_input_list().count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dispatch wrappers take --batch_size, not --input");
      }
      IREE_RETURN_IF_ERROR(
          iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                              iree_allocator_system(), &workload->inputs));
      iree_vm_value_t count = iree_vm_value_make_i32(workload->batch_size);
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_value(workload->inputs.get(), &count));
    } else if (selected) {
      IREE_RETURN_IF_ERROR(iree_tooling_parse_variants(
          arguments, FLAG_input_list(), device(), allocator_.get(),
          iree_allocator_system(), &workload->inputs));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                              iree_allocator_system(), &workload->inputs));
    }
    workloads_.push_back(std::move(workload));
    return iree_ok_status();
  }

  bool Check(iree_status_t status) {
    if (iree_status_is_ok(status)) return true;
    if (state_) {
      state_->SkipWithError(iree::Status(iree_status_clone(status)).ToString());
    }
    error_ = iree_status_join(error_, status);
    return false;
  }

  void Run(Workload& workload, benchmark::State& state) {
    state_ = &state;
    if (profiler_) profiler_->Attach(&workload, &state);
    const bool async = workload.kind == InvocationKind::kAsync;
    Batch batch;
    if (!iree_status_is_ok(error_) || (profiler_ && profiler_->failed())) {
      state.SkipWithError("an earlier benchmark or capture failed");
    } else if (!async) {
      Check(batch.Prepare(workload, device()));
    }
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, workload.name.data(),
                                        workload.name.size());
    IREE_TRACE_FRAME_MARK();
    while (state.KeepRunningBatch(workload.batch_size)) {
      if (async) {
        state.PauseTiming();
        bool prepared = Check(batch.Prepare(workload, device()));
        if (!prepared) continue;
        state.ResumeTiming();
      }
      IREE_TRACE_ZONE_BEGIN_NAMED(z1, "BenchmarkIteration");
      IREE_TRACE_FRAME_MARK_NAMED("Iteration");
      iree_status_t status = recorder_ ? iree_hal_replay_recorder_scope_begin(
                                             recorder_, IREE_SV("execute"))
                                       : iree_ok_status();
      if (iree_status_is_ok(status)) {
        status = batch.Execute(workload, context_.get());
        if (recorder_) {
          status = iree_status_join(status, iree_hal_replay_recorder_scope_end(
                                                recorder_, IREE_SV("execute")));
        }
      }
      IREE_TRACE_ZONE_END(z1);
      if (!Check(status)) continue;
      // Outputs belong to the measured pass; the profiling pass reruns it.
      const bool profiling_pass = profiler_ && profiler_->in_pass();
      iree_hal_profiling_session_t* profiling =
          profiling_pass ? profiler_->session() : profiling_;
      const bool pause = async || profiling ||
                         (workload.kind == InvocationKind::kSync && device());
      if (pause) state.PauseTiming();
      if (async) {
        if (FLAG_enable_output_processing && !profiling_pass) {
          workload.outputs = vm::retain_ref(batch.invocations.back().outputs);
        }
        batch = {};
      }
      if (pause && Check(iree_hal_profiling_session_flush(profiling))) {
        if (profiling_pass) profiler_->SampleBatch();
        state.ResumeTiming();
      }
    }
    state.SetItemsProcessed(state.iterations());
    const bool profiling_pass = profiler_ && profiler_->in_pass();
    if (!async && FLAG_enable_output_processing && !profiling_pass &&
        !state.skipped()) {
      workload.outputs = std::move(batch.invocations.front().outputs);
    }
    IREE_TRACE_ZONE_END(z0);
    if (profiler_) profiler_->Detach();
    state_ = nullptr;
  }

  vm::ref<iree_vm_instance_t> instance_;
  vm::ref<iree_vm_context_t> context_;
  vm::ref<iree_hal_allocator_t> allocator_;
  iree_tooling_module_list_t modules_;
  iree_hal_device_list_t* devices_ = nullptr;
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Session requested by the device profiling flags, if any.
  iree_hal_profiling_session_t* profiling_ = nullptr;
  std::unique_ptr<DispatchProfiler> profiler_;
  std::vector<std::unique_ptr<Workload>> workloads_;
  benchmark::State* state_ = nullptr;
  iree_status_t error_ = iree_ok_status();
};

// Reporting owns no executable callbacks. Buffer the completed reports so all
// names and counter columns are known before a console/CSV header is emitted.
// Both display and file adapters read the same immutable profiling results.
class DispatchReporter : public benchmark::BenchmarkReporter {
 public:
  DispatchReporter(benchmark::BenchmarkReporter* reporter,
                   const BenchmarkSession& session)
      : reporter_(reporter), session_(session) {}
  bool ReportContext(const Context& context) override {
    context_.emplace(context);
    return true;
  }
  void ReportRuns(const std::vector<Run>& runs) override {
    runs_.insert(runs_.end(), runs.begin(), runs.end());
  }
  void Finalize() override {
    if (!context_) return;
    std::vector<Run> reports;
    for (size_t i = 0; i < runs_.size(); ++i) {
      const auto& base = runs_[i];
      reports.push_back(base);
      if (i + 1 < runs_.size() &&
          runs_[i + 1].run_name.function_name == base.run_name.function_name) {
        continue;
      }
      for (const auto& workload : session_.workloads()) {
        if (workload->name == base.run_name.function_name) {
          Append(*workload, base, reports);
        }
      }
    }
    for (const auto& run : reports) {
      context_->name_field_width =
          std::max(context_->name_field_width, run.benchmark_name().size());
    }
    reporter_->SetOutputStream(&GetOutputStream());
    reporter_->SetErrorStream(&GetErrorStream());
    if (reporter_->ReportContext(*context_)) reporter_->ReportRuns(reports);
    reporter_->Finalize();
  }

 private:
  static void Append(const Workload& workload, const Run& base,
                     std::vector<Run>& reports) {
    const DispatchProfile& profile = workload.profile;
    if (!profile.iterations) return;
    std::vector<std::pair<std::string, DispatchRow>> rows(profile.rows.begin(),
                                                          profile.rows.end());
    std::stable_sort(rows.begin(), rows.end(),
                     [](const auto& a, const auto& b) {
                       return a.second.duration_ns > b.second.duration_ns;
                     });
    uint64_t total = 0;
    for (const auto& row : rows) total += row.second.duration_ns;
    for (const auto& entry : rows) {
      const auto& row = entry.second;
      Run run;
      run.run_name.function_name = workload.name;
      run.run_name.args = entry.first;
      run.family_index = base.family_index;
      run.per_family_instance_index = base.per_family_instance_index;
      run.repetition_index = 0;
      run.repetitions = 1;
      run.iterations = profile.iterations;
      run.time_unit = workload.unit;
      run.real_accumulated_time = row.duration_ns * 1e-9;
      run.cpu_accumulated_time = row.tile_duration_ns * 1e-9;
      run.counters["calls"] = static_cast<double>(row.calls) / run.iterations;
      run.counters["percent"] = total ? 100.0 * row.duration_ns / total : 0;
      // Per-call statistics in seconds.
      if (row.calls) {
        run.counters["mean"] = 1e-9 * row.duration_ns / row.calls;
        run.counters["min"] = 1e-9 * row.min_ns;
        run.counters["max"] = 1e-9 * row.max_ns;
      }
      if (row.batch_means.count > 1) {
        run.counters["stddev"] = 1e-9 * row.batch_means.stddev();
      }
      reports.push_back(std::move(run));
    }
  }
  std::unique_ptr<benchmark::BenchmarkReporter> reporter_;
  const BenchmarkSession& session_;
  std::optional<Context> context_;
  std::vector<Run> runs_;
};

// Google Benchmark owns file opening and flag validation, but requires custom
// file reporters to select a formatter. Read its output flags before
// Initialize consumes them, honoring its environment-variable defaults too.
static std::string BenchmarkFlag(int argc, char** argv, const char* name,
                                 const char* env, const char* default_value) {
  const char* value = getenv(env);
  std::string result = value ? value : default_value;
  std::string flag = std::string("--") + name;
  std::string prefix = flag + "=";
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) {
      result.clear();  // Bare boolean flags are true.
    } else if (std::string(argv[i]).compare(0, prefix.size(), prefix) == 0) {
      result = argv[i] + prefix.size();
    }
  }
  return result;
}

// Match Google Benchmark's boolean flag/environment conventions.
static bool BenchmarkBool(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value.size() == 1) {
    return std::isalnum(static_cast<unsigned char>(value[0])) && value != "0" &&
           value != "f" && value != "n";
  }
  return value != "false" && value != "no" && value != "off";
}

static benchmark::BenchmarkReporter* CreateFileReporter(
    const std::string& format, bool tabular) {
  if (format == "json") return new benchmark::JSONReporter;
  BENCHMARK_DISABLE_DEPRECATED_WARNING
  if (format == "csv") return new benchmark::CSVReporter;
  BENCHMARK_RESTORE_DEPRECATED_WARNING
  return new benchmark::ConsoleReporter(
      tabular ? benchmark::ConsoleReporter::OO_Tabular
              : benchmark::ConsoleReporter::OO_None);
}

static int RunBenchmarkModule(int argc, char** argv) {
  iree_flags_set_usage("iree-benchmark-module", kIreeBenchmarkModuleUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  if (FLAG_agents_md) {
    PrintBenchmarkModuleAgentMarkdown(stdout);
    return EXIT_SUCCESS;
  }
  std::string output =
      BenchmarkFlag(argc, argv, "benchmark_out", "BENCHMARK_OUT", "");
  std::string format = BenchmarkFlag(argc, argv, "benchmark_out_format",
                                     "BENCHMARK_OUT_FORMAT", "json");
  bool tabular =
      BenchmarkBool(BenchmarkFlag(argc, argv, "benchmark_counters_tabular",
                                  "BENCHMARK_COUNTERS_TABULAR", "false"));
  ::benchmark::Initialize(&argc, argv);
  BenchmarkSession session;
  iree_status_t status = session.Open();
  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    session.Register();
    std::unique_ptr<DispatchReporter> display, file;
    if (DispatchProfiler* profiler = session.profiler()) {
      ::benchmark::RegisterProfilerManager(profiler);
      display = std::make_unique<DispatchReporter>(
          ::benchmark::CreateDefaultDisplayReporter(), session);
      if (!output.empty()) {
        file = std::make_unique<DispatchReporter>(
            CreateFileReporter(format, tabular), session);
      }
    }
    ::benchmark::RunSpecifiedBenchmarks(display.get(), file.get());
    ::benchmark::RegisterProfilerManager(nullptr);
    ::benchmark::ClearRegisteredBenchmarks();
    status = session.FinishMeasurements();
    if (iree_status_is_ok(status)) status = session.ProcessOutputs(&exit_code);
    for (const auto& workload : session.workloads()) {
      const DispatchProfile& profile = workload->profile;
      if (!profile.iterations) continue;
      if (profile.rows.empty() || profile.dropped_records ||
          profile.unscaled_rows) {
        fprintf(stderr,
                "--dispatch_statistics: %s: %zu functions, %" PRIu64
                " dropped records, %" PRIu64
                " unscaled rows; timings may be incomplete\n",
                workload->name.c_str(), profile.rows.size(),
                profile.dropped_records, profile.unscaled_rows);
      }
    }
  }
  status = iree_status_join(status, session.Close());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return exit_code;
}

}  // namespace
}  // namespace iree

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree-benchmark-module");
  int exit_code = iree::RunBenchmarkModule(argc, argv);
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
