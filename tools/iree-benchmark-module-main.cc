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
#include <cinttypes>
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

// Google Benchmark creates its own file reporter only when the caller passes
// none, so wrapping it requires the output flags that Initialize parsed. They
// are defined by the library but not declared in its public header.
namespace benchmark {
BENCHMARK_EXPORT extern std::string FLAGS_benchmark_out;
BENCHMARK_EXPORT extern std::string FLAGS_benchmark_out_format;
BENCHMARK_EXPORT extern bool FLAGS_benchmark_counters_tabular;
}  // namespace benchmark

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
    "average all profiled repetitions. Requires a backend whose lightweight\n"
    "profiling attributes dispatches in existing command buffers; elsewhere\n"
    "the session fails to start or the rows are missing with a warning.\n"
    "Cannot combine with --device_profiling_mode, --print_device_statistics,\n"
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

// How a benchmarked function is called, which decides what one measured batch
// of --batch_size logical iterations executes.
enum class InvocationKind {
  // A synchronous function. One VM invocation per batch; any repetition is
  // compiled into the function (--iree-hal-benchmark-dispatch-repeat-count).
  kSync,
  // A function with the coarse-fences ABI. A batch is --batch_size VM
  // invocations spread over --batch_concurrency fence timelines, followed by
  // one wait for all of them.
  kAsync,
  // A generated dispatch benchmark (iree.benchmark = "dispatch"). One VM
  // invocation per batch that takes the batch size as its repeat count.
  kDispatch,
};

// Profiled time of one dispatch function, summed over all profiled iterations.
struct DispatchRow {
  // Number of dispatches of the function.
  uint64_t calls = 0;
  // Total device time of those dispatches.
  uint64_t duration_ns = 0;
  // Worker time summed over all tiles of those dispatches, reported as CPU
  // time. Zero on backends that do not report tile totals.
  uint64_t tile_duration_ns = 0;
};

// Per-dispatch results accumulated over all profiled repetitions.
struct DispatchProfile {
  // Rows keyed by dispatch function name, or by executable id and function
  // ordinal when the backend provides no name.
  std::map<std::string, DispatchRow> rows;
  // Logical iterations of all completed profiling passes. Reported rows are
  // per iteration, so the totals are divided by this.
  int64_t iterations = 0;
  // Records the statistics sink dropped; nonzero makes the rows incomplete.
  uint64_t dropped_records = 0;
  // Dispatch rows skipped because they had no timing or their timestamps
  // could not be scaled to nanoseconds.
  uint64_t unscaled_rows = 0;
};

// A benchmarked function with the ABI and iteration accounting resolved during
// discovery. Measurement and profiling run the same workload.
struct Workload {
  // Benchmark name, "BM_" followed by the function name.
  std::string name;
  // The exported VM function to invoke.
  iree_vm_function_t function = {};
  // Calling convention of the results, used to print or check outputs.
  std::string results_cconv;
  InvocationKind kind = InvocationKind::kSync;
  // Logical iterations per measured batch: --batch_size rounded up to a
  // multiple of |concurrency|.
  int32_t batch_size = 1;
  // Number of fence timelines that async invocations run on; 1 otherwise.
  int32_t concurrency = 1;
  // Time unit of the reported rows.
  benchmark::TimeUnit unit = benchmark::kMillisecond;
  // Arguments shared by every invocation. Async invocations clone them and
  // append their wait and signal fences; dispatch benchmarks take the repeat
  // count.
  vm::ref<iree_vm_list_t> inputs;
  // Outputs of the last measured invocation, retained only with
  // --enable_output_processing. NULL if the function did not run.
  vm::ref<iree_vm_list_t> outputs;
  // Dispatch rows collected by --dispatch_statistics.
  DispatchProfile profile;
};

// Arguments and results of one VM call within a batch.
struct Invocation {
  vm::ref<iree_vm_list_t> inputs;
  vm::ref<iree_vm_list_t> outputs;
};

// Owns one batch's arguments, outputs and synchronization. Only the async ABI
// requires host-side batching; dispatch wrappers repeat inside the VM call.
struct Batch {
  // One entry per VM call in the batch: |batch_size| for async workloads and
  // one otherwise.
  std::vector<Invocation> invocations;
  // Signaled when the last invocation on every timeline has completed. NULL
  // unless the workload is async.
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
  bool InPass() const { return in_pass_; }
  // The session of the running profiling pass, or NULL.
  iree_hal_profiling_session_t* Session() const { return session_; }
  bool Failed() const { return !iree_status_is_ok(status_); }
  // Returns and clears the failures of all profiling passes.
  iree_status_t ConsumeStatus() {
    return std::exchange(status_, iree_ok_status());
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
    IREE_RETURN_IF_ERROR(iree_hal_profile_statistics_sink_create(
        iree_allocator_system(), &sink_));
    iree_hal_device_profiling_options_t options = {};
    options.flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
    options.sink = iree_hal_profile_statistics_sink_base(sink_);
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
      workload->profile.dropped_records +=
          iree_hal_profile_statistics_sink_dropped_record_count(sink_);
      Collector collector = {sink_, &workload->profile};
      iree_hal_profile_statistics_row_callback_t callback = {CollectRow,
                                                             &collector};
      status = iree_hal_profile_statistics_sink_for_each_row(sink_, callback);
    }
    iree_hal_profile_statistics_sink_release(std::exchange(sink_, nullptr));
    return status;
  }

  // State passed to CollectRow while iterating the rows of a sink.
  struct Collector {
    // Sink being iterated, which scales timestamps and resolves names.
    iree_hal_profile_statistics_sink_t* sink;
    // Profile that receives the rows.
    DispatchProfile* profile;
  };

  static iree_status_t CollectRow(
      void* data, const iree_hal_profile_statistics_row_t* row) {
    Collector& collector = *static_cast<Collector*>(data);
    if (row->row_type !=
            IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_FUNCTION &&
        row->row_type !=
            IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_FUNCTION) {
      return iree_ok_status();
    }
    uint64_t duration = 0;
    if (!iree_all_bits_set(row->flags,
                           IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING) ||
        !iree_hal_profile_statistics_sink_scale_duration_to_ns(
            collector.sink, row, row->total_duration, &duration)) {
      ++collector.profile->unscaled_rows;
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
    DispatchRow& result = collector.profile->rows[key];
    result.calls += row->sample_count;
    result.duration_ns += duration;
    if (iree_all_bits_set(row->flags,
                          IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TILE_TOTALS)) {
      result.tile_duration_ns += row->tile_duration_sum_ns;
    }
    return iree_ok_status();
  }

  // Devices profiled in every pass. Unowned.
  iree_hal_device_list_t* devices_ = nullptr;
  // Session of the running pass, or NULL.
  iree_hal_profiling_session_t* session_ = nullptr;
  // Collects the records of |session_|; replaced for every pass.
  iree_hal_profile_statistics_sink_t* sink_ = nullptr;
  // Workload and state of the running benchmark function, set by Attach.
  Workload* workload_ = nullptr;
  benchmark::State* state_ = nullptr;
  bool in_pass_ = false;
  // Failures of all passes since the last ConsumeStatus.
  iree_status_t status_ = iree_ok_status();
};

// Owns the VM and HAL state of one tool run: loads the modules, discovers the
// workloads, runs them for Google Benchmark and processes their outputs.
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
      // The profiling pass would add untimed executions to the capture.
      if (iree_tooling_device_replay_capture_requested()) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--dispatch_statistics cannot be combined with "
                                "--device_replay_output");
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
      if (Device()) {
        iree_hal_buffer_params_t params = {};
        params.usage =
            IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
        params.access = IREE_HAL_MEMORY_ACCESS_ALL;
        params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
        params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
        IREE_RETURN_IF_ERROR(iree_tooling_transfer_variants(
            workload->outputs.get(), Device(), allocator_.get(), params,
            nullptr, nullptr));
      }
      IREE_RETURN_IF_ERROR(iree_tooling_process_results_and_print(
          Device(),
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

  // Ends the session of the device profiling flags so that output processing
  // is not profiled, and returns the failures of all benchmark runs.
  iree_status_t FinishMeasurements() {
    iree_status_t status = std::exchange(error_, iree_ok_status());
    if (profiler_) {
      status = iree_status_join(status, profiler_->ConsumeStatus());
    }
    return iree_status_join(status, iree_hal_profiling_session_end(
                                        std::exchange(profiling_, nullptr)));
  }

  // Null unless --dispatch_statistics is set.
  DispatchProfiler* Profiler() const { return profiler_.get(); }
  const std::vector<std::unique_ptr<Workload>>& Workloads() const {
    return workloads_;
  }

 private:
  iree_hal_device_t* Device() const {
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
          arguments, FLAG_input_list(), Device(), allocator_.get(),
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

  // Preserve values even when outputs alias inputs or mutable module storage.
  // Run after measurement stops and before the profiling pass can overwrite
  // them. Host-accessible outputs still need a copy; a transfer that only
  // ensures host accessibility may leave them aliasing the original storage.
  iree_status_t SnapshotOutputs(Workload& workload) {
    if (!workload.outputs) return iree_ok_status();
    return SnapshotList(workload.outputs.get());
  }

  // Replaces the buffers, buffer views and lists in |list| with copies. Nested
  // lists may be retained and mutated by the module, so they are cloned before
  // their elements are replaced; only |list| itself must be owned by the tool.
  iree_status_t SnapshotList(iree_vm_list_t* list) {
    iree_hal_buffer_params_t params = {};
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
      iree_vm_variant_t value = iree_vm_variant_empty();
      IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_assign(list, i, &value));
      if (!iree_vm_variant_is_ref(value) || !value.ref.ptr) continue;
      if (iree_vm_list_isa(value.ref)) {
        vm::ref<iree_vm_list_t> copy;
        IREE_RETURN_IF_ERROR(iree_vm_list_clone(
            iree_vm_list_deref(value.ref), iree_allocator_system(), &copy));
        IREE_RETURN_IF_ERROR(SnapshotList(copy.get()));
        IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_retain(list, i, copy));
        continue;
      }
      iree_hal_buffer_view_t* source_view = nullptr;
      iree_hal_buffer_t* source = nullptr;
      if (iree_hal_buffer_view_isa(value.ref)) {
        source_view = iree_hal_buffer_view_deref(value.ref);
        source = iree_hal_buffer_view_buffer(source_view);
      } else if (iree_hal_buffer_isa(value.ref)) {
        source = iree_hal_buffer_deref(value.ref);
      } else {
        continue;
      }
      const iree_device_size_t length = iree_hal_buffer_byte_length(source);
      vm::ref<iree_hal_buffer_t> copy;
      IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
          allocator_.get(), params, length, &copy));
      IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2d(
          Device(), source, 0, copy.get(), 0, length,
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
      if (source_view) {
        vm::ref<iree_hal_buffer_view_t> copy_view;
        IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create_like(
            copy.get(), source_view, iree_allocator_system(), &copy_view));
        IREE_RETURN_IF_ERROR(
            iree_vm_list_set_buffer_view_retain(list, i, copy_view.get()));
      } else {
        IREE_RETURN_IF_ERROR(
            iree_vm_list_set_buffer_retain(list, i, copy.get()));
      }
    }
    return iree_ok_status();
  }

  void Run(Workload& workload, benchmark::State& state) {
    state_ = &state;
    if (profiler_) profiler_->Attach(&workload, &state);
    const bool async = workload.kind == InvocationKind::kAsync;
    Batch batch;
    if (!iree_status_is_ok(error_) || (profiler_ && profiler_->Failed())) {
      state.SkipWithError("an earlier benchmark or capture failed");
    } else if (!async) {
      Check(batch.Prepare(workload, Device()));
    }
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, workload.name.data(),
                                        workload.name.size());
    IREE_TRACE_FRAME_MARK();
    while (state.KeepRunningBatch(workload.batch_size)) {
      if (async) {
        state.PauseTiming();
        bool prepared = Check(batch.Prepare(workload, Device()));
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
      const bool profiling_pass = profiler_ && profiler_->InPass();
      iree_hal_profiling_session_t* profiling =
          profiling_pass ? profiler_->Session() : profiling_;
      const bool pause = async || profiling ||
                         (workload.kind == InvocationKind::kSync && Device());
      if (pause) state.PauseTiming();
      if (async) {
        if (FLAG_enable_output_processing && !profiling_pass) {
          workload.outputs = vm::retain_ref(batch.invocations.back().outputs);
        }
        batch = {};
      }
      if (pause && Check(iree_hal_profiling_session_flush(profiling))) {
        state.ResumeTiming();
      }
    }
    state.SetItemsProcessed(state.iterations());
    const bool profiling_pass = profiler_ && profiler_->InPass();
    if (!async && FLAG_enable_output_processing && !profiling_pass &&
        !state.skipped()) {
      workload.outputs = std::move(batch.invocations.front().outputs);
    }
    if (profiler_ && FLAG_enable_output_processing && !profiling_pass &&
        !state.skipped()) {
      Check(SnapshotOutputs(workload));
    }
    IREE_TRACE_ZONE_END(z0);
    if (profiler_) profiler_->Detach();
    state_ = nullptr;
  }

  vm::ref<iree_vm_instance_t> instance_;
  vm::ref<iree_vm_context_t> context_;
  // Allocator of the first device, used for inputs and output transfers.
  vm::ref<iree_hal_allocator_t> allocator_;
  // Modules loaded from --module; the last one is benchmarked.
  iree_tooling_module_list_t modules_;
  // Devices the context uses; NULL or empty if the program uses none.
  iree_hal_device_list_t* devices_ = nullptr;
  // Recorder of --device_replay_output, or NULL.
  iree_hal_replay_recorder_t* recorder_ = nullptr;
  // Session requested by the device profiling flags, if any.
  iree_hal_profiling_session_t* profiling_ = nullptr;
  // Profiler of --dispatch_statistics, or NULL.
  std::unique_ptr<DispatchProfiler> profiler_;
  // Functions to benchmark, in registration order.
  std::vector<std::unique_ptr<Workload>> workloads_;
  // State of the running benchmark function, or NULL between functions.
  benchmark::State* state_ = nullptr;
  // Failures of the measured runs, reported after all benchmarks finish.
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
      for (const auto& workload : session_.Workloads()) {
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
      reports.push_back(std::move(run));
    }
  }
  // Formatter that writes the merged reports.
  std::unique_ptr<benchmark::BenchmarkReporter> reporter_;
  // Provides the workloads and their dispatch profiles.
  const BenchmarkSession& session_;
  // Context and measured runs buffered until Finalize.
  std::optional<Context> context_;
  std::vector<Run> runs_;
};

// Creates the reporter Google Benchmark would use for --benchmark_out, whose
// format Initialize has validated.
static benchmark::BenchmarkReporter* CreateFileReporter() {
  const std::string& format = ::benchmark::FLAGS_benchmark_out_format;
  if (format == "json") return new benchmark::JSONReporter;
  BENCHMARK_DISABLE_DEPRECATED_WARNING
  if (format == "csv") return new benchmark::CSVReporter;
  BENCHMARK_RESTORE_DEPRECATED_WARNING
  return new benchmark::ConsoleReporter(
      ::benchmark::FLAGS_benchmark_counters_tabular
          ? benchmark::ConsoleReporter::OO_Tabular
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
  ::benchmark::Initialize(&argc, argv);
  BenchmarkSession session;
  iree_status_t status = session.Open();
  int exit_code = EXIT_SUCCESS;
  if (iree_status_is_ok(status)) {
    session.Register();
    std::unique_ptr<DispatchReporter> display, file;
    if (DispatchProfiler* profiler = session.Profiler()) {
      ::benchmark::RegisterProfilerManager(profiler);
      display = std::make_unique<DispatchReporter>(
          ::benchmark::CreateDefaultDisplayReporter(), session);
      if (!::benchmark::FLAGS_benchmark_out.empty()) {
        file =
            std::make_unique<DispatchReporter>(CreateFileReporter(), session);
      }
    }
    ::benchmark::RunSpecifiedBenchmarks(display.get(), file.get());
    ::benchmark::RegisterProfilerManager(nullptr);
    ::benchmark::ClearRegisteredBenchmarks();
    status = session.FinishMeasurements();
    if (iree_status_is_ok(status)) status = session.ProcessOutputs(&exit_code);
    for (const auto& workload : session.Workloads()) {
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
