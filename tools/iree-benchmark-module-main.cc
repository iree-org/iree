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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/recorder.h"
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

// A benchmarked function with the ABI and iteration accounting resolved during
// discovery.
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
    return iree_hal_begin_profiling_from_flags(Device(), allocator,
                                               &profiling_);
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
        iree_hal_end_profiling_from_flags(std::exchange(profiling_, nullptr));
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
    return iree_status_join(status, iree_hal_end_profiling_from_flags(
                                        std::exchange(profiling_, nullptr)));
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

  void Run(Workload& workload, benchmark::State& state) {
    state_ = &state;
    const bool async = workload.kind == InvocationKind::kAsync;
    Batch batch;
    if (!iree_status_is_ok(error_)) {
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
      const bool pause = async || profiling_ ||
                         (workload.kind == InvocationKind::kSync && Device());
      if (pause) state.PauseTiming();
      if (async) {
        if (FLAG_enable_output_processing) {
          workload.outputs = vm::retain_ref(batch.invocations.back().outputs);
        }
        batch = {};
      }
      if (pause && Check(iree_hal_flush_profiling_from_flags(profiling_))) {
        state.ResumeTiming();
      }
    }
    state.SetItemsProcessed(state.iterations());
    if (!async && FLAG_enable_output_processing && !state.skipped()) {
      workload.outputs = std::move(batch.invocations.front().outputs);
    }
    IREE_TRACE_ZONE_END(z0);
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
  iree_hal_profiling_from_flags_t* profiling_ = nullptr;
  // Functions to benchmark, in registration order.
  std::vector<std::unique_ptr<Workload>> workloads_;
  // State of the running benchmark function, or NULL between functions.
  benchmark::State* state_ = nullptr;
  // Failures of the measured runs, reported after all benchmarks finish.
  iree_status_t error_ = iree_ok_status();
};

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
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::ClearRegisteredBenchmarks();
    status = session.FinishMeasurements();
    if (iree_status_is_ok(status)) status = session.ProcessOutputs(&exit_code);
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
