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

#include <array>
#include <cstdio>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/vm_util.h"
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
    "  2x2xi32=1 2 3 4\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  2x2xi32=[[1 2][3 4]]\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  2x2xi32=@some/file.bin\n"
    "numpy npy files (from numpy.save) can be read to provide 1+ values:\n"
    "  @some.npy\n"
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

namespace iree {
namespace {

static void BenchmarkGenericFunction(const std::string& benchmark_name,
                                     int32_t batch_size,
                                     iree_vm_context_t* context,
                                     iree_vm_function_t function,
                                     iree_vm_list_t* inputs,
                                     benchmark::State& state) {
  IREE_TRACE_SCOPE_DYNAMIC(benchmark_name.c_str());
  IREE_TRACE_FRAME_MARK();

  vm::ref<iree_vm_list_t> outputs;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                    iree_allocator_system(), &outputs));

  // Benchmarking loop.
  while (state.KeepRunningBatch(batch_size)) {
    IREE_TRACE_SCOPE0("BenchmarkIteration");
    IREE_TRACE_FRAME_MARK_NAMED("Iteration");
    IREE_CHECK_OK(iree_vm_invoke(
        context, function, IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/nullptr,
        inputs, outputs.get(), iree_allocator_system()));
    IREE_CHECK_OK(iree_vm_list_resize(outputs.get(), 0));
  }
  state.SetItemsProcessed(state.iterations());
}

void RegisterGenericBenchmark(const std::string& function_name,
                              iree_vm_context_t* context,
                              iree_vm_function_t function,
                              iree_vm_list_t* inputs) {
  auto benchmark_name = "BM_" + function_name;
  int32_t batch_size = FLAG_batch_size;
  benchmark::RegisterBenchmark(benchmark_name.c_str(),
                               [=](benchmark::State& state) -> void {
                                 BenchmarkGenericFunction(
                                     benchmark_name, batch_size, context,
                                     function, inputs, state);
                               })
      // By default only the main thread is included in CPU time. Include all
      // the threads instead.
      ->MeasureProcessCPUTime()
      // To make single and multi-threaded benchmarks more comparable, use the
      // wall time to determine how many iterations to run. See
      // https://github.com/google/benchmark#cpu-timers,
      ->UseRealTime()
      ->Unit(FLAG_time_unit.first ? FLAG_time_unit.second
                                  : benchmark::kMillisecond);
}

// Runs up to |batch_size| pipelined invocations in sequence along with
// concurrency. Example:
//   batch_size=1, concurrency=1:
//     [invocation 0]
//   batch_size=2, concurrency=1:
//     [invocation 0] -> [invocation 1]
//   batch_size=2, concurrency=2:
//     [invocation 0]
//     [invocation 1]
//   batch_size=4, concurrency=2:
//     [invocation 0] -> [invocation 2]
//     [invocation 1] -> [invocation 3]
static void BenchmarkAsyncFunction(
    const std::string& benchmark_name, int32_t batch_size,
    int32_t batch_concurrency, iree_hal_device_t* device,
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_vm_list_t* common_inputs, benchmark::State& state) {
  IREE_TRACE_SCOPE_DYNAMIC(benchmark_name.c_str());
  IREE_TRACE_FRAME_MARK();
  iree_allocator_t host_allocator = iree_allocator_system();

  // Round up batch size to some multiple of concurrency.
  batch_size = (int32_t)iree_host_align(batch_size, batch_concurrency);

  // Benchmarking loop.
  while (state.KeepRunningBatch(batch_size)) {
    IREE_TRACE_SCOPE0("BenchmarkIteration");
    IREE_TRACE_FRAME_MARK_NAMED("Iteration");

    state.PauseTiming();

    IREE_TRACE_ZONE_BEGIN_NAMED(z_begin, "PrepareBatch");

    // Each concurrent track of execution gets its own semaphore.
    std::vector<vm::ref<iree_hal_semaphore_t>> timeline_semaphores;
    for (int32_t i = 0; i < batch_concurrency; ++i) {
      vm::ref<iree_hal_semaphore_t> timeline_semaphore;
      IREE_CHECK_OK(
          iree_hal_semaphore_create(device, 0ull, &timeline_semaphore));
      timeline_semaphores.push_back(std::move(timeline_semaphore));
    }

    // Preallocate fences and I/O for each invocation.
    // The same inputs are used for each but we need a unique list to hold the
    // unique fences. Each fence represents when the invocation has completed.
    std::vector<vm::ref<iree_hal_fence_t>> invocation_fences;
    std::vector<vm::ref<iree_vm_list_t>> invocation_inputs;
    std::vector<vm::ref<iree_vm_list_t>> invocation_outputs;
    vm::ref<iree_hal_fence_t> completion_fence;
    IREE_CHECK_OK(iree_hal_fence_create(batch_concurrency, host_allocator,
                                        &completion_fence));
    for (int32_t i = 0; i < batch_size / batch_concurrency; ++i) {
      for (int32_t j = 0; j < batch_concurrency; ++j) {
        // Chain each concurrent minibatch to the previous. Note that to start
        // we wait on nothing and begin executing immediately.
        vm::ref<iree_hal_fence_t> wait_fence;
        if (i > 0) {
          wait_fence = vm::retain_ref(
              invocation_fences[(i - 1) * batch_concurrency + j]);
        }
        uint64_t signal_value = i + 1;
        vm::ref<iree_hal_fence_t> signal_fence;
        IREE_CHECK_OK(iree_hal_fence_create_at(timeline_semaphores[j].get(),
                                               signal_value, host_allocator,
                                               &signal_fence));
        invocation_fences.push_back(vm::retain_ref(signal_fence));

        // Join the final minibatch on the completion fence.
        if (i == batch_size / batch_concurrency - 1) {
          IREE_CHECK_OK(iree_hal_fence_insert(completion_fence.get(),
                                              timeline_semaphores[j].get(),
                                              signal_value));
        }

        // Clone common inputs and add the invocation-specific fences.
        vm::ref<iree_vm_list_t> inputs;
        IREE_CHECK_OK(
            iree_vm_list_clone(common_inputs, host_allocator, &inputs));
        IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs.get(), wait_fence));
        IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs.get(), signal_fence));
        invocation_inputs.push_back(std::move(inputs));

        // Setup empty outputs.
        vm::ref<iree_vm_list_t> outputs;
        IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                          host_allocator, &outputs));
        invocation_outputs.push_back(std::move(outputs));
      }
    }

    IREE_TRACE_ZONE_END(z_begin);

    state.ResumeTiming();
    {
      // TODO(benvanik): replace with async invocations. Today if the invocation
      // performs any waits this will block on the initial invoke instead of
      // actually overlapping things.
      for (int32_t i = 0; i < batch_size; ++i) {
        IREE_CHECK_OK(
            iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                           /*policy=*/nullptr, invocation_inputs[i].get(),
                           invocation_outputs[i].get(), host_allocator));
      }
      IREE_CHECK_OK(
          iree_hal_fence_wait(completion_fence.get(), iree_infinite_timeout()));
    }
    state.PauseTiming();

    IREE_TRACE_ZONE_BEGIN_NAMED(z_end, "CleanupBatch");
    for (int32_t i = 0; i < batch_size; ++i) {
      iree_vm_list_clear(invocation_outputs[i].get());
    }
    invocation_fences.clear();
    invocation_inputs.clear();
    invocation_outputs.clear();
    completion_fence.reset();
    timeline_semaphores.clear();
    IREE_TRACE_ZONE_END(z_end);

    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
}

void RegisterAsyncBenchmark(const std::string& function_name,
                            iree_hal_device_t* device,
                            iree_vm_context_t* context,
                            iree_vm_function_t function,
                            iree_vm_list_t* inputs) {
  auto benchmark_name = "BM_" + function_name;
  int32_t batch_size = FLAG_batch_size;
  int32_t batch_concurrency = FLAG_batch_concurrency;
  benchmark::RegisterBenchmark(
      benchmark_name.c_str(),
      [=](benchmark::State& state) -> void {
        BenchmarkAsyncFunction(benchmark_name, batch_size, batch_concurrency,
                               device, context, function, inputs, state);
      })
      // By default only the main thread is included in CPU time. Include all
      // the threads instead.
      ->MeasureProcessCPUTime()
      // To make single and multi-threaded benchmarks more comparable, use the
      // wall time to determine how many iterations to run. See
      // https://github.com/google/benchmark#cpu-timers,
      ->UseRealTime()
      ->Unit(FLAG_time_unit.first ? FLAG_time_unit.second
                                  : benchmark::kMillisecond);
}

static void BenchmarkDispatchFunction(const std::string& benchmark_name,
                                      iree_vm_context_t* context,
                                      iree_vm_function_t function,
                                      benchmark::State& state) {
  IREE_TRACE_SCOPE_DYNAMIC(benchmark_name.c_str());
  IREE_TRACE_FRAME_MARK();

  vm::ref<iree_vm_list_t> inputs;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                    iree_allocator_system(), &inputs));
  iree_vm_value_t batch_size = iree_vm_value_make_i32(FLAG_batch_size);
  IREE_CHECK_OK(iree_vm_list_push_value(inputs.get(), &batch_size));

  vm::ref<iree_vm_list_t> outputs;
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                    iree_allocator_system(), &outputs));

  // Benchmarking loop.
  while (state.KeepRunningBatch(FLAG_batch_size)) {
    IREE_TRACE_SCOPE0("BenchmarkIteration");
    IREE_TRACE_FRAME_MARK_NAMED("Iteration");
    IREE_CHECK_OK(iree_vm_invoke(
        context, function, IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/nullptr,
        inputs.get(), outputs.get(), iree_allocator_system()));
    IREE_CHECK_OK(iree_vm_list_resize(outputs.get(), 0));
  }
  state.SetItemsProcessed(state.iterations());
}

void RegisterDispatchBenchmark(const std::string& function_name,
                               iree_vm_context_t* context,
                               iree_vm_function_t function) {
  auto benchmark_name = "BM_" + function_name;
  benchmark::RegisterBenchmark(
      benchmark_name.c_str(),
      [benchmark_name, context, function](benchmark::State& state) -> void {
        BenchmarkDispatchFunction(benchmark_name, context, function, state);
      })
      // By default only the main thread is included in CPU time. Include all
      // the threads instead.
      ->MeasureProcessCPUTime()
      // To make single and multi-threaded benchmarks more comparable, use the
      // wall time to determine how many iterations to run. See
      // https://github.com/google/benchmark#cpu-timers,
      ->UseRealTime()
      ->Unit(FLAG_time_unit.first ? FLAG_time_unit.second
                                  : benchmark::kMicrosecond);
}

// The lifetime of IREEBenchmark should be as long as
// ::benchmark::RunSpecifiedBenchmarks() where the resources are used during
// benchmarking.
class IREEBenchmark {
 public:
  IREEBenchmark() = default;

  ~IREEBenchmark() {
    IREE_TRACE_SCOPE0("IREEBenchmark::dtor");

    // Order matters. Tear down modules first to release resources.
    inputs_.reset();
    context_.reset();
    main_module_.reset();
    instance_.reset();

    // Tear down device last in order to get accurate statistics.
    if (device_allocator_ && FLAG_print_statistics) {
      IREE_IGNORE_ERROR(iree_hal_allocator_statistics_fprint(
          stderr, device_allocator_.get()));
    }
    device_allocator_.reset();
    device_.reset();
  };

  iree_hal_device_t* device() const { return device_.get(); }

  iree_status_t Register() {
    IREE_TRACE_SCOPE0("IREEBenchmark::Register");

    if (!instance_ || !device_allocator_ || !context_ || !main_module_) {
      IREE_RETURN_IF_ERROR(Init());
    }

    auto function_name = std::string(FLAG_function);
    if (!function_name.empty()) {
      IREE_RETURN_IF_ERROR(RegisterSpecificFunction(function_name));
    } else {
      IREE_RETURN_IF_ERROR(RegisterAllExportedFunctions());
    }
    return iree_ok_status();
  }

 private:
  iree_status_t Init() {
    IREE_TRACE_SCOPE0("IREEBenchmark::Init");
    IREE_TRACE_FRAME_MARK_BEGIN_NAMED("init");

    iree_allocator_t host_allocator = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        iree_tooling_create_instance(host_allocator, &instance_));

    IREE_RETURN_IF_ERROR(iree_tooling_load_module_from_flags(
        instance_.get(), host_allocator, &main_module_));

    IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
        instance_.get(), /*user_module_count=*/1,
        /*user_modules=*/&main_module_,
        /*default_device_uri=*/iree_string_view_empty(), host_allocator,
        &context_, &device_, &device_allocator_));

    IREE_TRACE_FRAME_MARK_END_NAMED("init");
    return iree_ok_status();
  }

  iree_status_t RegisterSpecificFunction(const std::string& function_name) {
    IREE_TRACE_SCOPE0("IREEBenchmark::RegisterSpecificFunction");

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        main_module_.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{function_name.data(), function_name.size()},
        &function));

    IREE_CHECK_OK(iree_tooling_parse_to_variant_list(
        device_allocator_.get(), FLAG_input_list().values,
        FLAG_input_list().count, iree_vm_instance_allocator(instance_.get()),
        &inputs_));

    iree_string_view_t invocation_model = iree_vm_function_lookup_attr_by_name(
        &function, IREE_SV("iree.abi.model"));
    if (iree_string_view_equal(invocation_model, IREE_SV("coarse-fences"))) {
      // Asynchronous invocation.
      iree::RegisterAsyncBenchmark(function_name, device_.get(), context_.get(),
                                   function, inputs_.get());
    } else {
      // Synchronous invocation.
      iree::RegisterGenericBenchmark(function_name, context_.get(), function,
                                     inputs_.get());
    }
    return iree_ok_status();
  }

  iree_status_t RegisterAllExportedFunctions() {
    IREE_TRACE_SCOPE0("IREEBenchmark::RegisterAllExportedFunctions");
    iree_vm_module_signature_t signature =
        iree_vm_module_signature(main_module_.get());
    for (iree_host_size_t i = 0; i < signature.export_function_count; ++i) {
      iree_vm_function_t function;
      IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
          main_module_.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));
      iree_string_view_t function_name = iree_vm_function_name(&function);

      // We run anything with the 'benchmark' attribute.
      // If the attribute is not present we'll run anything that looks runnable.
      iree_string_view_t benchmark_type = iree_vm_function_lookup_attr_by_name(
          &function, IREE_SV("iree.benchmark"));
      if (iree_string_view_equal(benchmark_type, IREE_SV("dispatch"))) {
        iree::RegisterDispatchBenchmark(
            std::string(function_name.data, function_name.size), context_.get(),
            function);
      } else if (iree_string_view_equal(benchmark_type, IREE_SV("entry"))) {
        iree::RegisterGenericBenchmark(
            std::string(function_name.data, function_name.size), context_.get(),
            function,
            /*inputs=*/nullptr);
      } else {
        // Pick up generic () -> () functions.
        if (iree_string_view_starts_with(function_name,
                                         iree_make_cstring_view("__")) ||
            iree_string_view_find_char(function_name, '$', 0) !=
                IREE_STRING_VIEW_NPOS) {
          // Skip internal or special functions.
          continue;
        }

        // Query function information to determine how to run it.
        iree_vm_function_signature_t signature =
            iree_vm_function_signature(&function);
        iree_host_size_t argument_count = 0;
        iree_host_size_t result_count = 0;
        IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
            &signature, &argument_count, &result_count));
        iree_string_view_t invocation_model =
            iree_vm_function_lookup_attr_by_name(&function,
                                                 IREE_SV("iree.abi.model"));
        if (iree_string_view_equal(invocation_model,
                                   IREE_SV("coarse-fences"))) {
          // Asynchronous invocation with coarse fences. Expect just those.
          if (argument_count == 2) {
            // Only functions taking a (wait, signal) fence pair are run.
            iree::RegisterAsyncBenchmark(
                std::string(function_name.data, function_name.size),
                device_.get(), context_.get(), function,
                /*inputs=*/nullptr);
          }
        } else {
          // Basic synchronous invocation.
          if (argument_count == 0) {
            // Only functions with no inputs are run (because we can't pass
            // anything).
            iree::RegisterGenericBenchmark(
                std::string(function_name.data, function_name.size),
                context_.get(), function,
                /*inputs=*/nullptr);
          }
        }
      }
    }
    return iree_ok_status();
  }

  iree::vm::ref<iree_vm_instance_t> instance_;
  iree::vm::ref<iree_vm_context_t> context_;
  iree::vm::ref<iree_hal_device_t> device_;
  iree::vm::ref<iree_hal_allocator_t> device_allocator_;
  iree::vm::ref<iree_vm_module_t> main_module_;
  iree::vm::ref<iree_vm_list_t> inputs_;
};
}  // namespace
}  // namespace iree

int main(int argc, char** argv) {
  IREE_TRACE_SCOPE0("main");

  // Pass through flags to benchmark (allowing --help to fall through).
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::benchmark::Initialize(&argc, argv);

  iree::IREEBenchmark iree_benchmark;
  iree_status_t status = iree_benchmark.Register();
  if (!iree_status_is_ok(status)) {
    int ret = static_cast<int>(iree_status_code(status));
    printf("%s\n", iree::Status(std::move(status)).ToString().c_str());
    return ret;
  }
  IREE_CHECK_OK(iree_hal_begin_profiling_from_flags(iree_benchmark.device()));
  ::benchmark::RunSpecifiedBenchmarks();
  IREE_CHECK_OK(iree_hal_end_profiling_from_flags(iree_benchmark.device()));
  return 0;
}
