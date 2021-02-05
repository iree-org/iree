// Copyright 2020 Google LLC
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

#include "absl/flags/flag.h"
#include "absl/flags/internal/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

ABSL_FLAG(std::string, module_file, "-",
          "File containing the module to load that contains the entry "
          "function. Defaults to stdin.");

ABSL_FLAG(std::string, entry_function, "",
          "Name of a function contained in the module specified by module_file "
          "to run. If this is not set, all the exported functions will be "
          "benchmarked and they are expected to not have input arguments.");

ABSL_FLAG(std::string, driver, "vmla", "Backend driver to use.");

ABSL_FLAG(std::vector<std::string>, function_inputs, {},
          "A comma-separated list of of input buffers of the format:"
          "[shape]xtype=[value]\n"
          "2x2xi32=1 2 3 4\n"
          "Optionally, brackets may be used to separate the element values. "
          "They are ignored by the parser.\n"
          "2x2xi32=[[1 2][3 4]]\n"
          "Due to the absence of repeated flags in absl, commas should not be "
          "used to separate elements. They are reserved for separating input "
          "values:\n"
          "2x2xi32=[[1 2][3 4]], 1x2xf32=[[1 2]]");

ABSL_FLAG(std::string, function_inputs_file, "",
          "Provides a file for input shapes and optional values (see "
          "ParseToVariantListFromFile in vm_util.h for details)");

namespace iree {
namespace {

static void BenchmarkFunction(
    const std::string& benchmark_name, iree_vm_context_t* context,
    iree_vm_function_t function, iree_vm_list_t* inputs,
    const std::vector<RawSignatureParser::Description>& output_descs,
    benchmark::State& state) {
  IREE_TRACE_SCOPE_DYNAMIC(benchmark_name.c_str());
  IREE_TRACE_FRAME_MARK();

  // Benchmarking loop.
  for (auto _ : state) {
    IREE_TRACE_SCOPE0("BenchmarkIteration");
    IREE_TRACE_FRAME_MARK_NAMED("Iteration");
    vm::ref<iree_vm_list_t> outputs;
    IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr,
                                      output_descs.size(),
                                      iree_allocator_system(), &outputs));
    IREE_CHECK_OK(iree_vm_invoke(context, function, /*policy=*/nullptr, inputs,
                                 outputs.get(), iree_allocator_system()));
  }
}

void RegisterModuleBenchmarks(
    const std::string& function_name, iree_vm_context_t* context,
    iree_vm_function_t function, iree_vm_list_t* inputs,
    const std::vector<RawSignatureParser::Description>& output_descs) {
  auto benchmark_name = "BM_" + function_name;
  benchmark::RegisterBenchmark(benchmark_name.c_str(),
                               [benchmark_name, context, function, inputs,
                                output_descs](benchmark::State& state) -> void {
                                 BenchmarkFunction(benchmark_name, context,
                                                   function, inputs,
                                                   output_descs, state);
                               })
      // By default only the main thread is included in CPU time. Include all
      // the threads instead.
      ->MeasureProcessCPUTime()
      // To make single and multi-threaded benchmarks more comparable, use the
      // wall time to determine how many iterations to run. See
      // https://github.com/google/benchmark#cpu-timers,
      ->UseRealTime()
      // Report timing in milliseconds, which is the general order of magnitude
      // of model runs. The benchmark framework will print with precision
      // between 0 and 3 places after the decimal while aiming for three
      // significant digits. If we end up wanting precision beyond microseconds,
      // we can make this setting configurable with a custom command line flag.
      ->Unit(benchmark::kMillisecond);
}

Status GetModuleContentsFromFlags(std::string* out_contents) {
  IREE_TRACE_SCOPE0("GetModuleContentsFromFlags");
  auto module_file = absl::GetFlag(FLAGS_module_file);
  if (module_file == "-") {
    *out_contents = std::string{std::istreambuf_iterator<char>(std::cin),
                                std::istreambuf_iterator<char>()};
  } else {
    IREE_RETURN_IF_ERROR(file_io::GetFileContents(module_file, out_contents));
  }
  return OkStatus();
}

// TODO(hanchung): Consider to refactor this out and reuse in iree-run-module.
// This class helps organize required resources for IREE. The order of
// construction and destruction for resources matters. And the lifetime of
// resources also matters. The lifetime of IREEBenchmark should be as long as
// ::benchmark::RunSpecifiedBenchmarks() where the resources are used during
// benchmarking.
class IREEBenchmark {
 public:
  IREEBenchmark()
      : instance_(nullptr),
        device_(nullptr),
        hal_module_(nullptr),
        context_(nullptr),
        input_module_(nullptr){};
  ~IREEBenchmark() {
    IREE_TRACE_SCOPE0("IREEBenchmark::dtor");

    // Order matters.
    inputs_.reset();
    iree_vm_module_release(hal_module_);
    iree_vm_module_release(input_module_);
    iree_hal_device_release(device_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  };

  Status Register() {
    IREE_TRACE_SCOPE0("IREEBenchmark::Register");

    if (!instance_ || !device_ || !hal_module_ || !context_ || !input_module_) {
      IREE_RETURN_IF_ERROR(Init());
    }

    auto function_name = absl::GetFlag(FLAGS_entry_function);
    if (!function_name.empty()) {
      IREE_RETURN_IF_ERROR(RegisterSpecificFunction(function_name));
    } else {
      IREE_RETURN_IF_ERROR(RegisterAllExportedFunctions());
    }
    return iree::OkStatus();
  }

 private:
  Status Init() {
    IREE_TRACE_SCOPE0("IREEBenchmark::Init");
    IREE_TRACE_FRAME_MARK_BEGIN_NAMED("init");

    IREE_RETURN_IF_ERROR(GetModuleContentsFromFlags(&module_data_));

    IREE_RETURN_IF_ERROR(iree_hal_module_register_types());
    IREE_RETURN_IF_ERROR(
        iree_vm_instance_create(iree_allocator_system(), &instance_));

    // Create IREE's device and module.
    IREE_RETURN_IF_ERROR(
        iree::CreateDevice(absl::GetFlag(FLAGS_driver), &device_));
    IREE_RETURN_IF_ERROR(CreateHalModule(device_, &hal_module_));
    IREE_RETURN_IF_ERROR(LoadBytecodeModule(module_data_, &input_module_));

    // Order matters. The input module will likely be dependent on the hal
    // module.
    std::array<iree_vm_module_t*, 2> modules = {hal_module_, input_module_};
    IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), iree_allocator_system(),
        &context_));

    IREE_TRACE_FRAME_MARK_END_NAMED("init");
    return iree::OkStatus();
  }

  Status RegisterSpecificFunction(const std::string& function_name) {
    IREE_TRACE_SCOPE0("IREEBenchmark::RegisterSpecificFunction");

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(input_module_->lookup_function(
        input_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{function_name.data(), function_name.size()},
        &function));
    IREE_RETURN_IF_ERROR(ValidateFunctionAbi(function));

    // Construct inputs.
    std::vector<RawSignatureParser::Description> input_descs;
    IREE_RETURN_IF_ERROR(ParseInputSignature(function, &input_descs));
    if (!absl::GetFlag(FLAGS_function_inputs_file).empty()) {
      IREE_RETURN_IF_ERROR(ParseToVariantListFromFile(
          input_descs, iree_hal_device_allocator(device_),
          absl::GetFlag(FLAGS_function_inputs_file), &inputs_));
    } else {
      IREE_RETURN_IF_ERROR(
          ParseToVariantList(input_descs, iree_hal_device_allocator(device_),
                             absl::GetFlag(FLAGS_function_inputs), &inputs_));
    }

    // Creates output signature.
    std::vector<RawSignatureParser::Description> output_descs;
    IREE_RETURN_IF_ERROR(ParseOutputSignature(function, &output_descs));
    RegisterModuleBenchmarks(function_name, context_, function, inputs_.get(),
                             output_descs);
    return iree::OkStatus();
  }

  Status RegisterAllExportedFunctions() {
    IREE_TRACE_SCOPE0("IREEBenchmark::RegisterAllExportedFunctions");
    iree_vm_function_t function;
    iree_vm_module_signature_t signature =
        input_module_->signature(input_module_->self);
    for (iree_host_size_t i = 0; i < signature.export_function_count; ++i) {
      iree_string_view_t name;
      IREE_CHECK_OK(input_module_->get_function(input_module_->self,
                                                IREE_VM_FUNCTION_LINKAGE_EXPORT,
                                                i, &function, &name, nullptr));
      if (!ValidateFunctionAbi(function).ok()) continue;

      std::string function_name(name.data, name.size);
      std::vector<RawSignatureParser::Description> input_descs;
      IREE_RETURN_IF_ERROR(ParseInputSignature(function, &input_descs));
      if (!input_descs.empty()) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "expect not to have input arguments for '%.*s'",
                                (int)name.size, name.data);
      }
      std::vector<RawSignatureParser::Description> output_descs;
      IREE_RETURN_IF_ERROR(ParseOutputSignature(function, &output_descs));
      iree::RegisterModuleBenchmarks(function_name, context_, function,
                                     /*inputs=*/nullptr, output_descs);
    }
    return iree::OkStatus();
  }

  std::string module_data_;
  iree_vm_instance_t* instance_;
  iree_hal_device_t* device_;
  iree_vm_module_t* hal_module_;
  iree_vm_context_t* context_;
  iree_vm_module_t* input_module_;
  iree::vm::ref<iree_vm_list_t> inputs_;
};
}  // namespace
}  // namespace iree

int main(int argc, char** argv) {
  IREE_TRACE_SCOPE0("main");

  // We have to contend with two flag parsing libraries here: absl's and
  // benchmark's. To make matters worse, both define the `--help` flag. To
  // ensure that each is able to parse its own flags, we use an absl "internal"
  // function (still with public visibility) to parse while ignoring undefined
  // flags. If it sees `--help` it will exit here, so we include the benchmark
  // library usage information in the manually-set help output. Then we let
  // benchmark parse its flags. Finally we call the normal initialization
  // function to do other IREE initialization including flag parsing with
  // normal options. Any remaining flags will be unknown and result in an error.
  absl::SetProgramUsageMessage(
      "iree-benchmark-module \n"
      "    --module_file=module.vmfb\n"
      "    --entry_function=exported_function_to_benchmark\n"
      "      If this is not set, all the exported functions will be \n"
      "      benchmarked and they are expected to not have input arguments\n"
      "    [--function_inputs=2xi32=1 2,1x2xf32=2 1 | \n"
      "     --function_inputs_file=file_with_function_inputs]\n"
      "    [--driver=vmla]\n"
      "\n\n"
      "  Optional flags from third_party/benchmark/src/benchmark.cc:\n"
      "    [--benchmark_list_tests={true|false}]\n"
      "    [--benchmark_filter=<regex>]\n"
      "    [--benchmark_min_time=<min_time>]\n"
      "    [--benchmark_repetitions=<num_repetitions>]\n"
      "    [--benchmark_report_aggregates_only={true|false}]\n"
      "    [--benchmark_display_aggregates_only={true|false}]\n"
      "    [--benchmark_format=<console|json|csv>]\n"
      "    [--benchmark_out=<filename>]\n"
      "    [--benchmark_out_format=<json|console|csv>]\n"
      "    [--benchmark_color={auto|true|false}]\n"
      "    [--benchmark_counters_tabular={true|false}]\n"
      "    [--v=<verbosity>]\n");
  absl::flags_internal::ParseCommandLineImpl(
      argc, argv, absl::flags_internal::ArgvListAction::kRemoveParsedArgs,
      absl::flags_internal::UsageFlagsAction::kHandleUsage,
      absl::flags_internal::OnUndefinedFlag::kIgnoreUndefined);
  ::benchmark::Initialize(&argc, argv);
  iree_flags_parse_checked(&argc, &argv);
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  iree::IREEBenchmark iree_benchmark;
  auto status = iree_benchmark.Register();
  if (!status.ok()) {
    std::cout << status << std::endl;
    return static_cast<int>(status.code());
  }
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
