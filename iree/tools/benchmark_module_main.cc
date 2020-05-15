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
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "iree/base/api_util.h"
#include "iree/base/file_io.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/vm_util.h"
#include "iree/vm/bytecode_module.h"

// TODO(gcmn): Allow stdin in a non-gross way. The benchmark framework invokes
// the benchmarking function multiple times, so we have to do something to only
// process stdin once. Probably requires dynamic benchmark registration.
ABSL_FLAG(std::string, input_file, "",
          "File containing the module to load that contains the entry "
          "function. Required and cannot be stdin.");

ABSL_FLAG(std::string, entry_function, "",
          "Name of a function contained in the module specified by input_file "
          "to run.");

ABSL_FLAG(std::string, driver, "vmla", "Backend driver to use.");

ABSL_FLAG(std::vector<std::string>, inputs, {},
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

namespace iree {
namespace {

StatusOr<std::string> GetModuleContentsFromFlags() {
  IREE_TRACE_SCOPE0("GetModuleContentsFromFlags");
  auto input_file = absl::GetFlag(FLAGS_input_file);
  if (input_file.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "input_file must be specified";
  }
  return file_io::GetFileContents(input_file);
}

Status Run(::benchmark::State& state) {
  IREE_TRACE_SCOPE0("iree-benchmark-module");

  RETURN_IF_ERROR(FromApiStatus(iree_hal_module_register_types(), IREE_LOC))
      << "registering HAL types";
  iree_vm_instance_t* instance = nullptr;
  RETURN_IF_ERROR(FromApiStatus(
      iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance), IREE_LOC))
      << "creating instance";

  ASSIGN_OR_RETURN(auto module_data, GetModuleContentsFromFlags());
  iree_vm_module_t* input_module = nullptr;
  RETURN_IF_ERROR(LoadBytecodeModule(module_data, &input_module));

  iree_hal_device_t* device = nullptr;
  RETURN_IF_ERROR(CreateDevice(absl::GetFlag(FLAGS_driver), &device));
  iree_vm_module_t* hal_module = nullptr;
  RETURN_IF_ERROR(CreateHalModule(device, &hal_module));

  iree_vm_context_t* context = nullptr;
  // Order matters. The input module will likely be dependent on the hal module.
  std::array<iree_vm_module_t*, 2> modules = {hal_module, input_module};
  RETURN_IF_ERROR(FromApiStatus(iree_vm_context_create_with_modules(
                                    instance, modules.data(), modules.size(),
                                    IREE_ALLOCATOR_SYSTEM, &context),
                                IREE_LOC))
      << "creating context";

  std::string function_name = absl::GetFlag(FLAGS_entry_function);
  iree_vm_function_t function;
  RETURN_IF_ERROR(FromApiStatus(
      input_module->lookup_function(
          input_module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_string_view_t{function_name.data(), function_name.size()},
          &function),
      IREE_LOC))
      << "looking up function '" << function_name << "'";

  RETURN_IF_ERROR(ValidateFunctionAbi(function));
  ASSIGN_OR_RETURN(auto input_descs, ParseInputSignature(function));

  ASSIGN_OR_RETURN(
      iree_vm_variant_list_t * inputs,
      ParseToVariantList(input_descs, iree_hal_device_allocator(device),
                         absl::GetFlag(FLAGS_inputs)));

  ASSIGN_OR_RETURN(auto output_descs, ParseOutputSignature(function));

  iree_vm_variant_list_t* outputs = nullptr;

  // Execute once to make sure any first-iteration outliers are eliminated (e.g.
  // JITing the SPIR-V) and clearly separate out benchmark-related problems in
  // future debugging.
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_variant_list_alloc(output_descs.size(),
                                               IREE_ALLOCATOR_SYSTEM, &outputs),
                    IREE_LOC));
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                   inputs, outputs, IREE_ALLOCATOR_SYSTEM),
                    IREE_LOC));
  iree_vm_variant_list_free(outputs);

  for (auto _ : state) {
    // No status conversions and conditional returns in the benchmarked inner
    // loop.
    IREE_CHECK_OK(iree_vm_variant_list_alloc(output_descs.size(),
                                             IREE_ALLOCATOR_SYSTEM, &outputs));
    IREE_CHECK_OK(iree_vm_invoke(context, function, /*policy=*/nullptr, inputs,
                                 outputs, IREE_ALLOCATOR_SYSTEM));
    iree_vm_variant_list_free(outputs);
  }

  iree_vm_variant_list_free(inputs);
  iree_vm_module_release(hal_module);
  iree_vm_module_release(input_module);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return OkStatus();
}

void BM_RunModule(benchmark::State& state) {
  // Delegate to a status-returning function so we can use the status macros.
  CHECK_OK(Run(state));
}

// By default only the main thread is included in CPU time. Include all the
// threads instead. To make single and multi-threaded benchmarks more
// comparable, use the wall time to determine how many iterations to run.
// See https://github.com/google/benchmark#cpu-timers,
BENCHMARK(BM_RunModule)->MeasureProcessCPUTime()->UseRealTime();

}  // namespace

}  // namespace iree
