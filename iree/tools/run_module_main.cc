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

#include <iostream>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "iree/base/api_util.h"
#include "iree/base/file_io.h"
#include "iree/base/init.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/vm_util.h"
#include "iree/vm/bytecode_module.h"

ABSL_FLAG(std::string, input_file, "-",
          "File containing the module to load that contains the entry "
          "function. Defaults to stdin.");

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
  std::string contents;
  if (input_file == "-") {
    contents = std::string{std::istreambuf_iterator<char>(std::cin),
                           std::istreambuf_iterator<char>()};
  } else {
    ASSIGN_OR_RETURN(contents, file_io::GetFileContents(input_file));
  }
  return contents;
}

Status Run() {
  IREE_TRACE_SCOPE0("iree-run-module");

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
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_variant_list_alloc(output_descs.size(),
                                               IREE_ALLOCATOR_SYSTEM, &outputs),
                    IREE_LOC));

  std::cout << "EXEC @" << function_name << "\n";
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                   inputs, outputs, IREE_ALLOCATOR_SYSTEM),
                    IREE_LOC))
      << "invoking function " << function_name;

  RETURN_IF_ERROR(PrintVariantList(output_descs, outputs))
      << "printing results";

  // TODO(gcmn): Some nice wrappers to make this pattern shorter with generated
  // error messages.
  // Deallocate:
  RETURN_IF_ERROR(FromApiStatus(iree_vm_variant_list_free(inputs), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_vm_variant_list_free(outputs), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_vm_module_release(hal_module), IREE_LOC));
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_module_release(input_module), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_hal_device_release(device), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_vm_context_release(context), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_vm_instance_release(instance), IREE_LOC));
  return OkStatus();
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  iree::InitializeEnvironment(&argc, &argv);
  CHECK_OK(Run());
  return 0;
}

}  // namespace iree
