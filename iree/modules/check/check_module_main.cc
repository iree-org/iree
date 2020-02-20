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
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/file_io.h"
#include "iree/base/init.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/modules/check/native_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/vm_util.h"
#include "iree/vm/bytecode_module.h"

ABSL_FLAG(std::string, input_file, "-",
          "File containing the module to load that contains the entry "
          "function. Defaults to stdin.");

ABSL_FLAG(std::string, driver, "interpreter", "Backend driver to use.");

ABSL_FLAG(
    bool, expect_failure, false,
    "Whether running module is expected to fail. If set, failing "
    "statuses from function evaluation are logged and ignored and all "
    "evaluations succeeding is considered an error and will return a failure.");

namespace iree {
namespace {

StatusOr<std::string> GetModuleContentsFromFlags() {
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
  iree_vm_module_t* check_module = nullptr;
  check_native_module_create(IREE_ALLOCATOR_SYSTEM, &check_module);

  auto run_function = [&](int ordinal) -> Status {
    iree_vm_function_t function;
    RETURN_IF_ERROR(FromApiStatus(
        iree_vm_module_lookup_function_by_ordinal(
            input_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, ordinal, &function),
        IREE_LOC))
        << "Looking up function export " << ordinal;
    iree_string_view_t function_name_iree_sv = iree_vm_function_name(&function);
    // TODO(gcmn): Implicit conversion from iree to absl string view.
    auto function_name = absl::string_view(function_name_iree_sv.data,
                                           function_name_iree_sv.size);

    RETURN_IF_ERROR(ValidateFunctionAbi(function));
    ASSIGN_OR_RETURN(auto input_descs, ParseInputSignature(function));
    ASSIGN_OR_RETURN(auto output_descs, ParseOutputSignature(function));
    if (!input_descs.empty() || !output_descs.empty()) {
      iree_string_view_t sig_f = iree_vm_function_reflection_attr(
          &function, iree_make_cstring_view("f"));
      RawSignatureParser sig_parser;
      auto sig_str = sig_parser.FunctionSignatureToString(
          absl::string_view{sig_f.data, sig_f.size});
      if (!sig_str.has_value()) {
        return InvalidArgumentErrorBuilder(IREE_LOC)
               << "Parsing function signature '" << sig_f.data << "': "
               << sig_parser.GetError().value_or("<NO ERROR AND NO VALUE>");
      }
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Expected function with no inputs or outputs, but "
             << function_name << "' has signature '" << sig_str.value() << "'";
    }
    iree_vm_context_t* context = nullptr;
    // Order matters. The input module will likely be dependent on the hal and
    // check modules.
    std::array<iree_vm_module_t*, 3> modules = {hal_module, check_module,
                                                input_module};
    RETURN_IF_ERROR(FromApiStatus(iree_vm_context_create_with_modules(
                                      instance, modules.data(), modules.size(),
                                      IREE_ALLOCATOR_SYSTEM, &context),
                                  IREE_LOC))
        << "creating context";
    std::cout << "EXEC @" << function_name << "\n";
    // Still release the context even if invocation failed to avoid leaks.
    auto status = Annotate(
        FromApiStatus(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                     /*inputs=*/nullptr, /*outputs=*/nullptr,
                                     IREE_ALLOCATOR_SYSTEM),
                      IREE_LOC),
        absl::StrCat("invoking function ", function_name));
    RETURN_IF_ERROR(FromApiStatus(iree_vm_context_release(context), IREE_LOC));
    return status;
  };
  Status evaluate_status = OkStatus();
  auto module_signature = iree_vm_module_signature(input_module);
  for (int i = 0; i < module_signature.export_function_count; ++i) {
    evaluate_status = run_function(i);
    if (!evaluate_status.ok()) {
      break;
    }
  }

  // TODO(gcmn): Some nice wrappers to make this pattern shorter with generated
  // error messages.
  // Deallocate:
  RETURN_IF_ERROR(FromApiStatus(iree_vm_module_release(hal_module), IREE_LOC));
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_module_release(check_module), IREE_LOC));
  RETURN_IF_ERROR(
      FromApiStatus(iree_vm_module_release(input_module), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_hal_device_release(device), IREE_LOC));
  RETURN_IF_ERROR(FromApiStatus(iree_vm_instance_release(instance), IREE_LOC));

  if (absl::GetFlag(FLAGS_expect_failure)) {
    if (evaluate_status.ok()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Test passed but expected failure";
    }
    std::cout << "Test failed as expected " << evaluate_status << "\n";
    return OkStatus();
  }
  return evaluate_status;
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);
  auto status = Run();
  if (!status.ok()) {
    LOG(ERROR) << status << "\n";
    return 1;
  }
  return 0;
}

}  // namespace iree
