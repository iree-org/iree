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
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/file_io.h"
#include "iree/base/init.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/modules/check/native_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tools/vm_util.h"
#include "iree/vm/bytecode_module.h"

// On Windows stdin defaults to text mode and will get weird line ending
// expansion that will corrupt the input binary.
#if defined(IREE_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <io.h>
#define IREE_FORCE_BINARY_STDIN() setmode(_fileno(stdin), O_BINARY)
#else
#define IREE_FORCE_BINARY_STDIN()
#endif  // IREE_PLATFORM_WINDOWS

ABSL_FLAG(std::string, driver, "vmla", "Backend driver to use.");

ABSL_FLAG(
    bool, expect_failure, false,
    "Whether running module is expected to fail. If set, failing "
    "statuses from function evaluation are logged and ignored and all "
    "evaluations succeeding is considered an error and will return a failure. "
    "Mostly useful for testing the binary doesn't crash for failing tests.");

namespace iree {
namespace {

class CheckModuleTest : public ::testing::Test {
 public:
  explicit CheckModuleTest(iree_vm_instance_t* instance,
                           std::array<iree_vm_module_t*, 3> modules,
                           iree_vm_function_t function)
      : instance_(instance), modules_(modules), function_(function) {}
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_context_create_with_modules(
        instance_, modules_.data(), modules_.size(), iree_allocator_system(),
        &context_));
  }
  void TearDown() override { iree_vm_context_release(context_); }

  void TestBody() override {
    IREE_EXPECT_OK(iree_vm_invoke(context_, function_, /*policy=*/nullptr,
                                  /*inputs=*/nullptr, /*outputs=*/nullptr,
                                  iree_allocator_system()));
  }

 private:
  iree_vm_instance_t* instance_ = nullptr;
  std::array<iree_vm_module_t*, 3> modules_;
  iree_vm_function_t function_;

  iree_vm_context_t* context_ = nullptr;
};

StatusOr<int> Run(std::string input_file_path) {
  IREE_TRACE_SCOPE0("iree-check-module");

  IREE_RETURN_IF_ERROR(iree_hal_module_register_types())
      << "registering HAL types";
  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance))
      << "creating instance";

  std::string module_data;
  if (input_file_path == "-") {
    module_data = std::string{std::istreambuf_iterator<char>(std::cin),
                              std::istreambuf_iterator<char>()};
  } else {
    IREE_ASSIGN_OR_RETURN(module_data,
                          file_io::GetFileContents(input_file_path));
  }

  iree_vm_module_t* input_module = nullptr;
  IREE_RETURN_IF_ERROR(LoadBytecodeModule(module_data, &input_module));

  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(CreateDevice(absl::GetFlag(FLAGS_driver), &device));
  iree_vm_module_t* hal_module = nullptr;
  IREE_RETURN_IF_ERROR(CreateHalModule(device, &hal_module));
  iree_vm_module_t* check_module = nullptr;
  check_native_module_create(iree_allocator_system(), &check_module);

  std::array<iree_vm_module_t*, 3> modules = {hal_module, check_module,
                                              input_module};
  auto module_signature = iree_vm_module_signature(input_module);
  for (int ordinal = 0; ordinal < module_signature.export_function_count;
       ++ordinal) {
    iree_vm_function_t function;
    iree_string_view_t export_name_sv;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        input_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, ordinal, &function,
        &export_name_sv))
        << "Looking up function export " << ordinal;

    // TODO(gcmn): Implicit conversion from iree to absl string view.
    auto export_name =
        absl::string_view(export_name_sv.data, export_name_sv.size);

    iree_string_view_t module_name_iree_sv = iree_vm_module_name(input_module);
    auto module_name =
        absl::string_view(module_name_iree_sv.data, module_name_iree_sv.size);
    if (absl::StartsWith(export_name, "__") ||
        export_name.find('$') != absl::string_view::npos) {
      // Skip internal or special functions.
      continue;
    }

    IREE_RETURN_IF_ERROR(ValidateFunctionAbi(function));
    IREE_ASSIGN_OR_RETURN(auto input_descs, ParseInputSignature(function));
    IREE_ASSIGN_OR_RETURN(auto output_descs, ParseOutputSignature(function));
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
             << export_name << "' has signature '" << sig_str.value() << "'";
    }

    ::testing::RegisterTest(
        module_name.data(), export_name.data(), nullptr,
        std::to_string(ordinal).c_str(), __FILE__, __LINE__,
        [&instance, modules, function]() -> CheckModuleTest* {
          return new CheckModuleTest(instance, modules, function);
        });
  }
  int ret = RUN_ALL_TESTS();

  iree_vm_module_release(hal_module);
  iree_vm_module_release(check_module);
  iree_vm_module_release(input_module);
  iree_hal_device_release(device);
  iree_vm_instance_release(instance);

  return ret;
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  InitializeEnvironment(&argc, &argv);
  IREE_FORCE_BINARY_STDIN();

  if (argc < 2) {
    LOG(ERROR)
        << "A binary module file path to run (or - for stdin) must be passed";
    return -1;
  }
  auto input_file_path = std::string(argv[1]);

  auto ret_or = Run(std::move(input_file_path));
  int ret = ret_or.ok() ? ret_or.value() : 1;
  if (absl::GetFlag(FLAGS_expect_failure)) {
    if (ret == 0) {
      std::cout << "Test passed but expected failure\n";
      std::cout << ret_or.status();
      return 1;
    }
    std::cout << "Test failed as expected\n";
    return 0;
  }

  return ret;
}

}  // namespace iree
