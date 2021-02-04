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

#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#include "iree/vm/test/emitc/arithmetic_ops.module"
#include "iree/vm/test/emitc/shift_ops.module"

namespace {

typedef iree_status_t (*create_function_t)(iree_allocator_t,
                                           iree_vm_module_t**);

struct TestParams {
  std::string module_name;
  std::string local_name;
  create_function_t create_function;
};

struct ModuleDescription {
  iree_vm_native_module_descriptor_t descriptor;
  create_function_t create_function;
};

std::ostream& operator<<(std::ostream& os, const TestParams& params) {
  return os << absl::StrReplaceAll(params.local_name, {{":", "_"}, {".", "_"}});
}

std::vector<TestParams> GetModuleTestParams() {
  std::vector<TestParams> test_params;

  // TODO(simon-camp): get these automatically
  std::vector<ModuleDescription> modules = {
      {arithmetic_ops_descriptor_, arithmetic_ops_create},
      {shift_ops_descriptor_, shift_ops_create}};

  for (size_t i = 0; i < modules.size(); i++) {
    iree_vm_native_module_descriptor_t descriptor = modules[i].descriptor;
    create_function_t function = modules[i].create_function;

    std::string module_name =
        std::string(descriptor.module_name.data, descriptor.module_name.size);

    for (iree_host_size_t i = 0; i < descriptor.export_count; i++) {
      iree_vm_native_export_descriptor_t export_descriptor =
          descriptor.exports[i];
      std::string local_name = std::string(export_descriptor.local_name.data,
                                           export_descriptor.local_name.size);
      test_params.push_back({module_name, local_name, function});
    }
  }

  return test_params;
}

class VMCModuleTest : public ::testing::Test,
                      public ::testing::WithParamInterface<TestParams> {
 protected:
  virtual void SetUp() {
    const auto& test_params = GetParam();

    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    iree_vm_module_t* module_ = nullptr;
    IREE_CHECK_OK(
        test_params.create_function(iree_allocator_system(), &module_))
        << "Module failed to load";

    std::vector<iree_vm_module_t*> modules = {module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), iree_allocator_system(),
        &context_));

    iree_vm_module_release(module_);
  }

  virtual void TearDown() {
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_status_t RunFunction(std::string module_name, std::string local_name) {
    std::string qualified_name = module_name + "." + local_name;
    iree_vm_function_t function;
    IREE_CHECK_OK(iree_vm_context_resolve_function(
        context_,
        iree_string_view_t{qualified_name.data(), qualified_name.size()},
        &function))
        << "Exported function '" << local_name << "' not found";

    return iree_vm_invoke(context_, function,
                          /*policy=*/nullptr, /*inputs=*/nullptr,
                          /*outputs=*/nullptr, iree_allocator_system());
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

TEST_P(VMCModuleTest, Check) {
  const auto& test_params = GetParam();
  bool expect_failure = absl::StartsWith(test_params.local_name, "fail_");

  iree::Status result =
      RunFunction(test_params.module_name, test_params.local_name);
  if (result.ok()) {
    if (expect_failure) {
      GTEST_FAIL() << "Function expected failure but succeeded";
    } else {
      GTEST_SUCCEED();
    }
  } else {
    if (expect_failure) {
      GTEST_SUCCEED();
    } else {
      GTEST_FAIL() << "Function expected success but failed with error: "
                   << result.ToString();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(VMIRFunctions, VMCModuleTest,
                         ::testing::ValuesIn(GetModuleTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
