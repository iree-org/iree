// Copyright 2019 Google LLC
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

// Tests covering the dispatch logic for individual ops.
//
// bytecode_dispatch_test.mlir contains the functions used here for testing. We
// avoid defining the IR inline here so that we can run this test on platforms
// that we can't run the full MLIR compiler stack on.

#include "absl/strings/match.h"
#include "iree/base/logging.h"
#include "iree/testing/gtest.h"
#include "iree/vm/bytecode_dispatch_test_module.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/invocation.h"
#include "iree/vm/module.h"

namespace {

struct TestParams {
  std::string function_name;
};

std::ostream& operator<<(std::ostream& os, const TestParams& params) {
  return os << params.function_name;
}

std::vector<TestParams> GetModuleTestParams() {
  std::vector<TestParams> function_names;

  const auto* module_file_toc =
      iree::vm::bytecode_dispatch_test_module_create();
  iree_vm_module_t* module = nullptr;
  CHECK_EQ(IREE_STATUS_OK,
           iree_vm_bytecode_module_create(
               iree_const_byte_span_t{
                   reinterpret_cast<const uint8_t*>(module_file_toc->data),
                   module_file_toc->size},
               IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &module))
      << "Bytecode module failed to load";
  iree_vm_module_signature_t signature = module->signature(module->self);
  function_names.reserve(signature.export_function_count);
  for (int i = 0; i < signature.export_function_count; ++i) {
    iree_string_view_t name;
    CHECK_EQ(IREE_STATUS_OK,
             module->get_function(module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                                  i, nullptr, &name, nullptr));
    function_names.push_back({std::string(name.data, name.size)});
  }
  iree_vm_module_release(module);

  return function_names;
}

class VMBytecodeDispatchTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<TestParams> {
 protected:
  virtual void SetUp() {
    CHECK_EQ(IREE_STATUS_OK,
             iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_));

    const auto* module_file_toc =
        iree::vm::bytecode_dispatch_test_module_create();
    CHECK_EQ(IREE_STATUS_OK,
             iree_vm_bytecode_module_create(
                 iree_const_byte_span_t{
                     reinterpret_cast<const uint8_t*>(module_file_toc->data),
                     module_file_toc->size},
                 IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module_))
        << "Bytecode module failed to load";

    std::vector<iree_vm_module_t*> modules = {bytecode_module_};
    CHECK_EQ(IREE_STATUS_OK, iree_vm_context_create_with_modules(
                                 instance_, modules.data(), modules.size(),
                                 IREE_ALLOCATOR_SYSTEM, &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_status_t RunFunction(absl::string_view function_name) {
    iree_vm_function_t function;
    CHECK_EQ(IREE_STATUS_OK,
             bytecode_module_->lookup_function(
                 bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                 iree_string_view_t{function_name.data(), function_name.size()},
                 &function))
        << "Exported function '" << function_name << "' not found";

    return iree_vm_invoke(context_, function,
                          /*policy=*/nullptr, /*inputs=*/nullptr,
                          /*outputs=*/nullptr, IREE_ALLOCATOR_SYSTEM);
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

TEST_P(VMBytecodeDispatchTest, Check) {
  const auto& test_params = GetParam();
  bool expect_failure = absl::StartsWith(test_params.function_name, "fail_");

  iree_status_t result = RunFunction(test_params.function_name);
  if (result == IREE_STATUS_OK) {
    if (expect_failure) {
      GTEST_FAIL() << "Function expected failure but succeeded";
    } else {
      GTEST_SUCCEED();
    }
  } else {
    if (expect_failure) {
      GTEST_SUCCEED();
    } else {
      GTEST_FAIL() << "Function expected success but failed with error "
                   << result;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(VMIRFunctions, VMBytecodeDispatchTest,
                         ::testing::ValuesIn(GetModuleTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
