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
#include "iree/testing/gtest.h"
#include "iree/vm2/bytecode_dispatch_test_module.h"
#include "iree/vm2/bytecode_module.h"
#include "iree/vm2/module.h"
#include "iree/vm2/stack.h"

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
               IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_DEFAULT, &module))
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
  module->destroy(module->self);

  return function_names;
}

class VMBytecodeDispatchTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<TestParams> {
 protected:
  virtual void SetUp() {
    const auto* module_file_toc =
        iree::vm::bytecode_dispatch_test_module_create();
    CHECK_EQ(IREE_STATUS_OK,
             iree_vm_bytecode_module_create(
                 iree_const_byte_span_t{
                     reinterpret_cast<const uint8_t*>(module_file_toc->data),
                     module_file_toc->size},
                 IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_DEFAULT, &module_))
        << "Bytecode module failed to load";

    CHECK_EQ(IREE_STATUS_OK,
             module_->alloc_state(module_->self, IREE_ALLOCATOR_DEFAULT,
                                  &module_state_));

    // Since we only have a single state we pack it in the state_resolver ptr.
    iree_vm_state_resolver_t state_resolver = {
        module_state_,
        +[](void* state_resolver, iree_vm_module_t* module,
            iree_vm_module_state_t** out_module_state) -> iree_status_t {
          *out_module_state = (iree_vm_module_state_t*)state_resolver;
          return IREE_STATUS_OK;
        }};

    CHECK_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, &stack_));
  }

  virtual void TearDown() {
    iree_vm_stack_deinit(&stack_);
    module_->free_state(module_->self, module_state_);
    module_state_ = nullptr;
    module_->destroy(module_->self);
    module_ = nullptr;
  }

  iree_status_t RunFunction(absl::string_view function_name) {
    iree_vm_function_t function;
    CHECK_EQ(IREE_STATUS_OK,
             module_->lookup_function(
                 module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                 iree_string_view_t{function_name.data(), function_name.size()},
                 &function))
        << "Exported function '" << function_name << "' not found";

    iree_vm_stack_frame_t* entry_frame;
    iree_vm_stack_function_enter(&stack_, function, &entry_frame);

    iree_vm_execution_result_t result;
    iree_status_t execution_result =
        module_->execute(module_->self, &stack_, entry_frame, &result);

    iree_vm_stack_function_leave(&stack_);

    return execution_result;
  }

  iree_vm_module_t* module_ = nullptr;
  iree_vm_module_state_t* module_state_ = nullptr;
  iree_vm_stack_t stack_;
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
