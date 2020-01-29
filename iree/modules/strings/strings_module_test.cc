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

#include "iree/modules/strings/strings_module.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/modules/strings/strings_module_test_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/types.h"

using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

namespace {

class StringsModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_));

    IREE_CHECK_OK(strings_module_register_types());

    IREE_CHECK_OK(
        strings_module_create(IREE_ALLOCATOR_SYSTEM, &strings_module_))
        << "Native module failed to init";

    const auto* module_file_toc =
        iree::strings_module_test::strings_module_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module_))
        << "Bytecode module failed to load";

    std::vector<iree_vm_module_t*> modules = {strings_module_,
                                              bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), IREE_ALLOCATOR_SYSTEM,
        &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(strings_module_);
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_function_t LookupFunction(absl::string_view function_name) {
    iree_vm_function_t function;
    IREE_CHECK_OK(bytecode_module_->lookup_function(
        bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{function_name.data(), function_name.size()},
        &function))
        << "Exported function '" << function_name << "' not found";
    return function;
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* strings_module_ = nullptr;
};

TEST_F(StringsModuleTest, Run) {
  const int input = 42;
  std::string expected_output = "42\n";

  // Construct the input list for execution.
  iree_vm_variant_list_t* inputs = nullptr;
  IREE_ASSERT_OK(iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &inputs));

  // Add the value parameter.
  iree_vm_value_t value = IREE_VM_VALUE_MAKE_I32(input);
  IREE_ASSERT_OK(iree_vm_variant_list_append_value(inputs, value));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_variant_list_t* outputs = nullptr;
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(0, IREE_ALLOCATOR_SYSTEM, &outputs));

  CaptureStdout();
  IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("print_example_func"),
                                /*policy=*/nullptr, inputs, outputs,
                                IREE_ALLOCATOR_SYSTEM));
  EXPECT_EQ(GetCapturedStdout(), expected_output);

  iree_vm_variant_list_free(inputs);
  iree_vm_variant_list_free(outputs);
}

}  // namespace
