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

// Tests that our bytecode module can call through into our native module.

#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/samples/custom_modules/custom_modules_test_module.h"
#include "iree/samples/custom_modules/native_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

namespace {

class CustomModulesTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    CHECK_EQ(IREE_STATUS_OK,
             iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_));

    CHECK_EQ(IREE_STATUS_OK, iree_custom_native_module_register_types());

    CHECK_EQ(IREE_STATUS_OK, iree_custom_native_module_create(
                                 IREE_ALLOCATOR_SYSTEM, &native_module_))
        << "Native module failed to init";

    const auto* module_file_toc =
        iree::samples::custom_modules::custom_modules_test_module_create();
    CHECK_EQ(IREE_STATUS_OK,
             iree_vm_bytecode_module_create(
                 iree_const_byte_span_t{
                     reinterpret_cast<const uint8_t*>(module_file_toc->data),
                     module_file_toc->size},
                 IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module_))
        << "Bytecode module failed to load";

    std::vector<iree_vm_module_t*> modules = {native_module_, bytecode_module_};
    CHECK_EQ(IREE_STATUS_OK, iree_vm_context_create_with_modules(
                                 instance_, modules.data(), modules.size(),
                                 IREE_ALLOCATOR_SYSTEM, &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(native_module_);
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_function_t LookupFunction(absl::string_view function_name) {
    iree_vm_function_t function;
    CHECK_EQ(IREE_STATUS_OK,
             bytecode_module_->lookup_function(
                 bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                 iree_string_view_t{function_name.data(), function_name.size()},
                 &function))
        << "Exported function '" << function_name << "' not found";
    return function;
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
};

TEST_F(CustomModulesTest, Run) {
  // Allocate one of our custom message types to pass in.
  iree_vm_ref_t input_message = {0};
  ASSERT_EQ(IREE_STATUS_OK,
            iree_custom_message_wrap(iree_make_cstring_view("hello world!"),
                                     IREE_ALLOCATOR_SYSTEM, &input_message));
  iree_vm_value_t count = IREE_VM_VALUE_MAKE_I32(5);

  // Pass in the message and number of times to print it.
  // TODO(benvanik): make a macro/magic.
  iree_vm_variant_list_t* inputs = nullptr;
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_alloc(2, IREE_ALLOCATOR_SYSTEM, &inputs));
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_append_ref_move(inputs, &input_message));
  ASSERT_EQ(IREE_STATUS_OK, iree_vm_variant_list_append_value(inputs, count));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_variant_list_t* outputs = nullptr;
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &outputs));

  // Synchronously invoke the function.
  ASSERT_EQ(IREE_STATUS_OK,
            iree_vm_invoke(context_, LookupFunction("reverseAndPrint"),
                           /*policy=*/nullptr, inputs, outputs,
                           IREE_ALLOCATOR_SYSTEM));
  iree_vm_variant_list_free(inputs);

  // Read back the message that we reversed inside of the module.
  iree_vm_ref_t* reversed_message = &iree_vm_variant_list_get(outputs, 0)->ref;
  char result_buffer[256];
  ASSERT_EQ(IREE_STATUS_OK,
            iree_custom_message_read_value(reversed_message, result_buffer,
                                           ABSL_ARRAYSIZE(result_buffer)));
  EXPECT_STREQ("!dlrow olleh", result_buffer);

  iree_vm_variant_list_free(outputs);
}

}  // namespace
