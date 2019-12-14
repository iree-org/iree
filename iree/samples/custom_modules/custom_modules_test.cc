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
#include "iree/vm2/bytecode_module.h"
#include "iree/vm2/stack.h"

namespace {

class CustomModulesTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    CHECK_EQ(IREE_STATUS_OK, iree_custom_native_module_register_types());

    CHECK_EQ(IREE_STATUS_OK, iree_custom_native_module_create(
                                 IREE_ALLOCATOR_DEFAULT, &native_module_))
        << "Native module failed to init";

    CHECK_EQ(IREE_STATUS_OK, native_module_->alloc_state(
                                 native_module_->self, IREE_ALLOCATOR_DEFAULT,
                                 &native_module_state_));

    const auto* module_file_toc =
        iree::samples::custom_modules::custom_modules_test_module_create();
    CHECK_EQ(
        IREE_STATUS_OK,
        iree_vm_bytecode_module_create(
            iree_const_byte_span_t{
                reinterpret_cast<const uint8_t*>(module_file_toc->data),
                module_file_toc->size},
            IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_DEFAULT, &bytecode_module_))
        << "Bytecode module failed to load";

    CHECK_EQ(IREE_STATUS_OK, bytecode_module_->alloc_state(
                                 bytecode_module_->self, IREE_ALLOCATOR_DEFAULT,
                                 &bytecode_module_state_));

    // NOTE: temporary, will happen in rt::Context.
    for (int i = 0; i < bytecode_module_->signature(bytecode_module_->self)
                            .import_function_count;
         ++i) {
      iree_string_view_t import_name;
      CHECK_EQ(IREE_STATUS_OK,
               bytecode_module_->get_function(
                   bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_IMPORT, i,
                   nullptr, &import_name, nullptr));
      // Eat `custom.` from the name.
      import_name.data += 7;
      import_name.size -= 7;
      iree_vm_function_t import_function;
      CHECK_EQ(IREE_STATUS_OK,
               native_module_->lookup_function(native_module_->self,
                                               IREE_VM_FUNCTION_LINKAGE_EXPORT,
                                               import_name, &import_function));
      CHECK_EQ(IREE_STATUS_OK, bytecode_module_->resolve_import(
                                   bytecode_module_->self,
                                   bytecode_module_state_, i, import_function));
    }

    iree_vm_state_resolver_t state_resolver = {
        this,
        +[](void* state_resolver, iree_vm_module_t* module,
            iree_vm_module_state_t** out_module_state) -> iree_status_t {
          auto* test = reinterpret_cast<CustomModulesTest*>(state_resolver);
          if (module == test->bytecode_module_) {
            *out_module_state = test->bytecode_module_state_;
            return IREE_STATUS_OK;
          } else if (module == test->native_module_) {
            *out_module_state = test->native_module_state_;
            return IREE_STATUS_OK;
          } else {
            return IREE_STATUS_NOT_FOUND;
          }
        }};

    CHECK_EQ(IREE_STATUS_OK, iree_vm_stack_init(state_resolver, &stack_));
  }

  virtual void TearDown() {
    iree_vm_stack_deinit(&stack_);

    bytecode_module_->free_state(bytecode_module_->self,
                                 bytecode_module_state_);
    bytecode_module_state_ = nullptr;
    bytecode_module_->destroy(bytecode_module_->self);
    bytecode_module_ = nullptr;

    native_module_->free_state(native_module_->self, native_module_state_);
    native_module_state_ = nullptr;
    native_module_->destroy(native_module_->self);
    native_module_ = nullptr;
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

  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
  iree_vm_module_state_t* bytecode_module_state_ = nullptr;
  iree_vm_module_state_t* native_module_state_ = nullptr;
  iree_vm_stack_t stack_;
};

TEST_F(CustomModulesTest, Run) {
  iree_vm_function_t function = LookupFunction("reverseAndPrint");

  iree_vm_stack_frame_t* entry_frame;
  iree_vm_stack_function_enter(&stack_, function, &entry_frame);

  // Pass in the message and number of times to print it.
  // TODO(benvanik): make a macro/magic.
  entry_frame->registers.ref_register_count = 1;
  memset(entry_frame->registers.ref, 0,
         sizeof(iree_vm_ref_t) * entry_frame->registers.ref_register_count);
  ASSERT_EQ(IREE_STATUS_OK,
            iree_custom_message_wrap(iree_make_cstring_view("hello world!"),
                                     IREE_ALLOCATOR_DEFAULT,
                                     &entry_frame->registers.ref[0]));
  entry_frame->registers.i32[0] = 5;

  // TODO(benvanik): replace with macro? helper for none/i32/etc
  static const union {
    uint8_t reserved[2];
    iree_vm_register_list_t list;
  } return_registers = {
      {1, 0 | IREE_REF_REGISTER_MOVE_BIT | IREE_REF_REGISTER_TYPE_BIT}};
  entry_frame->return_registers = &return_registers.list;

  iree_vm_execution_result_t result;
  iree_status_t execution_status = bytecode_module_->execute(
      bytecode_module_->self, &stack_, entry_frame, &result);

  // Read back the message that we reversed inside of the module.
  char result_buffer[256];
  ASSERT_EQ(IREE_STATUS_OK, iree_custom_message_read_value(
                                &entry_frame->registers.ref[0], result_buffer,
                                ABSL_ARRAYSIZE(result_buffer)));
  EXPECT_STREQ("!dlrow olleh", result_buffer);

  iree_vm_stack_function_leave(&stack_);

  ASSERT_EQ(IREE_STATUS_OK, execution_status);
}

}  // namespace
