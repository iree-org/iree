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
#include "iree/hal/api.h"
#include "iree/modules/check/check_test_module.h"
#include "iree/modules/check/native_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref.h"

namespace {

class CheckTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    // NOTE: we only use the HAL types here, we don't need the full module.
    // TODO(benvanik): move to instance-based registration.
    IREE_ASSERT_OK(iree_hal_module_register_types());

    IREE_CHECK_OK(iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_));

    IREE_CHECK_OK(
        check_native_module_create(IREE_ALLOCATOR_SYSTEM, &native_module_))
        << "Native module failed to init";

    const auto* module_file_toc = iree::check::check_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module_))
        << "Bytecode module failed to load";

    std::vector<iree_vm_module_t*> modules = {native_module_, bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), IREE_ALLOCATOR_SYSTEM,
        &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(native_module_);
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
    if (inputs_) iree_vm_variant_list_free(inputs_);
    if (outputs_) iree_vm_variant_list_free(outputs_);
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
  iree_vm_module_t* native_module_ = nullptr;
  iree_vm_variant_list_t* inputs_ = nullptr;
  iree_vm_variant_list_t* outputs_ = nullptr;
};

TEST_F(CheckTest, ExpectTrueSuccess) {
  iree_vm_value_t arg = IREE_VM_VALUE_MAKE_I32(1);
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &inputs_));
  IREE_ASSERT_OK(iree_vm_variant_list_append_value(inputs_, arg));
  IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("expectTrue"),
                                /*policy=*/nullptr, inputs_, outputs_,
                                IREE_ALLOCATOR_SYSTEM));
}

TEST_F(CheckTest, ExpectTrueFailure) {
  iree_vm_value_t arg = IREE_VM_VALUE_MAKE_I32(0);
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &inputs_));
  IREE_ASSERT_OK(iree_vm_variant_list_append_value(inputs_, arg));
  auto status = iree_vm_invoke(context_, LookupFunction("expectTrue"),
                               /*policy=*/nullptr, inputs_, outputs_,
                               IREE_ALLOCATOR_SYSTEM);
  ASSERT_NE(IREE_STATUS_OK, status);
}

}  // namespace
