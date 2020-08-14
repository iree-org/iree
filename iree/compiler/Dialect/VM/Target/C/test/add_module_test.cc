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

#include "iree/compiler/Dialect/VM/Target/C/test/add_module_test.h"

#include "iree/base/status.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/invocation.h"
#include "iree/vm/list.h"
#include "iree/vm/ref_cc.h"

namespace iree {
namespace {

typedef struct {
  int32_t first;
  int32_t second;
} i32_i32_t;

class VMAddModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    iree_vm_module_t* module_a = nullptr;
    IREE_CHECK_OK(module_a_create(iree_allocator_system(), &module_a));

    std::vector<iree_vm_module_t*> modules = {module_a};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), iree_allocator_system(),
        &context_));

    iree_vm_module_release(module_a);
  }

  virtual void TearDown() {
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  StatusOr<i32_i32_t> RunFunction(iree_string_view_t function_name,
                                  int32_t arg0, int32_t arg1) {
    // Lookup the entry function. This can be cached in an application if
    // multiple calls will be made.
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(
        iree_vm_context_resolve_function(
            context_, iree_make_cstring_view("module_a.test_function"),
            &function),
        "unable to resolve entry point");

    // Setup I/O lists and pass in the argument. The result list will be
    // populated upon return.
    vm::ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, 1, iree_allocator_system(), &input_list));
    auto arg0_value = iree_vm_value_make_i32(arg0);
    auto arg1_value = iree_vm_value_make_i32(arg1);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_value(input_list.get(), &arg0_value));
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_value(input_list.get(), &arg1_value));
    vm::ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, 2, iree_allocator_system(), &output_list));

    // Invoke the entry function to do our work. Runs synchronously.
    IREE_RETURN_IF_ERROR(iree_vm_invoke(context_, function,
                                        /*policy=*/nullptr, input_list.get(),
                                        output_list.get(),
                                        iree_allocator_system()));

    // Load the output result.
    iree_vm_value_t ret0_value, ret1_value;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value(output_list.get(), 0, &ret0_value));
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value(output_list.get(), 1, &ret1_value));
    return i32_i32_t{ret0_value.i32, ret1_value.i32};
  }

 private:
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
};

TEST_F(VMAddModuleTest, Example) {
  StatusOr<i32_i32_t> v0 =
      RunFunction(iree_make_cstring_view("module_a.test_function"), 17, 42);
  ASSERT_EQ(v0->first, 59);
  ASSERT_EQ(v0->second, 118);
}

}  // namespace
}  // namespace iree
