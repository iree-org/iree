// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "samples/emitc_modules/import_module_a.h"
#include "samples/emitc_modules/import_module_b.h"

namespace iree {
namespace {

class VMImportModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    iree_vm_module_t* module_a = nullptr;
    IREE_CHECK_OK(
        module_a_create(instance_, iree_allocator_system(), &module_a));

    iree_vm_module_t* module_b = nullptr;
    IREE_CHECK_OK(
        module_b_create(instance_, iree_allocator_system(), &module_b));

    // Note: order matters as module_a imports from module_b
    std::vector<iree_vm_module_t*> modules = {module_b, module_a};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));

    iree_vm_module_release(module_a);
    iree_vm_module_release(module_b);
  }

  virtual void TearDown() {
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  StatusOr<int32_t> RunFunction(iree_string_view_t function_name, int32_t arg) {
    // Lookup the entry function. This can be cached in an application if
    // multiple calls will be made.
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(
        iree_vm_context_resolve_function(context_, function_name, &function),
        "unable to resolve entry point");

    // Setup I/O lists and pass in the argument. The result list will be
    // populated upon return.
    vm::ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, iree_allocator_system(),
                                             &input_list));
    auto arg_value = iree_vm_value_make_i32(arg);
    IREE_RETURN_IF_ERROR(iree_vm_list_push_value(input_list.get(), &arg_value));
    vm::ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, iree_allocator_system(),
                                             &output_list));

    // Invoke the entry function to do our work. Runs synchronously.
    IREE_RETURN_IF_ERROR(
        iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/nullptr, input_list.get(), output_list.get(),
                       iree_allocator_system()));

    // Load the output result.
    iree_vm_value_t ret_value;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value(output_list.get(), 0, &ret_value));
    return ret_value.i32;
  }

 private:
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
};

TEST_F(VMImportModuleTest, ImportedCallTest) {
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v, RunFunction(iree_make_cstring_view("module_a.test_call"), 9));
  ASSERT_EQ(v, 81);
}

}  // namespace
}  // namespace iree
