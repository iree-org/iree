// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "samples/emitc_modules/add_module.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

class VMAddModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    iree_vm_module_t* add_module = nullptr;
    IREE_CHECK_OK(
        add_module_create(instance_, iree_allocator_system(), &add_module));

    std::vector<iree_vm_module_t*> modules = {add_module};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));

    iree_vm_module_release(add_module);
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

  StatusOr<int32_t> RunFunction(iree_string_view_t function_name, int32_t arg0,
                                int32_t arg1) {
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
    auto arg0_value = iree_vm_value_make_i32(arg0);
    auto arg1_value = iree_vm_value_make_i32(arg1);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_value(input_list.get(), &arg0_value));
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_value(input_list.get(), &arg1_value));
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

TEST_F(VMAddModuleTest, AddTest) {
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v,
      RunFunction(iree_make_cstring_view("add_module.add_and_double"), 17, 42));
  ASSERT_EQ(v, 118);
}

TEST_F(VMAddModuleTest, AddCallTest) {
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v,
      RunFunction(iree_make_cstring_view("add_module.test_call"), 17));
  ASSERT_EQ(v, 136);
}

}  // namespace
}  // namespace iree
