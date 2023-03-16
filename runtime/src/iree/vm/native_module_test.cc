// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/native_module_test.h"

#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/invocation.h"
#include "iree/vm/list.h"
#include "iree/vm/ref.h"
#include "iree/vm/value.h"

namespace iree {
namespace {

// Test suite that uses module_a and module_b defined in native_module_test.h.
// Both modules are put in a context and the module_b.entry function can be
// executed with RunFunction.
class VMNativeModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    // Create both modules shared instances. These are generally immutable and
    // can be shared by multiple contexts.
    iree_vm_module_t* module_a = nullptr;
    IREE_CHECK_OK(
        module_a_create(instance_, iree_allocator_system(), &module_a));
    iree_vm_module_t* module_b = nullptr;
    IREE_CHECK_OK(
        module_b_create(instance_, iree_allocator_system(), &module_b));

    // Create the context with both modules and perform runtime linkage.
    // Imports from module_a -> module_b will be resolved and per-context state
    // will be allocated.
    std::vector<iree_vm_module_t*> modules = {module_a, module_b};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));

    // No longer need the modules as the context retains them.
    iree_vm_module_release(module_a);
    iree_vm_module_release(module_b);
  }

  virtual void TearDown() {
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  StatusOr<int32_t> RunFunction(iree_string_view_t function_name,
                                int32_t arg0) {
    // Lookup the entry function. This can be cached in an application if
    // multiple calls will be made.
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(
        iree_vm_context_resolve_function(
            context_, iree_make_cstring_view("module_b.entry"), &function),
        "unable to resolve entry point");

    // Setup I/O lists and pass in the argument. The result list will be
    // populated upon return.
    vm::ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             1, iree_allocator_system(),
                                             &input_list));
    auto arg0_value = iree_vm_value_make_i32(arg0);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_value(input_list.get(), &arg0_value));
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
    iree_vm_value_t ret0_value;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value(output_list.get(), 0, &ret0_value));
    return ret0_value.i32;
  }

 private:
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
};

TEST_F(VMNativeModuleTest, Example) {
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v0, RunFunction(iree_make_cstring_view("module_b.entry"), 1));
  ASSERT_EQ(v0, 1);
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v1, RunFunction(iree_make_cstring_view("module_b.entry"), 2));
  ASSERT_EQ(v1, 4);
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v2, RunFunction(iree_make_cstring_view("module_b.entry"), 3));
  ASSERT_EQ(v2, 8);
}

}  // namespace
}  // namespace iree
