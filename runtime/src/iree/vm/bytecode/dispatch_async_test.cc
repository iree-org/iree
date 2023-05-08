// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests covering the dispatch logic for individual ops.
//
// iree/vm/test/async_ops.mlir contains the functions used here for testing. We
// avoid defining the IR inline here so that we can run this test on platforms
// that we can't run the full MLIR compiler stack on.

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// Compiled module embedded here to avoid file IO:
#include "iree/vm/test/async_bytecode_modules.h"

namespace iree {
namespace {

using iree::testing::status::StatusIs;

class VMBytecodeDispatchAsyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_TRACE_SCOPE();
    const iree_file_toc_t* file = async_bytecode_modules_c_create();

    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_,
        iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(file->data),
                               file->size},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

// Tests a simple straight-line yield sequence that requires 3 resumes.
// See iree/vm/test/async_ops.mlir > @yield_sequence
TEST_F(VMBytecodeDispatchAsyncTest, YieldSequence) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("yield_sequence"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 97;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // 0/3
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // 1/3
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // 2/3
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // 3/3
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  ASSERT_EQ(ret_value, arg_value + 3);

  iree_vm_stack_deinitialize(stack);
}

// Tests a yield with data-dependent control, ensuring that we run the
// alternating branches and pass along branch args on resume.
// See iree/vm/test/async_ops.mlir > @yield_divergent
TEST_F(VMBytecodeDispatchAsyncTest, YieldDivergent) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("yield_divergent"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  // result = %arg0 ? %arg1 : %arg2
  struct {
    uint32_t arg0;
    uint32_t arg1;
    uint32_t arg2;
  } arg_values = {
      0,
      100,
      200,
  };
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_values, sizeof(arg_values));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // arg0=0: result = %arg0 ? %arg1 : %arg2 => %arg2
  arg_values.arg0 = 0;
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));
  ASSERT_EQ(ret_value, arg_values.arg2);

  // arg0=1: result = %arg0 ? %arg1 : %arg2 => %arg1
  arg_values.arg0 = 1;
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));
  ASSERT_EQ(ret_value, arg_values.arg1);

  iree_vm_stack_deinitialize(stack);
}

}  // namespace
}  // namespace iree
