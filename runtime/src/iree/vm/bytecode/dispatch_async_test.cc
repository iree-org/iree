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

// Native test module for yieldable imports.
#include "iree/vm/test/async_ops_test_module.h"

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

    // Create native yieldable_test module (required by async_ops imports).
    IREE_CHECK_OK(yieldable_test_module_create(
        instance_, iree_allocator_system(), &native_module_));

    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(file->data),
                               static_cast<iree_host_size_t>(file->size)},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    // Native module first for import resolution.
    std::vector<iree_vm_module_t*> modules = {native_module_, bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_vm_module_release(bytecode_module_);
    iree_vm_module_release(native_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
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

//===----------------------------------------------------------------------===//
// CallYieldable tests
//===----------------------------------------------------------------------===//

class VMBytecodeDispatchCallYieldableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_TRACE_SCOPE();
    const iree_file_toc_t* file = async_bytecode_modules_c_create();

    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    // Create native yieldable_test module (required by async_ops imports).
    IREE_CHECK_OK(yieldable_test_module_create(
        instance_, iree_allocator_system(), &native_module_));

    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(file->data),
                               static_cast<iree_host_size_t>(file->size)},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    // Native module first for import resolution.
    std::vector<iree_vm_module_t*> modules = {native_module_, bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_vm_module_release(bytecode_module_);
    iree_vm_module_release(native_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

// Tests calling an internal function that yields 4 times via vm.call.yieldable.
// See iree/vm/test/call_yieldable_ops.mlir > @call_yieldable_internal
TEST_F(VMBytecodeDispatchCallYieldableTest, CallYieldableInternal) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_internal"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(nullptr, 0);  // No arguments
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // The callee yields 3 times, so we need 3 resumes.
  // begin -> 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 3rd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be 4 (0 + 4 increments across 4 basic blocks)
  ASSERT_EQ(ret_value, 4u);

  iree_vm_stack_deinitialize(stack);
}

// Tests calling an internal yieldable function with an argument.
// See iree/vm/test/call_yieldable_ops.mlir > @call_yieldable_with_arg
TEST_F(VMBytecodeDispatchCallYieldableTest, CallYieldableWithArg) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_with_arg"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 42;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // The callee yields 1 time.
  // 0/1
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // 1/1
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be arg_value + 1
  ASSERT_EQ(ret_value, arg_value + 1);

  iree_vm_stack_deinitialize(stack);
}

//===----------------------------------------------------------------------===//
// CallYieldable to Imports tests
//===----------------------------------------------------------------------===//
// Tests vm.call.yieldable calling native module functions that yield.

class VMBytecodeDispatchCallYieldableImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_TRACE_SCOPE();
    const iree_file_toc_t* file = async_bytecode_modules_c_create();

    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    // Create native yieldable_test module.
    IREE_CHECK_OK(yieldable_test_module_create(
        instance_, iree_allocator_system(), &native_module_));

    // Create bytecode module that imports from native module.
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(file->data),
                               static_cast<iree_host_size_t>(file->size)},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    // Create context with both modules (native first for import resolution).
    std::vector<iree_vm_module_t*> modules = {native_module_, bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  void TearDown() override {
    IREE_TRACE_SCOPE();
    iree_vm_module_release(bytecode_module_);
    iree_vm_module_release(native_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

// Tests calling a yieldable import that yields 3 times.
// This exercises Bug 1 fix: PC must be saved at instruction start, not after
// decode.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, YieldableImportYields3) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_import_yields_3"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 100;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // The import yields 3 times.
  // begin -> 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 3rd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be arg + 3
  ASSERT_EQ(ret_value, arg_value + 3);

  iree_vm_stack_deinitialize(stack);
}

// Tests calling a yieldable import that yields 0 times (synchronous).
TEST_F(VMBytecodeDispatchCallYieldableImportTest, YieldableImportYields0) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_import_yields_0"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 42;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // No yields, should complete immediately.
  IREE_ASSERT_OK(
      function.module->begin_call(function.module->self, stack, call));

  // Result should be arg + 0
  ASSERT_EQ(ret_value, arg_value);

  iree_vm_stack_deinitialize(stack);
}

// Tests calling a yieldable import after an internal function call.
// This exercises Bug 2 fix: return_registers must be cleared after internal
// call.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, YieldableAfterInternal) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_after_internal"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 5;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // The function:
  // 1. Calls internal_add_10(arg) -> arg + 10
  // 2. Calls yieldable import yield_n(arg+10, 2) which yields 2 times
  // Expected: 2 yields, result = (arg + 10) + 2

  // begin -> internal call completes, import yields 1st time -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be (arg + 10) + 2 = arg + 12
  ASSERT_EQ(ret_value, arg_value + 12);

  iree_vm_stack_deinitialize(stack);
}

// Tests two sequential yieldable import calls in the same function.
// This catches bugs where the second call sees stale state from the first.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, YieldableImportSequential) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_import_sequential"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 10;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // First import yields 2 times, second import yields 3 times = 5 total yields.
  // begin -> 1st import, 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 1st import, 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 1st import done, 2nd import, 1st yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 2nd import, 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 2nd import, 3rd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 2nd import done, return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be (arg + 2) + 3 = arg + 5
  ASSERT_EQ(ret_value, arg_value + 5);

  iree_vm_stack_deinitialize(stack);
}

// Tests a yieldable import nested inside an internal yieldable function.
// This is the most complex frame stack scenario.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, YieldableImportNested) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_nested_yieldable"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 50;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // Sequence: 1 yield (internal) + 2 yields (import) + 1 yield (internal) = 4
  // begin -> internal 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> import 1st yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> import 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> import done, internal 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> internal done, return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be ((arg + 1) + 2) + 1 = arg + 4
  ASSERT_EQ(ret_value, arg_value + 4);

  iree_vm_stack_deinitialize(stack);
}

// Tests a yieldable import with many yields to catch state accumulation bugs.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, YieldableImportStress) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_yieldable_import_stress"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 1000;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // 10 yields total.
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  for (int i = 1; i < 10; ++i) {
    ASSERT_THAT(function.module->resume_call(function.module->self, stack,
                                             call.results),
                StatusIs(StatusCode::kDeferred))
        << "Expected DEFERRED at resume " << i;
  }

  // Final resume should complete.
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be arg + 10
  ASSERT_EQ(ret_value, arg_value + 10);

  iree_vm_stack_deinitialize(stack);
}

//===----------------------------------------------------------------------===//
// CallVariadicYieldable to Imports tests
//===----------------------------------------------------------------------===//
// Tests vm.call.variadic.yieldable calling native module functions that yield.

// Tests calling a variadic yieldable import with 2 args and 3 yields.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, VariadicYieldable2Args) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_variadic_yieldable_2args"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  struct {
    uint32_t arg0;
    uint32_t arg1;
  } arg_values = {10, 20};
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_values, sizeof(arg_values));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // The import sums the variadic args and yields 3 times.
  // begin -> 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 3rd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be (arg0 + arg1) + 3 = 10 + 20 + 3 = 33
  ASSERT_EQ(ret_value, arg_values.arg0 + arg_values.arg1 + 3);

  iree_vm_stack_deinitialize(stack);
}

// Tests calling a variadic yieldable import with 0 yields (synchronous).
TEST_F(VMBytecodeDispatchCallYieldableImportTest, VariadicYieldable0Yields) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_variadic_yieldable_0yields"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  struct {
    uint32_t arg0;
    uint32_t arg1;
    uint32_t arg2;
  } arg_values = {5, 10, 15};
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_values, sizeof(arg_values));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // No yields, should complete immediately.
  IREE_ASSERT_OK(
      function.module->begin_call(function.module->self, stack, call));

  // Result should be arg0 + arg1 + arg2 = 5 + 10 + 15 = 30
  ASSERT_EQ(ret_value, arg_values.arg0 + arg_values.arg1 + arg_values.arg2);

  iree_vm_stack_deinitialize(stack);
}

// Tests calling a variadic yieldable import with 1 arg.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, VariadicYieldable1Arg) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_variadic_yieldable_1arg"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t arg_value = 100;
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_value, sizeof(arg_value));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // 2 yields.
  // begin -> 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be arg0 + 2 = 100 + 2 = 102
  ASSERT_EQ(ret_value, arg_value + 2);

  iree_vm_stack_deinitialize(stack);
}

// Tests calling a variadic yieldable import with empty variadic list.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, VariadicYieldableEmpty) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_variadic_yieldable_empty"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(nullptr, 0);  // No arguments
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // 1 yield.
  // begin -> 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be 0 + 1 = 1
  ASSERT_EQ(ret_value, 1u);

  iree_vm_stack_deinitialize(stack);
}

// Tests two sequential variadic yieldable calls.
TEST_F(VMBytecodeDispatchCallYieldableImportTest, VariadicYieldableSequential) {
  IREE_TRACE_SCOPE();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      IREE_SV("call_variadic_yieldable_sequential"), &function));
  IREE_VM_INLINE_STACK_INITIALIZE(stack, IREE_VM_CONTEXT_FLAG_NONE,
                                  iree_vm_context_state_resolver(context_),
                                  iree_allocator_system());

  struct {
    uint32_t arg0;
    uint32_t arg1;
    uint32_t arg2;
  } arg_values = {10, 20, 5};
  uint32_t ret_value = 0;

  iree_vm_function_call_t call;
  memset(&call, 0, sizeof(call));
  call.function = function;
  call.arguments = iree_make_byte_span(&arg_values, sizeof(arg_values));
  call.results = iree_make_byte_span(&ret_value, sizeof(ret_value));

  // First variadic: 2 yields, second variadic: 1 yield = 3 yields total.
  // begin -> 1st call, 1st yield -> DEFERRED
  ASSERT_THAT(function.module->begin_call(function.module->self, stack, call),
              StatusIs(StatusCode::kDeferred));

  // resume -> 1st call, 2nd yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 1st call done, 2nd call, 1st yield -> DEFERRED
  ASSERT_THAT(
      function.module->resume_call(function.module->self, stack, call.results),
      StatusIs(StatusCode::kDeferred));

  // resume -> 2nd call done, return -> OK
  IREE_ASSERT_OK(
      function.module->resume_call(function.module->self, stack, call.results));

  // Result should be:
  // First call: sum(arg0, arg1) + 2 yields = (10 + 20) + 2 = 32
  // Second call: sum(32, arg2) + 1 yield = (32 + 5) + 1 = 38
  ASSERT_EQ(ret_value, 38u);

  iree_vm_stack_deinitialize(stack);
}

}  // namespace
}  // namespace iree
