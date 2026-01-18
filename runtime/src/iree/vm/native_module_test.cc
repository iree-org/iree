// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/native_module_test.h"

#include <string>
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
  }

  virtual void TearDown() { iree_vm_instance_release(instance_); }

  iree_vm_context_t* CreateContext() {
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
    iree_vm_context_t* context = NULL;
    std::vector<iree_vm_module_t*> modules = {module_a, module_b};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context));

    // No longer need the modules as the context retains them.
    iree_vm_module_release(module_a);
    iree_vm_module_release(module_b);

    return context;
  }

  StatusOr<int32_t> RunFunction(iree_vm_context_t* context,
                                iree_string_view_t function_name,
                                int32_t arg0) {
    // Lookup the entry function. This can be cached in an application if
    // multiple calls will be made.
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(
        iree_vm_context_resolve_function(
            context, iree_make_cstring_view("module_b.entry"), &function),
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
        iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
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
};

TEST_F(VMNativeModuleTest, Example) {
  iree_vm_context_t* context = CreateContext();

  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v0,
      RunFunction(context, iree_make_cstring_view("module_b.entry"), 1));
  ASSERT_EQ(v0, 1);
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v1,
      RunFunction(context, iree_make_cstring_view("module_b.entry"), 2));
  ASSERT_EQ(v1, 4);
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t v2,
      RunFunction(context, iree_make_cstring_view("module_b.entry"), 3));
  ASSERT_EQ(v2, 8);

  iree_vm_context_release(context);
}

TEST_F(VMNativeModuleTest, Fork) {
  iree_vm_context_t* parent_context = CreateContext();

  // Run one tick of the parent context to mutate state.
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t parent_v0,
      RunFunction(parent_context, iree_make_cstring_view("module_b.entry"), 1));
  ASSERT_EQ(parent_v0, 1);

  // Fork the parent context, preserving its state.
  iree_vm_context_t* child_context = NULL;
  IREE_ASSERT_OK(iree_vm_context_fork(parent_context, iree_allocator_system(),
                                      &child_context));

  // Run a tick in both the parent and child contexts; the values should match
  // as they should have started from the same forked state but operate
  // independently.
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t parent_v1,
      RunFunction(parent_context, iree_make_cstring_view("module_b.entry"), 2));
  ASSERT_EQ(parent_v1, 4);
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t child_v1,
      RunFunction(child_context, iree_make_cstring_view("module_b.entry"), 2));
  ASSERT_EQ(child_v1, 4);

  // Drop the parent context; the forked one should be independent.
  iree_vm_context_release(parent_context);

  // Run another step in the child context to ensure it isn't relying on the
  // parent.
  IREE_ASSERT_OK_AND_ASSIGN(
      int32_t child_v2,
      RunFunction(child_context, iree_make_cstring_view("module_b.entry"), 3));
  ASSERT_EQ(child_v2, 8);

  iree_vm_context_release(child_context);
}

//===----------------------------------------------------------------------===//
// Parameterized alignment tests for C and C++ native modules
//===----------------------------------------------------------------------===//
// Tests alignment-sensitive parameter unpacking and result packing patterns.
// Both C (module_c) and C++ (module_cpp) implementations are tested to ensure
// consistent behavior across both APIs.

enum class ModuleImpl { kC, kCpp };

std::string ModuleImplName(const ::testing::TestParamInfo<ModuleImpl>& info) {
  return info.param == ModuleImpl::kC ? "C" : "Cpp";
}

class VMNativeModuleAlignmentTest
    : public ::testing::TestWithParam<ModuleImpl> {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));
  }

  virtual void TearDown() { iree_vm_instance_release(instance_); }

  // Returns the module name based on the test parameter.
  std::string module_name() const {
    return GetParam() == ModuleImpl::kC ? "module_c" : "module_cpp";
  }

  iree_vm_context_t* CreateAlignContext() {
    iree_vm_module_t* module = nullptr;
    if (GetParam() == ModuleImpl::kC) {
      IREE_CHECK_OK(
          module_c_align_create(instance_, iree_allocator_system(), &module));
    } else {
      IREE_CHECK_OK(
          module_cpp_create(instance_, iree_allocator_system(), &module));
    }

    iree_vm_context_t* context = NULL;
    std::vector<iree_vm_module_t*> modules = {module};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context));

    iree_vm_module_release(module);
    return context;
  }

  iree_vm_instance_t* instance_ = nullptr;
};

// Test: (i32) -> i32 - basic entry point.
TEST_P(VMNativeModuleAlignmentTest, Entry) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".entry";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(5);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // counter = 0 + (5 + 1) = 6, return 6 - 1 = 5
  EXPECT_EQ(result.i32, 5);

  iree_vm_context_release(context);
}

// Test: (i32, ref<buffer>) -> i32 - ref after i32 triggers alignment.
// Uses null ref to test alignment without buffer type registration.
TEST_P(VMNativeModuleAlignmentTest, MixedI32Ref) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".mixed_i32_ref";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(7);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  // Push null ref to test alignment without type registration complexity.
  iree_vm_ref_t null_ref = iree_vm_ref_null();
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &null_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // 7 + 0 (null buffer) = 7
  EXPECT_EQ(result.i32, 7);

  iree_vm_context_release(context);
}

// Test: (ref<buffer>, i32, ref<buffer>) -> i32 - ref/i32/ref pattern.
// Uses null refs to test alignment without buffer type registration.
TEST_P(VMNativeModuleAlignmentTest, MixedRefI32Ref) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".mixed_ref_i32_ref";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &inputs));
  iree_vm_ref_t null_ref1 = iree_vm_ref_null();
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &null_ref1));
  auto arg1 = iree_vm_value_make_i32(3);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  iree_vm_ref_t null_ref2 = iree_vm_ref_null();
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &null_ref2));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // 0 + 3 + 0 = 3
  EXPECT_EQ(result.i32, 3);

  iree_vm_context_release(context);
}

// Test: (i32, i64) -> i64 - i64 after i32 triggers alignment.
TEST_P(VMNativeModuleAlignmentTest, MixedI32I64) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".mixed_i32_i64";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(100);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i64(0x100000000LL);  // 4 billion
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // 100 + 0x100000000 = 0x100000064
  EXPECT_EQ(result.i64, 0x100000064LL);

  iree_vm_context_release(context);
}

// Test: (i32, i32, i32, ref<buffer>) -> i32 - ref after 12 bytes of i32s.
// Uses null ref to test alignment without buffer type registration.
TEST_P(VMNativeModuleAlignmentTest, MixedI32x3Ref) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".mixed_i32x3_ref";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 4,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(1);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i32(2);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  auto arg2 = iree_vm_value_make_i32(3);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg2));
  iree_vm_ref_t null_ref = iree_vm_ref_null();
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &null_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // 1 + 2 + 3 + 0 = 6
  EXPECT_EQ(result.i32, 6);

  iree_vm_context_release(context);
}

// Test: (i64, i32) -> i32 - i32 after i64.
TEST_P(VMNativeModuleAlignmentTest, MixedI64I32) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".mixed_i64_i32";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i64(42);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i32(8);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // 42 + 8 = 50
  EXPECT_EQ(result.i32, 50);

  iree_vm_context_release(context);
}

// Test: (i32, i64, i32) -> i64 - i64 sandwiched between i32s.
TEST_P(VMNativeModuleAlignmentTest, MixedI32I64I32) {
  iree_vm_context_t* context = CreateAlignContext();

  std::string func_name = module_name() + ".mixed_i32_i64_i32";
  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_string_view(func_name.data(), func_name.size()),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(10);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i64(1000);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  auto arg2 = iree_vm_value_make_i32(20);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg2));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  // 10 + 1000 + 20 = 1030
  EXPECT_EQ(result.i64, 1030);

  iree_vm_context_release(context);
}

INSTANTIATE_TEST_SUITE_P(ModuleImpls, VMNativeModuleAlignmentTest,
                         ::testing::Values(ModuleImpl::kC, ModuleImpl::kCpp),
                         ModuleImplName);

//===----------------------------------------------------------------------===//
// Borrowed pointer (T*) alignment tests - C++ only
//===----------------------------------------------------------------------===//
// These tests specifically exercise ParamUnpack<T*> which had a missing
// alignment bug. They use actual non-null buffers to verify the data is
// correctly read after alignment.

class VMNativeModuleBorrowedPtrTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));
  }

  virtual void TearDown() { iree_vm_instance_release(instance_); }

  iree_vm_context_t* CreateCppContext() {
    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(
        module_cpp_create(instance_, iree_allocator_system(), &module));

    iree_vm_context_t* context = NULL;
    std::vector<iree_vm_module_t*> modules = {module};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context));

    iree_vm_module_release(module);
    return context;
  }

  // Creates a buffer with the given length filled with zeros.
  vm::ref<iree_vm_buffer_t> CreateBuffer(iree_host_size_t length) {
    vm::ref<iree_vm_buffer_t> buffer;
    IREE_CHECK_OK(iree_vm_buffer_create(
        IREE_VM_BUFFER_ACCESS_MUTABLE | IREE_VM_BUFFER_ACCESS_ORIGIN_HOST,
        length, /*alignment=*/0, iree_allocator_system(), &buffer));
    return buffer;
  }

  iree_vm_instance_t* instance_ = nullptr;
};

// Test borrowed ref after 4 bytes (i32) - needs alignment.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32Ref_NullBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32_ref"),
      &function));

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(42);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  iree_vm_ref_t null_ref = iree_vm_ref_null();
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &null_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 42);  // 42 + 0 (null buffer)

  iree_vm_context_release(context);
}

TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(10);  // 10 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(42);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 52);  // 42 + 10 (buffer length)

  iree_vm_context_release(context);
}

// Test borrowed ref after 8 bytes (2x i32) - already aligned.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32x2Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32x2_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(5);  // 5 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(10);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i32(20);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 35);  // 10 + 20 + 5

  iree_vm_context_release(context);
}

// Test borrowed ref after 12 bytes (3x i32) - THIS IS THE BUG CASE!
// The ref needs 8-byte alignment, but 12 bytes is not aligned.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32x3Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32x3_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(7);  // 7 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 4,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(1);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i32(2);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  auto arg2 = iree_vm_value_make_i32(3);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg2));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 13);  // 1 + 2 + 3 + 7

  iree_vm_context_release(context);
}

// Test borrowed ref after 16 bytes (4x i32) - already aligned.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32x4Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32x4_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(8);  // 8 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 5,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(1);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i32(2);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  auto arg2 = iree_vm_value_make_i32(3);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg2));
  auto arg3 = iree_vm_value_make_i32(4);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg3));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 18);  // 1 + 2 + 3 + 4 + 8

  iree_vm_context_release(context);
}

// Test borrowed ref after 20 bytes (5x i32) - needs alignment.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32x5Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32x5_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(11);  // 11 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 6,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(1);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i32(2);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  auto arg2 = iree_vm_value_make_i32(3);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg2));
  auto arg3 = iree_vm_value_make_i32(4);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg3));
  auto arg4 = iree_vm_value_make_i32(5);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg4));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 26);  // 1 + 2 + 3 + 4 + 5 + 11

  iree_vm_context_release(context);
}

// Test borrowed ref/i32/ref pattern with actual buffers.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedRefI32Ref_WithBuffers) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_ref_i32_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer1 = CreateBuffer(3);   // 3 bytes
  vm::ref<iree_vm_buffer_t> buffer2 = CreateBuffer(13);  // 13 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &inputs));
  iree_vm_ref_t buffer1_ref = iree_vm_buffer_retain_ref(buffer1.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer1_ref));
  auto arg1 = iree_vm_value_make_i32(100);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  iree_vm_ref_t buffer2_ref = iree_vm_buffer_retain_ref(buffer2.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer2_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 116);  // 3 + 100 + 13

  iree_vm_context_release(context);
}

// Test two consecutive borrowed refs.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedRefRef_WithBuffers) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_ref_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer1 = CreateBuffer(17);  // 17 bytes
  vm::ref<iree_vm_buffer_t> buffer2 = CreateBuffer(23);  // 23 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  iree_vm_ref_t buffer1_ref = iree_vm_buffer_retain_ref(buffer1.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer1_ref));
  iree_vm_ref_t buffer2_ref = iree_vm_buffer_retain_ref(buffer2.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer2_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 40);  // 17 + 23

  iree_vm_context_release(context);
}

// Test borrowed ref after i64 (already aligned).
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI64Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i64_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(6);  // 6 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i64(50);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 56);  // 50 + 6

  iree_vm_context_release(context);
}

// Test borrowed ref after i32+i64 (tests complex alignment).
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedI32I64Ref_WithBuffer) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_i32_i64_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer = CreateBuffer(4);  // 4 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &inputs));
  auto arg0 = iree_vm_value_make_i32(10);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg0));
  auto arg1 = iree_vm_value_make_i64(200);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  iree_vm_ref_t buffer_ref = iree_vm_buffer_retain_ref(buffer.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 214);  // 10 + 200 + 4

  iree_vm_context_release(context);
}

// Test i64 between two borrowed refs.
TEST_F(VMNativeModuleBorrowedPtrTest, BorrowedRefI64Ref_WithBuffers) {
  iree_vm_context_t* context = CreateCppContext();

  iree_vm_function_t function;
  IREE_ASSERT_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view("module_cpp.borrowed_ref_i64_ref"),
      &function));

  vm::ref<iree_vm_buffer_t> buffer1 = CreateBuffer(8);   // 8 bytes
  vm::ref<iree_vm_buffer_t> buffer2 = CreateBuffer(12);  // 12 bytes

  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                                     iree_allocator_system(), &inputs));
  iree_vm_ref_t buffer1_ref = iree_vm_buffer_retain_ref(buffer1.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer1_ref));
  auto arg1 = iree_vm_value_make_i64(500);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &arg1));
  iree_vm_ref_t buffer2_ref = iree_vm_buffer_retain_ref(buffer2.get());
  IREE_ASSERT_OK(iree_vm_list_push_ref_move(inputs.get(), &buffer2_ref));

  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                     iree_allocator_system(), &outputs));

  IREE_ASSERT_OK(iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                                nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  iree_vm_value_t result;
  IREE_ASSERT_OK(iree_vm_list_get_value(outputs.get(), 0, &result));
  EXPECT_EQ(result.i32, 520);  // 8 + 500 + 12

  iree_vm_context_release(context);
}

}  // namespace
}  // namespace iree
