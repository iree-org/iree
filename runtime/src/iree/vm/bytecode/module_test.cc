// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for bytecode_module.cc implementations.
// This means mostly just FlatBuffer verification, module interface functions,
// etc. bytecode_dispatch_test.cc covers actual dispatch.

#include "iree/vm/bytecode/module.h"

#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module_test_module_c.h"

static bool operator==(const iree_vm_value_t& lhs,
                       const iree_vm_value_t& rhs) noexcept {
  if (lhs.type != rhs.type) return false;
  switch (lhs.type) {
    default:
    case IREE_VM_VALUE_TYPE_NONE:
      return true;  // none == none
    case IREE_VM_VALUE_TYPE_I8:
      return lhs.i8 == rhs.i8;
    case IREE_VM_VALUE_TYPE_I16:
      return lhs.i16 == rhs.i16;
    case IREE_VM_VALUE_TYPE_I32:
      return lhs.i32 == rhs.i32;
    case IREE_VM_VALUE_TYPE_I64:
      return lhs.i64 == rhs.i64;
    case IREE_VM_VALUE_TYPE_F32:
      return lhs.f32 == rhs.f32;
    case IREE_VM_VALUE_TYPE_F64:
      return lhs.f64 == rhs.f64;
  }
}

static std::ostream& operator<<(std::ostream& os,
                                const iree_vm_value_t& value) {
  switch (value.type) {
    default:
    case IREE_VM_VALUE_TYPE_NONE:
      return os << "??";
    case IREE_VM_VALUE_TYPE_I8:
      return os << value.i8;
    case IREE_VM_VALUE_TYPE_I16:
      return os << value.i16;
    case IREE_VM_VALUE_TYPE_I32:
      return os << value.i32;
    case IREE_VM_VALUE_TYPE_I64:
      return os << value.i64;
    case IREE_VM_VALUE_TYPE_F32:
      return os << value.f32;
    case IREE_VM_VALUE_TYPE_F64:
      return os << value.f64;
  }
}

template <size_t N>
static std::vector<iree_vm_value_t> MakeValuesList(const int32_t (&values)[N]) {
  std::vector<iree_vm_value_t> result;
  result.resize(N);
  for (size_t i = 0; i < N; ++i) result[i] = iree_vm_value_make_i32(values[i]);
  return result;
}

static std::vector<iree_vm_value_t> MakeValueRangeList(int32_t start,
                                                       int32_t end) {
  std::vector<iree_vm_value_t> result;
  result.resize(abs(start - end) + 1);
  int32_t value = start;
  int32_t delta = start < end ? 1 : -1;
  for (size_t i = 0; i < result.size(); ++i, value += delta) {
    result[i] = iree_vm_value_make_i32(value);
  }
  return result;
}

static bool operator==(const iree_vm_ref_t& lhs,
                       const iree_vm_ref_t& rhs) noexcept {
  return lhs.type == rhs.type && lhs.ptr == rhs.ptr;
}

static std::ostream& operator<<(std::ostream& os, const iree_vm_ref_t& value) {
  // Just nulls today.
  return os << (iree_vm_ref_is_null(&value) ? "(null)" : "??");
}

static std::vector<iree_vm_ref_t> MakeNullRefList(size_t count) {
  std::vector<iree_vm_ref_t> result;
  result.resize(count);
  return result;
}

namespace {

using iree::StatusCode;
using iree::StatusOr;
using iree::testing::status::IsOkAndHolds;
using iree::testing::status::StatusIs;
using iree::vm::ref;
using testing::Eq;

class VMBytecodeModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    const auto* module_file_toc = iree_vm_bytecode_module_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  StatusOr<std::vector<iree_vm_value_t>> RunFunction(
      const char* function_name, std::vector<iree_vm_value_t> inputs) {
    ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(/*element_type=*/nullptr, inputs.size(),
                            iree_allocator_system(), &input_list));
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.get(), inputs.size()));
    for (iree_host_size_t i = 0; i < inputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_set_value(input_list.get(), i, &inputs[i]));
    }

    ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, 8, iree_allocator_system(), &output_list));

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));
    IREE_RETURN_IF_ERROR(
        iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/nullptr, input_list.get(), output_list.get(),
                       iree_allocator_system()));

    std::vector<iree_vm_value_t> outputs;
    outputs.resize(iree_vm_list_size(output_list.get()));
    for (iree_host_size_t i = 0; i < outputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_get_value(output_list.get(), i, &outputs[i]));
    }
    return outputs;
  }

  StatusOr<std::vector<iree_vm_ref_t>> RunFunction(
      const char* function_name, std::vector<iree_vm_ref_t> inputs) {
    ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(/*element_type=*/nullptr, inputs.size(),
                            iree_allocator_system(), &input_list));
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.get(), inputs.size()));
    for (iree_host_size_t i = 0; i < inputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_set_ref_retain(input_list.get(), i, &inputs[i]));
    }

    ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, 8, iree_allocator_system(), &output_list));

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));
    IREE_RETURN_IF_ERROR(
        iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/nullptr, input_list.get(), output_list.get(),
                       iree_allocator_system()));

    std::vector<iree_vm_ref_t> outputs;
    outputs.resize(iree_vm_list_size(output_list.get()));
    for (iree_host_size_t i = 0; i < outputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_get_ref_retain(output_list.get(), i, &outputs[i]));
    }
    return outputs;
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

TEST_F(VMBytecodeModuleTest, FuncIOEmpty) {
  EXPECT_THAT(RunFunction("FuncIOEmpty", std::vector<iree_vm_value_t>()),
              IsOkAndHolds(Eq(std::vector<iree_vm_value_t>())));
}

TEST_F(VMBytecodeModuleTest, FuncIO1) {
  EXPECT_THAT(RunFunction("FuncIO1", MakeValuesList({1})),
              IsOkAndHolds(Eq(MakeValuesList({1}))));
}

TEST_F(VMBytecodeModuleTest, FuncIO8) {
  EXPECT_THAT(RunFunction("FuncIO8", MakeValueRangeList(0, 7)),
              IsOkAndHolds(Eq(MakeValueRangeList(7, 0))));
}

TEST_F(VMBytecodeModuleTest, FuncIO600) {
  EXPECT_THAT(RunFunction("FuncIO600", MakeNullRefList(600)),
              IsOkAndHolds(Eq(MakeNullRefList(600))));
}

}  // namespace
