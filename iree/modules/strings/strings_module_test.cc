// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/modules/strings/strings_module.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/hal/vmvx/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/strings/api.h"
#include "iree/modules/strings/api_detail.h"
#include "iree/modules/strings/strings_module_test_module_c.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

namespace iree {

namespace {

class StringsModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_hal_vmvx_driver_module_register(
        iree_hal_driver_registry_default()));
  }

  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    // Setup strings module:
    IREE_CHECK_OK(iree_strings_module_register_types());
    IREE_CHECK_OK(
        iree_strings_module_create(iree_allocator_system(), &strings_module_));

    // Setup hal module:
    IREE_CHECK_OK(iree_hal_module_register_types());
    iree_hal_driver_t* hal_driver = nullptr;
    IREE_CHECK_OK(iree_hal_driver_registry_try_create_by_name(
        iree_hal_driver_registry_default(), iree_make_cstring_view("vmvx"),
        iree_allocator_system(), &hal_driver));
    IREE_CHECK_OK(iree_hal_driver_create_default_device(
        hal_driver, iree_allocator_system(), &device_));
    IREE_CHECK_OK(
        iree_hal_module_create(device_, iree_allocator_system(), &hal_module_));
    iree_hal_driver_release(hal_driver);

    // Setup the test module.
    const auto* module_file_toc = iree_strings_module_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {strings_module_, hal_module_,
                                              bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), iree_allocator_system(),
        &context_));
  }

  virtual void TearDown() {
    iree_hal_device_release(device_);
    iree_vm_module_release(strings_module_);
    iree_vm_module_release(bytecode_module_);
    iree_vm_module_release(hal_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_vm_function_t LookupFunction(const char* function_name) {
    iree_vm_function_t function;
    IREE_CHECK_OK(bytecode_module_->lookup_function(
        bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));
    return function;
  }

  template <typename T, iree_hal_element_type_t E>
  void CreateBufferView(absl::Span<const T> contents,
                        absl::Span<const int32_t> shape,
                        iree_hal_buffer_view_t** out_buffer_view) {
    size_t num_elements = 1;
    for (int32_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    vm::ref<iree_hal_buffer_t> buffer;
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device_);
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator,
        static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        IREE_HAL_BUFFER_USAGE_ALL, contents.size() * sizeof(T), &buffer));
    IREE_ASSERT_OK(iree_hal_buffer_write_data(buffer.get(), 0, contents.data(),
                                              contents.size() * sizeof(T)));
    IREE_ASSERT_OK(iree_hal_buffer_view_create(
        buffer.get(), shape.data(), shape.size(), E, &*out_buffer_view));
  }

  void TestStringTensorToString(
      absl::Span<const iree_string_view_t> string_views,
      absl::Span<const int32_t> shape, const std::string& expected_output) {
    vm::ref<strings_string_tensor_t> input_string_tensor;
    IREE_ASSERT_OK(strings_string_tensor_create(
        iree_allocator_system(), string_views.data(), string_views.size(),
        shape.data(), shape.size(), &input_string_tensor));

    // Construct the input list for execution.
    vm::ref<iree_vm_list_t> inputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &inputs));

    // Add the string tensor to the input list.
    IREE_ASSERT_OK(
        iree_vm_list_push_ref_retain(inputs.get(), input_string_tensor));

    // Construct the output list for accepting results from the invocation.
    vm::ref<iree_vm_list_t> outputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &outputs));

    // Invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(context_,
                                  LookupFunction("string_tensor_to_string"),
                                  /*policy=*/nullptr, inputs.get(),
                                  outputs.get(), iree_allocator_system()));

    // Retrieve and validate the string tensor.
    auto* output_string =
        reinterpret_cast<strings_string_t*>(iree_vm_list_get_ref_deref(
            outputs.get(), 0, strings_string_get_descriptor()));
    ASSERT_EQ(output_string->value.size, expected_output.size());
    EXPECT_EQ(std::string(output_string->value.data, output_string->value.size),
              expected_output);
  }

  template <typename T, iree_hal_element_type_t E>
  void TestToStringTensor(absl::Span<const T> contents,
                          absl::Span<const int32_t> shape,
                          absl::Span<const iree_string_view_t> expected) {
    vm::ref<iree_hal_buffer_view_t> input_buffer_view;
    CreateBufferView<T, E>(contents, shape, &input_buffer_view);

    // Construct the input list for execution.
    vm::ref<iree_vm_list_t> inputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &inputs));

    // Add the buffer view to the input list.
    IREE_ASSERT_OK(
        iree_vm_list_push_ref_retain(inputs.get(), input_buffer_view));

    // Construct the output list for accepting results from the invocation.
    vm::ref<iree_vm_list_t> outputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &outputs));

    // Invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("to_string_tensor"),
                                  /*policy=*/nullptr, inputs.get(),
                                  outputs.get(), iree_allocator_system()));

    // Compare the output to the expected result.
    CompareResults(expected, shape, outputs);
  }

  void TestGather(absl::Span<const iree_string_view_t> dict,
                  absl::Span<const int32_t> dict_shape,
                  absl::Span<const int32_t> ids,
                  absl::Span<const int32_t> ids_shape,
                  absl::Span<const iree_string_view_t> expected) {
    vm::ref<strings_string_tensor_t> dict_string_tensor;
    IREE_ASSERT_OK(strings_string_tensor_create(
        iree_allocator_system(), dict.data(), dict.size(), dict_shape.data(),
        dict_shape.size(), &dict_string_tensor));

    // Construct the input list for execution.
    vm::ref<iree_vm_list_t> inputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 2,
                                       iree_allocator_system(), &inputs));

    // Add the dict to the input list.
    iree_vm_ref_t dict_string_tensor_ref =
        strings_string_tensor_move_ref(dict_string_tensor.get());
    IREE_ASSERT_OK(
        iree_vm_list_push_ref_retain(inputs.get(), &dict_string_tensor_ref));

    vm::ref<iree_hal_buffer_view_t> input_buffer_view;
    CreateBufferView<int32_t, IREE_HAL_ELEMENT_TYPE_SINT_32>(
        ids, ids_shape, &input_buffer_view);

    // Add the ids tensor to the input list.
    IREE_ASSERT_OK(
        iree_vm_list_push_ref_retain(inputs.get(), input_buffer_view));

    // Construct the output list for accepting results from the invocation.
    vm::ref<iree_vm_list_t> outputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &outputs));

    // Invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("gather"),
                                  /*policy=*/nullptr, inputs.get(),
                                  outputs.get(), iree_allocator_system()));

    // Compare the output to the expected result.
    CompareResults(expected, ids_shape, std::move(outputs));
  }

  void TestConcat(absl::Span<const iree_string_view_t> string_views,
                  absl::Span<const int32_t> shape,
                  absl::Span<const iree_string_view_t> expected) {
    vm::ref<strings_string_tensor_t> string_tensor;
    IREE_ASSERT_OK(strings_string_tensor_create(
        iree_allocator_system(), string_views.data(), string_views.size(),
        shape.data(), shape.size(), &string_tensor));

    // Construct the input list for execution.
    vm::ref<iree_vm_list_t> inputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &inputs));

    // Add the dict to the input list.
    IREE_ASSERT_OK(iree_vm_list_push_ref_retain(inputs.get(), string_tensor));

    // Construct the output list for accepting results from the invocation.
    vm::ref<iree_vm_list_t> outputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &outputs));

    // Invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("concat"),
                                  /*policy=*/nullptr, inputs.get(),
                                  outputs.get(), iree_allocator_system()));

    // Remove the last dimension from the shape to get the expected shape
    shape.remove_suffix(1);

    // Compare the output to the expected result.
    CompareResults(expected, shape, std::move(outputs));
  }

  void CompareResults(absl::Span<const iree_string_view_t> expected,
                      absl::Span<const int32_t> expected_shape,
                      vm::ref<iree_vm_list_t> outputs) {
    // Retrieve and validate the string tensor.
    auto* output_tensor =
        reinterpret_cast<strings_string_tensor_t*>(iree_vm_list_get_ref_deref(
            outputs.get(), 0, strings_string_tensor_get_descriptor()));

    // Validate the count.
    size_t count;
    IREE_ASSERT_OK(strings_string_tensor_get_count(output_tensor, &count));
    EXPECT_EQ(count, expected.size());

    // Validate the rank.
    int32_t rank;
    IREE_ASSERT_OK(strings_string_tensor_get_rank(output_tensor, &rank));
    ASSERT_EQ(rank, expected_shape.size());

    // Validate the shape.
    std::vector<int32_t> out_shape(rank);
    IREE_ASSERT_OK(
        strings_string_tensor_get_shape(output_tensor, out_shape.data(), rank));
    for (int i = 0; i < rank; i++) {
      EXPECT_EQ(out_shape[i], expected_shape[i])
          << "Dimension : " << i << " does not match";
    }

    // Fetch and validate string contents.
    std::vector<iree_string_view_t> out_strings(expected.size());
    IREE_ASSERT_OK(strings_string_tensor_get_elements(
        output_tensor, out_strings.data(), out_strings.size(), 0));
    for (iree_host_size_t i = 0; i < expected.size(); i++) {
      EXPECT_EQ(iree_string_view_compare(out_strings[i], expected[i]), 0)
          << "Expected: " << expected[i].data << " found "
          << out_strings[i].data;
    }
  }

  iree_hal_device_t* device_ = nullptr;
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* strings_module_ = nullptr;
  iree_vm_module_t* hal_module_ = nullptr;
};

TEST_F(StringsModuleTest, Prototype) {
  const int input = 42;
  std::string expected_output = "42\n";

  // Construct the input list for execution.
  vm::ref<iree_vm_list_t> inputs;
  IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                     iree_allocator_system(), &inputs));

  // Add the value parameter.
  iree_vm_value_t value = iree_vm_value_make_i32(input);
  IREE_ASSERT_OK(iree_vm_list_push_value(inputs.get(), &value));

  // Prepare outputs list to accept the results from the invocation.
  vm::ref<iree_vm_list_t> outputs;
  IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 0,
                                     iree_allocator_system(), &outputs));

  CaptureStdout();
  IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("print_example_func"),
                                /*policy=*/nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));
  EXPECT_EQ(GetCapturedStdout(), expected_output);
}

TEST_F(StringsModuleTest, StringTensorToString_Scalar) {
  // Expected output.
  std::string expected_output = "str";
  std::vector<iree_string_view_t> string_views = {
      iree_make_cstring_view("str")};
  TestStringTensorToString(string_views, {}, expected_output);
}

TEST_F(StringsModuleTest, StringTensorToString_Vector) {
  // Expected output.
  std::string expected_output = "[str1, str2]";
  std::vector<iree_string_view_t> string_views = {
      iree_make_cstring_view("str1"),
      iree_make_cstring_view("str2"),
  };
  TestStringTensorToString(string_views, {2}, expected_output);
}

TEST_F(StringsModuleTest, StringTensorToString_Tensor) {
  // Expected output.
  std::string expected_output = "[[[str1, str2]],\n[[str3, str4]]]";
  std::vector<iree_string_view_t> string_views = {
      iree_make_cstring_view("str1"), iree_make_cstring_view("str2"),
      iree_make_cstring_view("str3"), iree_make_cstring_view("str4")};
  std::vector<int32_t> shape = {2, 1, 2};
  TestStringTensorToString(string_views, shape, expected_output);
}

TEST_F(StringsModuleTest, ToString_Scalar) {
  std::vector<iree_string_view_t> expected{iree_make_cstring_view("14.000000")};
  std::vector<float> contents{14.0f};
  TestToStringTensor<float, IREE_HAL_ELEMENT_TYPE_FLOAT_32>(contents, {},
                                                            expected);
}

TEST_F(StringsModuleTest, ToString_Vector) {
  std::vector<iree_string_view_t> expected{iree_make_cstring_view("42.000000"),
                                           iree_make_cstring_view("43.000000")};

  std::vector<float> contents{42.0f, 43.0f};
  std::vector<int32_t> shape{2};
  TestToStringTensor<float, IREE_HAL_ELEMENT_TYPE_FLOAT_32>(contents, shape,
                                                            expected);
}

TEST_F(StringsModuleTest, ToString_Tensor) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("1.000000"), iree_make_cstring_view("2.000000"),
      iree_make_cstring_view("3.000000"), iree_make_cstring_view("4.000000")};

  std::vector<float> contents{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<float, IREE_HAL_ELEMENT_TYPE_FLOAT_32>(contents, shape,
                                                            expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Signed_Int8) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("-1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("127")};

  std::vector<int8_t> contents{-1, 2, 3, 127};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<int8_t, IREE_HAL_ELEMENT_TYPE_SINT_8>(contents, shape,
                                                           expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Unsigned_Int8) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("255")};

  std::vector<uint8_t> contents{1, 2, 3, 255};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<uint8_t, IREE_HAL_ELEMENT_TYPE_UINT_8>(contents, shape,
                                                            expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Signed_Int16) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("-1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("32700")};

  std::vector<int16_t> contents{-1, 2, 3, 32700};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<int16_t, IREE_HAL_ELEMENT_TYPE_SINT_16>(contents, shape,
                                                             expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Unsigned_Int16) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("65000")};

  std::vector<uint16_t> contents{1, 2, 3, 65000};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<uint16_t, IREE_HAL_ELEMENT_TYPE_UINT_16>(contents, shape,
                                                              expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Signed_Int32) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("-1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("2140000000")};

  std::vector<int32_t> contents{-1, 2, 3, 2140000000};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<int32_t, IREE_HAL_ELEMENT_TYPE_SINT_32>(contents, shape,
                                                             expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Unsigned_Int32) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("4290000000")};

  std::vector<uint32_t> contents{1, 2, 3, 4290000000};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<uint32_t, IREE_HAL_ELEMENT_TYPE_UINT_32>(contents, shape,
                                                              expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Signed_Int64) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("-1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("4300000000")};

  std::vector<int64_t> contents{-1, 2, 3, 4300000000};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<int64_t, IREE_HAL_ELEMENT_TYPE_SINT_64>(contents, shape,
                                                             expected);
}

TEST_F(StringsModuleTest, ToString_Tensor_Unsigned_Int64) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("1"), iree_make_cstring_view("2"),
      iree_make_cstring_view("3"), iree_make_cstring_view("4300000000")};

  std::vector<uint64_t> contents{1, 2, 3, 4300000000};
  std::vector<int32_t> shape{1, 2, 1, 1, 2};
  TestToStringTensor<uint64_t, IREE_HAL_ELEMENT_TYPE_UINT_64>(contents, shape,
                                                              expected);
}

TEST_F(StringsModuleTest, ToString_Vector_Float_64) {
  std::vector<iree_string_view_t> expected{iree_make_cstring_view("42.000000"),
                                           iree_make_cstring_view("43.000000")};

  std::vector<double> contents{42.0f, 43.0f};
  std::vector<int32_t> shape{2};
  TestToStringTensor<double, IREE_HAL_ELEMENT_TYPE_FLOAT_64>(contents, shape,
                                                             expected);
}

TEST_F(StringsModuleTest, GatherSingleElement) {
  std::vector<iree_string_view_t> expected{iree_make_cstring_view("World")};

  std::vector<iree_string_view_t> dict{iree_make_cstring_view("Hello"),
                                       iree_make_cstring_view("World"),
                                       iree_make_cstring_view("!")};
  std::vector<int32_t> dict_shape{3};

  std::vector<int32_t> ids{1};
  std::vector<int32_t> ids_shape{1};

  TestGather(dict, dict_shape, ids, ids_shape, expected);
}

TEST_F(StringsModuleTest, GatherMultipleElements) {
  std::vector<iree_string_view_t> expected{iree_make_cstring_view("World"),
                                           iree_make_cstring_view("Hello")};

  std::vector<iree_string_view_t> dict{iree_make_cstring_view("Hello"),
                                       iree_make_cstring_view("World"),
                                       iree_make_cstring_view("!")};
  std::vector<int32_t> dict_shape{3};

  std::vector<int32_t> ids{1, 0};
  std::vector<int32_t> ids_shape{2};

  TestGather(dict, dict_shape, ids, ids_shape, expected);
}

TEST_F(StringsModuleTest, GatherHigherRank) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("!"),     iree_make_cstring_view("!"),
      iree_make_cstring_view("Hello"), iree_make_cstring_view("World"),
      iree_make_cstring_view("!"),     iree_make_cstring_view("!")};

  std::vector<iree_string_view_t> dict{iree_make_cstring_view("Hello"),
                                       iree_make_cstring_view("World"),
                                       iree_make_cstring_view("!")};
  std::vector<int32_t> dict_shape{3};

  std::vector<int32_t> ids{2, 2, 0, 1, 2, 2};
  std::vector<int32_t> ids_shape{2, 3, 1, 1};

  TestGather(dict, dict_shape, ids, ids_shape, expected);
}

TEST_F(StringsModuleTest, Concat) {
  std::vector<iree_string_view_t> expected{iree_make_cstring_view("abc"),
                                           iree_make_cstring_view("def")};

  std::vector<iree_string_view_t> contents{
      iree_make_cstring_view("a"), iree_make_cstring_view("b"),
      iree_make_cstring_view("c"), iree_make_cstring_view("d"),
      iree_make_cstring_view("e"), iree_make_cstring_view("f")};
  std::vector<int32_t> shape{2, 3};

  TestConcat(contents, shape, expected);
}

TEST_F(StringsModuleTest, ConcatMultiDim) {
  std::vector<iree_string_view_t> expected{
      iree_make_cstring_view("abc"), iree_make_cstring_view("def"),
      iree_make_cstring_view("ghi"), iree_make_cstring_view("jkl")};

  std::vector<iree_string_view_t> contents{
      iree_make_cstring_view("a"), iree_make_cstring_view("b"),
      iree_make_cstring_view("c"), iree_make_cstring_view("d"),
      iree_make_cstring_view("e"), iree_make_cstring_view("f"),
      iree_make_cstring_view("g"), iree_make_cstring_view("h"),
      iree_make_cstring_view("i"), iree_make_cstring_view("j"),
      iree_make_cstring_view("k"), iree_make_cstring_view("l")};
  std::vector<int32_t> shape{2, 2, 3};

  TestConcat(contents, shape, expected);
}

TEST_F(StringsModuleTest, IdsToStrings) {
  std::vector<iree_string_view_t> intermediate_expected{
      iree_make_cstring_view("Hello"), iree_make_cstring_view("World"),
      iree_make_cstring_view("World"), iree_make_cstring_view("World"),
      iree_make_cstring_view("Hello"), iree_make_cstring_view("World"),
      iree_make_cstring_view("Hello"), iree_make_cstring_view("World"),
      iree_make_cstring_view("!"),     iree_make_cstring_view("!"),
      iree_make_cstring_view("World"), iree_make_cstring_view("!")};

  std::vector<iree_string_view_t> dict{iree_make_cstring_view("Hello"),
                                       iree_make_cstring_view("World"),
                                       iree_make_cstring_view("!")};
  std::vector<int32_t> dict_shape{3};

  std::vector<int32_t> ids{0, 1, 1, 1, 0, 1, 0, 1, 2, 2, 1, 2};
  std::vector<int32_t> ids_shape{1, 1, 4, 3};

  TestGather(dict, dict_shape, ids, ids_shape, intermediate_expected);

  std::vector<iree_string_view_t> final_expected{
      iree_make_cstring_view("HelloWorldWorld"),
      iree_make_cstring_view("WorldHelloWorld"),
      iree_make_cstring_view("HelloWorld!"), iree_make_cstring_view("!World!")};

  TestConcat(intermediate_expected, ids_shape, final_expected);
}

}  // namespace
}  // namespace iree
