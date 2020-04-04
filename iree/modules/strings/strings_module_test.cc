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

#include "iree/modules/strings/strings_module.h"

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/strings/api.h"
#include "iree/modules/strings/api_detail.h"
#include "iree/modules/strings/strings_module_test_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/ref_cc.h"
#include "iree/vm/stack.h"
#include "iree/vm/types.h"

using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

namespace iree {

namespace {

class StringsModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_));

    // Setup strings module:
    IREE_CHECK_OK(iree_strings_module_register_types());
    IREE_CHECK_OK(
        iree_strings_module_create(IREE_ALLOCATOR_SYSTEM, &strings_module_))
        << "Strings module failed to init";

    // Setup hal module:
    IREE_CHECK_OK(iree_hal_module_register_types());
    iree_hal_driver_t* hal_driver = nullptr;
    IREE_CHECK_OK(iree_hal_driver_registry_create_driver(
        iree_make_cstring_view("vmla"), IREE_ALLOCATOR_SYSTEM, &hal_driver));
    IREE_CHECK_OK(iree_hal_driver_create_default_device(
        hal_driver, IREE_ALLOCATOR_SYSTEM, &device_));
    IREE_CHECK_OK(
        iree_hal_module_create(device_, IREE_ALLOCATOR_SYSTEM, &hal_module_));
    iree_hal_driver_release(hal_driver);

    // Setup the test module.
    const auto* module_file_toc =
        iree::strings_module_test::strings_module_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &bytecode_module_))
        << "Bytecode module failed to load";

    std::vector<iree_vm_module_t*> modules = {strings_module_, hal_module_,
                                              bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), IREE_ALLOCATOR_SYSTEM,
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

  iree_vm_function_t LookupFunction(absl::string_view function_name) {
    iree_vm_function_t function;
    IREE_CHECK_OK(bytecode_module_->lookup_function(
        bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{function_name.data(), function_name.size()},
        &function))
        << "Exported function '" << function_name << "' not found";
    return function;
  }

  void CreateBufferView(absl::Span<const float> contents,
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
        IREE_HAL_BUFFER_USAGE_ALL, contents.size() * sizeof(float), &buffer));
    iree_hal_mapped_memory_t mapped_memory;
    IREE_ASSERT_OK(iree_hal_buffer_map(buffer.get(),
                                       IREE_HAL_MEMORY_ACCESS_WRITE, 0,
                                       IREE_WHOLE_BUFFER, &mapped_memory));
    memcpy(mapped_memory.contents.data,
           static_cast<const void*>(contents.data()),
           mapped_memory.contents.data_length);
    IREE_ASSERT_OK(iree_hal_buffer_unmap(buffer.get(), &mapped_memory));
    IREE_ASSERT_OK(
        iree_hal_buffer_view_create(buffer.get(), shape.data(), shape.size(),
                                    IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                                    IREE_ALLOCATOR_SYSTEM, &*out_buffer_view));
  }

  void TestStringTensorToString(
      absl::Span<const iree_string_view_t> string_views,
      absl::Span<const int32_t> shape, absl::string_view expected_output) {
    vm::ref<strings_string_tensor_t> input_string_tensor;
    IREE_ASSERT_OK(strings_string_tensor_create(
        IREE_ALLOCATOR_SYSTEM, string_views.data(), string_views.size(),
        shape.data(), shape.size(), &input_string_tensor));

    // Construct the input list for execution.
    iree_vm_variant_list_t* inputs = nullptr;
    IREE_ASSERT_OK(
        iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &inputs));

    // Add the string tensor to the input list.
    iree_vm_ref_t input_string_tensor_ref =
        strings_string_tensor_move_ref(input_string_tensor.get());
    IREE_ASSERT_OK(iree_vm_variant_list_append_ref_retain(
        inputs, &input_string_tensor_ref));

    // Construct the output list for accepting results from the invocation.
    iree_vm_variant_list_t* outputs = nullptr;
    IREE_ASSERT_OK(
        iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &outputs));

    // Invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(
        context_, LookupFunction("string_tensor_to_string"),
        /*policy=*/nullptr, inputs, outputs, IREE_ALLOCATOR_SYSTEM));

    // Retrieve and validate the string tensor;
    strings_string_t* output_string =
        strings_string_deref(&iree_vm_variant_list_get(outputs, 0)->ref);
    ASSERT_EQ(output_string->value.size, expected_output.length());
    EXPECT_EQ(
        absl::string_view(output_string->value.data, output_string->value.size),
        expected_output);

    // Free the lists.
    iree_vm_variant_list_free(inputs);
    iree_vm_variant_list_free(outputs);
  }

  void TestToStringTensor(absl::Span<const float> contents,
                          absl::Span<const int32_t> shape,
                          absl::Span<const iree_string_view_t> expected) {
    vm::ref<iree_hal_buffer_view_t> input_buffer_view;
    CreateBufferView(contents, shape, &input_buffer_view);

    // Construct the input list for execution.
    iree_vm_variant_list_t* inputs = nullptr;
    IREE_ASSERT_OK(
        iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &inputs));

    // Add the buffer view to the input list.
    iree_vm_ref_t input_buffer_view_ref =
        iree_hal_buffer_view_move_ref(input_buffer_view.get());

    IREE_ASSERT_OK(
        iree_vm_variant_list_append_ref_retain(inputs, &input_buffer_view_ref));

    // Construct the output list for accepting results from the invocation.
    iree_vm_variant_list_t* outputs = nullptr;
    IREE_ASSERT_OK(
        iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &outputs));

    // Invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("to_string_tensor"),
                                  /*policy=*/nullptr, inputs, outputs,
                                  IREE_ALLOCATOR_SYSTEM));

    // Retrieve and validate the string tensor;
    strings_string_tensor_t* output_tensor =
        strings_string_tensor_deref(&iree_vm_variant_list_get(outputs, 0)->ref);

    // Validate the count.
    size_t count;
    IREE_ASSERT_OK(strings_string_tensor_get_count(output_tensor, &count));
    EXPECT_EQ(count, contents.size());

    // Validate the rank.
    int32_t rank;
    IREE_ASSERT_OK(strings_string_tensor_get_rank(output_tensor, &rank));
    ASSERT_EQ(rank, shape.size());

    // Validate the shape.
    std::vector<int32_t> out_shape(rank);
    IREE_ASSERT_OK(
        strings_string_tensor_get_shape(output_tensor, out_shape.data(), rank));
    for (int i = 0; i < rank; i++) {
      EXPECT_EQ(out_shape[i], shape[i])
          << "Dimension : " << i << " does not match";
    }

    // Fetch and validate string contents.
    std::vector<iree_string_view_t> out_strings(expected.size());
    IREE_ASSERT_OK(strings_string_tensor_get_elements(
        output_tensor, out_strings.data(), out_strings.size(), 0));
    for (int i = 0; i < expected.size(); i++) {
      EXPECT_EQ(iree_string_view_compare(out_strings[i], expected[i]), 0)
          << "Expected: " << expected[i].data << " found "
          << out_strings[i].data;
    }

    // Free the lists.
    iree_vm_variant_list_free(inputs);
    iree_vm_variant_list_free(outputs);
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
  iree_vm_variant_list_t* inputs = nullptr;
  IREE_ASSERT_OK(iree_vm_variant_list_alloc(1, IREE_ALLOCATOR_SYSTEM, &inputs));

  // Add the value parameter.
  iree_vm_value_t value = IREE_VM_VALUE_MAKE_I32(input);
  IREE_ASSERT_OK(iree_vm_variant_list_append_value(inputs, value));

  // Prepare outputs list to accept the results from the invocation.
  iree_vm_variant_list_t* outputs = nullptr;
  IREE_ASSERT_OK(
      iree_vm_variant_list_alloc(0, IREE_ALLOCATOR_SYSTEM, &outputs));

  CaptureStdout();
  IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction("print_example_func"),
                                /*policy=*/nullptr, inputs, outputs,
                                IREE_ALLOCATOR_SYSTEM));
  EXPECT_EQ(GetCapturedStdout(), expected_output);

  iree_vm_variant_list_free(inputs);
  iree_vm_variant_list_free(outputs);
}

TEST_F(StringsModuleTest, StringTensorToString_Scalar) {
  // Expected output.
  std::string expected_output = "str";
  absl::InlinedVector<iree_string_view_t, 1> string_views = {
      iree_make_cstring_view("str")};
  TestStringTensorToString(string_views, {}, expected_output);
}

TEST_F(StringsModuleTest, StringTensorToString_Vector) {
  // Expected output.
  std::string expected_output = "[str1, str2]";
  absl::InlinedVector<iree_string_view_t, 1> string_views = {
      iree_make_cstring_view("str1"),
      iree_make_cstring_view("str2"),
  };
  TestStringTensorToString(string_views, {2}, expected_output);
}

TEST_F(StringsModuleTest, StringTensorToString_Tensor) {
  // Expected output.
  std::string expected_output = "[[[str1, str2]],\n[[str3, str4]]]";
  absl::InlinedVector<iree_string_view_t, 4> string_views = {
      iree_make_cstring_view("str1"), iree_make_cstring_view("str2"),
      iree_make_cstring_view("str3"), iree_make_cstring_view("str4")};
  absl::InlinedVector<int32_t, 4> shape = {2, 1, 2};
  TestStringTensorToString(string_views, shape, expected_output);
}

TEST_F(StringsModuleTest, ToString_Scalar) {
  absl::InlinedVector<iree_string_view_t, 1> expected{
      iree_make_cstring_view("14.000000")};

  absl::InlinedVector<float, 1> contents{14.0f};
  TestToStringTensor(contents, {}, expected);
}

TEST_F(StringsModuleTest, ToString_Vector) {
  absl::InlinedVector<iree_string_view_t, 2> expected{
      iree_make_cstring_view("42.000000"), iree_make_cstring_view("43.000000")};

  absl::InlinedVector<float, 2> contents{42.0f, 43.0f};
  absl::InlinedVector<int32_t, 4> shape{2};
  TestToStringTensor(contents, shape, expected);
}

TEST_F(StringsModuleTest, ToString_Tensor) {
  absl::InlinedVector<iree_string_view_t, 4> expected{
      iree_make_cstring_view("1.000000"), iree_make_cstring_view("2.000000"),
      iree_make_cstring_view("3.000000"), iree_make_cstring_view("4.000000")};

  absl::InlinedVector<float, 4> contents{1.0f, 2.0f, 3.0f, 4.0f};
  absl::InlinedVector<int32_t, 5> shape{1, 2, 1, 1, 2};
  TestToStringTensor(contents, shape, expected);
}

}  // namespace
}  // namespace iree
