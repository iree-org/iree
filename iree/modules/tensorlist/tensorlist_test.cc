// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests that our bytecode module can call through into our native module.

#include <vector>

#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/hal/vmvx/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/tensorlist/native_module.h"
#include "iree/modules/tensorlist/tensorlist_test_module_c.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

namespace iree {

namespace {

class TensorListModulesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_CHECK_OK(iree_hal_vmvx_driver_module_register(
        iree_hal_driver_registry_default()));
  }

  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    // TODO(benvanik): move to instance-based registration.
    IREE_CHECK_OK(iree_hal_module_register_types());
    // TODO(benvanik): make a 'don't care' helper method.
    iree_hal_driver_t* hal_driver = nullptr;
    IREE_CHECK_OK(iree_hal_driver_registry_try_create_by_name(
        iree_hal_driver_registry_default(), iree_make_cstring_view("vmvx"),
        iree_allocator_system(), &hal_driver));
    IREE_CHECK_OK(iree_hal_driver_create_default_device(
        hal_driver, iree_allocator_system(), &device_));
    IREE_CHECK_OK(
        iree_hal_module_create(device_, iree_allocator_system(), &hal_module_));
    iree_hal_driver_release(hal_driver);

    IREE_CHECK_OK(iree_tensorlist_module_register_types());
    IREE_CHECK_OK(iree_tensorlist_module_create(iree_allocator_system(),
                                                &native_module_));

    const auto* module_file_toc = iree_tensorlist_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {hal_module_, native_module_,
                                              bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), iree_allocator_system(),
        &context_));
  }

  virtual void TearDown() {
    iree_hal_device_release(device_);
    iree_vm_module_release(native_module_);
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

  void Invoke(const char* function_name, absl::Span<const float> input_values,
              absl::Span<const int32_t> input_shape,
              absl::Span<const float> expected_values,
              absl::Span<const int32_t> expected_shape) {
    vm::ref<iree_hal_buffer_view_t> input_buffer_view;
    CreateBufferView(input_values, input_shape, device_, &input_buffer_view);

    // Pass in the tensor as a HAL buffer view.
    vm::ref<iree_vm_list_t> inputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &inputs));
    iree_vm_ref_t input_buffer_view_ref =
        iree_hal_buffer_view_move_ref(input_buffer_view.get());
    IREE_ASSERT_OK(
        iree_vm_list_push_ref_retain(inputs.get(), &input_buffer_view_ref));

    // Prepare outputs list to accept the results from the invocation.
    vm::ref<iree_vm_list_t> outputs;
    IREE_ASSERT_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                       iree_allocator_system(), &outputs));

    // Synchronously invoke the function.
    IREE_ASSERT_OK(iree_vm_invoke(context_, LookupFunction(function_name),
                                  /*policy=*/nullptr, inputs.get(),
                                  outputs.get(), iree_allocator_system()));

    auto* returned_buffer_view =
        reinterpret_cast<iree_hal_buffer_view_t*>(iree_vm_list_get_ref_deref(
            outputs.get(), 0, iree_hal_buffer_view_get_descriptor()));

    std::vector<int32_t> returned_shape(
        iree_hal_buffer_view_shape_rank(returned_buffer_view));
    if (returned_shape.size() > 0) {
      iree_hal_buffer_view_shape(returned_buffer_view, returned_shape.size(),
                                 returned_shape.data(), nullptr);
    }

    EXPECT_EQ(returned_shape, expected_shape);

    iree_hal_buffer_t* returned_buffer =
        iree_hal_buffer_view_buffer(returned_buffer_view);
    ASSERT_NE(returned_buffer, nullptr);

    iree_hal_buffer_mapping_t mapped_memory;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_range(returned_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                  0, IREE_WHOLE_BUFFER, &mapped_memory));
    for (int i = 0; i < expected_values.size(); i++) {
      EXPECT_EQ(reinterpret_cast<float*>(mapped_memory.contents.data)[i],
                expected_values[i]);
    }

    iree_hal_buffer_unmap_range(&mapped_memory);
  }

  void CreateBufferView(absl::Span<const float> contents,
                        absl::Span<const int32_t> shape,
                        iree_hal_device_t* device,
                        iree_hal_buffer_view_t** out_buffer_view) {
    size_t num_elements = 1;
    for (int32_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    vm::ref<iree_hal_buffer_t> buffer;
    iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator,
        static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        IREE_HAL_BUFFER_USAGE_ALL, contents.size() * sizeof(float), &buffer));
    IREE_ASSERT_OK(iree_hal_buffer_write_data(buffer.get(), 0, contents.data(),
                                              contents.size() * sizeof(float)));
    IREE_ASSERT_OK(iree_hal_buffer_view_create(
        buffer.get(), shape.data(), shape.size(),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, &*out_buffer_view));
  }

  iree_hal_device_t* device_ = nullptr;
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
  iree_vm_module_t* hal_module_ = nullptr;
};

TEST_F(TensorListModulesTest, IdentityThroughSetItemGetItem) {
  // Allocate the buffer we'll be passing through.
  std::vector<float> input = {42.0f};
  std::vector<int32_t> input_shape = {};
  Invoke("identity_through_set_item_get_item", input, input_shape, input,
         input_shape);
}

TEST_F(TensorListModulesTest, IdentityThroughSetItemGetItem2D) {
  // Allocate the buffer we'll be passing through.
  std::vector<float> input = {42.0f};
  std::vector<int32_t> input_shape = {1, 1};
  Invoke("identity_through_set_item_get_item", input, input_shape, input,
         input_shape);
}

TEST_F(TensorListModulesTest, IdentityThroughConcat) {
  // Allocate the buffer we'll be passing through.
  std::vector<float> input = {42.0f, 43.0f, 44.0f, 45.0f};
  std::vector<int32_t> input_shape = {4, 1};
  std::vector<int32_t> expected_shape = {4};
  Invoke("identity_through_concat", input, input_shape, input, expected_shape);
}

TEST_F(TensorListModulesTest, ConcatAppendsEmpty) {
  // Allocate the buffer we'll be passing through.
  std::vector<float> input = {42.0f};
  std::vector<int32_t> input_shape = {1};
  std::vector<float> expected = {42.0f, 0.0f};
  std::vector<int32_t> expected_shape = {2};
  Invoke("concat_appends_empty", input, input_shape, expected, expected_shape);
}

TEST_F(TensorListModulesTest, IdentityThroughStack) {
  // Allocate the buffer we'll be passing through.
  std::vector<float> input = {42.0f, 43.0f};
  std::vector<int32_t> input_shape = {2, 1};
  Invoke("identity_through_stack", input, input_shape, input, input_shape);
}

TEST_F(TensorListModulesTest, StackAppendsEmpty) {
  // Allocate the buffer we'll be passing through.
  std::vector<float> input = {42.0f};
  std::vector<int32_t> input_shape = {};
  std::vector<float> expected = {42.0f, 0.0f};
  std::vector<int32_t> expected_shape = {2};
  Invoke("stack_appends_empty", input, input_shape, expected, expected_shape);
}

}  // namespace
}  // namespace iree
