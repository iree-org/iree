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

// Tests that our bytecode module can call through into our native module.

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/tensorlist/native_module.h"
#include "iree/modules/tensorlist/tensorlist_test_module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

namespace iree {

namespace {

class TensorListModulesTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    // TODO(benvanik): move to instance-based registration.
    IREE_CHECK_OK(iree_hal_module_register_types());
    // TODO(benvanik): make a 'don't care' helper method.
    iree_hal_driver_t* hal_driver = nullptr;
    IREE_CHECK_OK(iree_hal_driver_registry_create_driver(
        iree_make_cstring_view("vmla"), iree_allocator_system(), &hal_driver));
    IREE_CHECK_OK(iree_hal_driver_create_default_device(
        hal_driver, iree_allocator_system(), &device_));
    IREE_CHECK_OK(
        iree_hal_module_create(device_, iree_allocator_system(), &hal_module_));
    iree_hal_driver_release(hal_driver);

    IREE_CHECK_OK(iree_tensorlist_module_register_types());
    IREE_CHECK_OK(
        iree_tensorlist_module_create(iree_allocator_system(), &native_module_))
        << "Native module failed to init";

    const auto* module_file_toc =
        iree::modules::tensorlist::tensorlist_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_))
        << "Bytecode module failed to load";

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

  iree_vm_function_t LookupFunction(absl::string_view function_name) {
    iree_vm_function_t function;
    IREE_CHECK_OK(bytecode_module_->lookup_function(
        bytecode_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{function_name.data(), function_name.size()},
        &function))
        << "Exported function '" << function_name << "' not found";
    return function;
  }

  iree_hal_device_t* device_ = nullptr;
  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* native_module_ = nullptr;
  iree_vm_module_t* hal_module_ = nullptr;
};

void CreateBufferView(absl::Span<float> contents,
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
      static_cast<iree_hal_memory_type_t>(IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
      IREE_HAL_BUFFER_USAGE_ALL, contents.size() * sizeof(float), &buffer));
  iree_hal_mapped_memory_t mapped_memory;
  IREE_ASSERT_OK(iree_hal_buffer_map(buffer.get(), IREE_HAL_MEMORY_ACCESS_WRITE,
                                     0, IREE_WHOLE_BUFFER, &mapped_memory));
  memcpy(mapped_memory.contents.data, static_cast<void*>(contents.data()),
         mapped_memory.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap(buffer.get(), &mapped_memory));
  IREE_ASSERT_OK(iree_hal_buffer_view_create(
      buffer.get(), shape.data(), shape.size(), IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      iree_allocator_system(), &*out_buffer_view));
}

TEST_F(TensorListModulesTest, IdentityThroughSetItemGetItem) {
  // Allocate the buffer we'll be passing through.
  static float kBufferContents[1] = {42.0f};
  absl::InlinedVector<int32_t, 4> shape;
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  CreateBufferView(kBufferContents, shape, device_, &input_buffer_view);

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
  IREE_ASSERT_OK(iree_vm_invoke(
      context_, LookupFunction("identity_through_set_item_get_item"),
      /*policy=*/nullptr, inputs.get(), outputs.get(),
      iree_allocator_system()));

  auto* returned_buffer_view =
      reinterpret_cast<iree_hal_buffer_view_t*>(iree_vm_list_get_ref_deref(
          outputs.get(), 0, iree_hal_buffer_view_get_descriptor()));
  ASSERT_NE(nullptr, returned_buffer_view);
  iree_hal_buffer_t* returned_buffer =
      iree_hal_buffer_view_buffer(returned_buffer_view);
  ASSERT_NE(nullptr, returned_buffer);

  iree_hal_mapped_memory_t mapped_memory;
  IREE_ASSERT_OK(iree_hal_buffer_map(returned_buffer,
                                     IREE_HAL_MEMORY_ACCESS_READ, 0,
                                     IREE_WHOLE_BUFFER, &mapped_memory));
  EXPECT_EQ(reinterpret_cast<float*>(mapped_memory.contents.data)[0],
            kBufferContents[0]);
  IREE_ASSERT_OK(iree_hal_buffer_unmap(returned_buffer, &mapped_memory));
}

TEST_F(TensorListModulesTest, IdentityThroughConcat) {
  // Allocate the buffer we'll be passing through.
  static float kBufferContents[4] = {42.0f, 43.0f, 44.0f, 45.0f};
  absl::InlinedVector<int32_t, 4> shape = {4, 1};
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  CreateBufferView(kBufferContents, shape, device_, &input_buffer_view);

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
  IREE_ASSERT_OK(iree_vm_invoke(context_,
                                LookupFunction("identity_through_concat"),
                                /*policy=*/nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  auto* returned_buffer_view =
      reinterpret_cast<iree_hal_buffer_view_t*>(iree_vm_list_get_ref_deref(
          outputs.get(), 0, iree_hal_buffer_view_get_descriptor()));
  ASSERT_NE(nullptr, returned_buffer_view);
  iree_hal_buffer_t* returned_buffer =
      iree_hal_buffer_view_buffer(returned_buffer_view);
  ASSERT_NE(nullptr, returned_buffer);

  // Dimemsionality is reduced by 1.
  auto returned_rank = iree_hal_buffer_view_shape_rank(returned_buffer_view);
  ASSERT_EQ(returned_rank, shape.size() - 1);

  iree_hal_dim_t returned_shape[1];
  iree_hal_buffer_view_shape(returned_buffer_view, 1, returned_shape,
                             &returned_rank);
  EXPECT_EQ(returned_shape[0], shape[0] * shape[1]);

  iree_hal_mapped_memory_t mapped_memory;
  IREE_ASSERT_OK(iree_hal_buffer_map(returned_buffer,
                                     IREE_HAL_MEMORY_ACCESS_READ, 0,
                                     IREE_WHOLE_BUFFER, &mapped_memory));
  EXPECT_EQ(std::memcmp(mapped_memory.contents.data,
                        static_cast<void*>(&kBufferContents[0]),
                        mapped_memory.contents.data_length),
            0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap(returned_buffer, &mapped_memory));
}

TEST_F(TensorListModulesTest, IdentityThroughStack) {
  // Allocate the buffer we'll be passing through.
  static float kBufferContents[2] = {42.0f, 43.0f};
  absl::InlinedVector<int32_t, 4> shape = {2, 1};
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  CreateBufferView(kBufferContents, shape, device_, &input_buffer_view);

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
  IREE_ASSERT_OK(iree_vm_invoke(context_,
                                LookupFunction("identity_through_stack"),
                                /*policy=*/nullptr, inputs.get(), outputs.get(),
                                iree_allocator_system()));

  auto* returned_buffer_view =
      reinterpret_cast<iree_hal_buffer_view_t*>(iree_vm_list_get_ref_deref(
          outputs.get(), 0, iree_hal_buffer_view_get_descriptor()));
  ASSERT_NE(nullptr, returned_buffer_view);
  iree_hal_buffer_t* returned_buffer =
      iree_hal_buffer_view_buffer(returned_buffer_view);
  ASSERT_NE(nullptr, returned_buffer);

  iree_hal_mapped_memory_t mapped_memory;
  IREE_ASSERT_OK(iree_hal_buffer_map(returned_buffer,
                                     IREE_HAL_MEMORY_ACCESS_READ, 0,
                                     IREE_WHOLE_BUFFER, &mapped_memory));
  EXPECT_EQ(std::memcmp(mapped_memory.contents.data,
                        static_cast<void*>(&kBufferContents[0]),
                        mapped_memory.contents.data_length),
            0);
  IREE_ASSERT_OK(iree_hal_buffer_unmap(returned_buffer, &mapped_memory));
}

}  // namespace
}  // namespace iree
