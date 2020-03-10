// Copyright 2019 Google LLC
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
#include "absl/strings/string_view.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/logging.h"
#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/api.h"
#include "iree/modules/check/check_test_module.h"
#include "iree/modules/check/native_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref.h"
#include "iree/vm/ref_cc.h"

namespace iree {
namespace {

class CheckTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // NOTE: we only use the HAL types here, we don't need the full module.
    // TODO(benvanik): move to instance-based registration.
    IREE_ASSERT_OK(iree_hal_module_register_types());

    iree_hal_driver_t* hal_driver = nullptr;
    IREE_ASSERT_OK(iree_hal_driver_registry_create_driver(
        iree_make_cstring_view("vmla"), IREE_ALLOCATOR_SYSTEM, &hal_driver));
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        hal_driver, IREE_ALLOCATOR_SYSTEM, &device_));
    IREE_ASSERT_OK(
        iree_hal_module_create(device_, IREE_ALLOCATOR_SYSTEM, &hal_module_));
    iree_hal_driver_release(hal_driver);

    IREE_ASSERT_OK(iree_vm_instance_create(IREE_ALLOCATOR_SYSTEM, &instance_));

    IREE_ASSERT_OK(
        check_native_module_create(IREE_ALLOCATOR_SYSTEM, &check_module_))
        << "Native module failed to init";

    const auto* module_file_toc = iree::check::check_test_module_create();
    IREE_ASSERT_OK(iree_vm_bytecode_module_create(
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        IREE_ALLOCATOR_NULL, IREE_ALLOCATOR_SYSTEM, &input_module_))
        << "Bytecode module failed to load";
  }

  static void TearDownTestSuite() {
    iree_hal_device_release(device_);
    iree_vm_module_release(check_module_);
    iree_vm_module_release(input_module_);
    iree_vm_module_release(hal_module_);
    iree_vm_instance_release(instance_);
  }

  void SetUp() override {
    std::vector<iree_vm_module_t*> modules = {hal_module_, check_module_,
                                              input_module_};
    IREE_ASSERT_OK(iree_vm_context_create_with_modules(
        instance_, modules.data(), modules.size(), IREE_ALLOCATOR_SYSTEM,
        &context_));
    allocator_ = iree_hal_device_allocator(device_);
  }

  void TearDown() override {
    iree_vm_context_release(context_);
    if (inputs_) iree_vm_variant_list_free(inputs_);
  }

  void CreateInt32BufferView(absl::Span<const int32_t> contents,
                             absl::Span<const int32_t> shape,
                             iree_hal_buffer_view_t** out_buffer_view) {
    size_t num_elements = 1;
    for (int32_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    vm::ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator_,
        static_cast<iree_hal_memory_type_t>(
            IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE),
        IREE_HAL_BUFFER_USAGE_ALL, contents.size() * sizeof(int32_t), &buffer));
    iree_hal_mapped_memory_t mapped_memory;
    IREE_ASSERT_OK(iree_hal_buffer_map(buffer.get(),
                                       IREE_HAL_MEMORY_ACCESS_WRITE, 0,
                                       IREE_WHOLE_BUFFER, &mapped_memory));
    memcpy(mapped_memory.contents.data,
           static_cast<const void*>(contents.data()),
           mapped_memory.contents.data_length);
    IREE_ASSERT_OK(iree_hal_buffer_unmap(buffer.get(), &mapped_memory));
    IREE_ASSERT_OK(iree_hal_buffer_view_create(
        buffer.get(), shape.data(), shape.size(), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_ALLOCATOR_SYSTEM, &*out_buffer_view));
  }

  Status Invoke(absl::string_view function_name) {
    iree_vm_function_t function;
    RETURN_IF_ERROR(FromApiStatus(
        input_module_->lookup_function(
            input_module_->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
            iree_string_view_t{function_name.data(), function_name.size()},
            &function),
        IREE_LOC))
        << "Exported function '" << function_name << "' not found";
    return FromApiStatus(
        iree_vm_invoke(context_, function,
                       /*policy=*/nullptr, inputs_,
                       /*outputs=*/nullptr, IREE_ALLOCATOR_SYSTEM),
        IREE_LOC);
  }

  Status Invoke(absl::string_view function_name,
                std::vector<iree_vm_value> args) {
    RETURN_IF_ERROR(
        FromApiStatus(iree_vm_variant_list_alloc(
                          args.size(), IREE_ALLOCATOR_SYSTEM, &inputs_),
                      IREE_LOC));
    for (iree_vm_value& arg : args) {
      RETURN_IF_ERROR(FromApiStatus(
          iree_vm_variant_list_append_value(inputs_, arg), IREE_LOC));
    }
    return Invoke(function_name);
  }

  Status Invoke(absl::string_view function_name,
                std::vector<iree_vm_ref_t*> args) {
    RETURN_IF_ERROR(
        FromApiStatus(iree_vm_variant_list_alloc(
                          args.size(), IREE_ALLOCATOR_SYSTEM, &inputs_),
                      IREE_LOC));
    for (iree_vm_ref_t* arg : args) {
      RETURN_IF_ERROR(FromApiStatus(
          iree_vm_variant_list_append_ref_retain(inputs_, arg), IREE_LOC));
    }
    return Invoke(function_name);
  }

 private:
  static iree_hal_device_t* device_;
  static iree_vm_instance_t* instance_;
  static iree_vm_module_t* input_module_;
  static iree_vm_module_t* check_module_;
  static iree_vm_module_t* hal_module_;

  iree_vm_context_t* context_ = nullptr;
  iree_vm_variant_list_t* inputs_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
};
iree_hal_device_t* CheckTest::device_ = nullptr;
iree_vm_instance_t* CheckTest::instance_ = nullptr;
iree_vm_module_t* CheckTest::input_module_ = nullptr;
iree_vm_module_t* CheckTest::check_module_ = nullptr;
iree_vm_module_t* CheckTest::hal_module_ = nullptr;

TEST_F(CheckTest, ExpectTrueSuccess) {
  ASSERT_OK(Invoke("expectTrue", {IREE_VM_VALUE_MAKE_I32(1)}));
}

TEST_F(CheckTest, ExpectTrueFailure) {
  EXPECT_NONFATAL_FAILURE(
      ASSERT_OK(Invoke("expectTrue", {IREE_VM_VALUE_MAKE_I32(0)})),
      "Expected 0 to be nonzero");
}

TEST_F(CheckTest, ExpectFalseSuccess) {
  ASSERT_OK(Invoke("expectFalse", {IREE_VM_VALUE_MAKE_I32(0)}));
}

TEST_F(CheckTest, ExpectFalseFailure) {
  EXPECT_NONFATAL_FAILURE(
      ASSERT_OK(Invoke("expectFalse", {IREE_VM_VALUE_MAKE_I32(1)})),
      "Expected 1 to be zero");
}

TEST_F(CheckTest, ExpectFalseNotOneFailure) {
  EXPECT_NONFATAL_FAILURE(
      ASSERT_OK(Invoke("expectFalse", {IREE_VM_VALUE_MAKE_I32(42)})),
      "Expected 42 to be zero");
}

TEST_F(CheckTest, ExpectAllTrueSuccess) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1};
  int32_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  iree_vm_ref_t input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(input_buffer_view.get());
  ASSERT_OK(Invoke("expectAllTrue", {&input_buffer_view_ref}));
}

TEST_F(CheckTest, ExpectAllTrue3DTrueSuccess) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  iree_vm_ref_t input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(input_buffer_view.get());
  ASSERT_OK(Invoke("expectAllTrue", {&input_buffer_view_ref}));
}

TEST_F(CheckTest, ExpectAllTrueFailure) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {0};
  int32_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  iree_vm_ref_t input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(input_buffer_view.get());
  EXPECT_NONFATAL_FAILURE(
      ASSERT_OK(Invoke("expectAllTrue", {&input_buffer_view_ref})), "0");
}

TEST_F(CheckTest, ExpectAllTrueSingleElementFailure) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1, 2, 3, 0, 4};
  int32_t shape[] = {5};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  iree_vm_ref_t input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(input_buffer_view.get());
  EXPECT_NONFATAL_FAILURE(
      ASSERT_OK(Invoke("expectAllTrue", {&input_buffer_view_ref})),
      "1, 2, 3, 0, 4");
}

TEST_F(CheckTest, ExpectAllTrue3DSingleElementFailure) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1, 2, 3, 4, 5, 6, 0, 8};
  int32_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  iree_vm_ref_t input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(input_buffer_view.get());
  EXPECT_NONFATAL_FAILURE(
      ASSERT_OK(Invoke("expectAllTrue", {&input_buffer_view_ref})),
      "1, 2, 3, 4, 5, 6, 0, 8");
}

}  // namespace
}  // namespace iree
