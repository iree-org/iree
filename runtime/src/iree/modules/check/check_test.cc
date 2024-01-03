// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests that our bytecode module can call through into our native module.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/span.h"
#include "iree/hal/api.h"
#include "iree/modules/check/module.h"
#include "iree/modules/hal/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

class CheckTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    IREE_ASSERT_OK(iree_vm_instance_create(
        IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance_));
    IREE_ASSERT_OK(iree_hal_module_register_all_types(instance_));

    iree_hal_driver_t* hal_driver = nullptr;
    iree_status_t status = iree_hal_driver_registry_try_create(
        iree_hal_available_driver_registry(),
        iree_make_cstring_view("local-task"), iree_allocator_system(),
        &hal_driver);
    if (iree_status_is_not_found(status)) {
      fprintf(stderr, "Skipping test as 'local-task' driver was not found:\n");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      return;
    }
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        hal_driver, iree_allocator_system(), &device_));
    IREE_ASSERT_OK(iree_hal_module_create(
        instance_, /*device_count=*/1, &device_, IREE_HAL_MODULE_FLAG_NONE,
        iree_allocator_system(), &hal_module_));
    iree_hal_driver_release(hal_driver);

    IREE_ASSERT_OK(iree_check_module_create(instance_, iree_allocator_system(),
                                            &check_module_))
        << "Native module failed to init";
  }

  static void TearDownTestSuite() {
    iree_hal_device_release(device_);
    iree_vm_module_release(check_module_);
    iree_vm_module_release(hal_module_);
    iree_vm_instance_release(instance_);
  }

  void SetUp() override {
    if (!device_) {
      GTEST_SKIP();
    }
    std::vector<iree_vm_module_t*> modules = {hal_module_, check_module_};
    IREE_ASSERT_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
    allocator_ = iree_hal_device_allocator(device_);
  }

  void TearDown() override {
    inputs_.reset();
    iree_vm_context_release(context_);
  }

  void CreateInt32BufferView(iree::span<const int32_t> contents,
                             iree::span<const iree_hal_dim_t> shape,
                             iree_hal_buffer_view_t** out_buffer_view) {
    iree_hal_dim_t num_elements = 1;
    for (iree_hal_dim_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    IREE_ASSERT_OK(iree_hal_buffer_view_allocate_buffer_copy(
        device_, allocator_, shape.size(), shape.data(),
        IREE_HAL_ELEMENT_TYPE_INT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params,
        iree_make_const_byte_span(contents.data(),
                                  contents.size() * sizeof(int32_t)),
        &*out_buffer_view));
  }

  void CreateFloat16BufferView(iree::span<const uint16_t> contents,
                               iree::span<const iree_hal_dim_t> shape,
                               iree_hal_buffer_view_t** out_buffer_view) {
    iree_hal_dim_t num_elements = 1;
    for (iree_hal_dim_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    IREE_ASSERT_OK(iree_hal_buffer_view_allocate_buffer_copy(
        device_, allocator_, shape.size(), shape.data(),
        IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params,
        iree_make_const_byte_span(contents.data(),
                                  contents.size() * sizeof(uint16_t)),
        &*out_buffer_view));
  }

  void CreateFloat32BufferView(iree::span<const float> contents,
                               iree::span<const iree_hal_dim_t> shape,
                               iree_hal_buffer_view_t** out_buffer_view) {
    iree_hal_dim_t num_elements = 1;
    for (iree_hal_dim_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    IREE_ASSERT_OK(iree_hal_buffer_view_allocate_buffer_copy(
        device_, allocator_, shape.size(), shape.data(),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params,
        iree_make_const_byte_span(contents.data(),
                                  contents.size() * sizeof(float)),
        &*out_buffer_view));
  }

  void CreateFloat64BufferView(iree::span<const double> contents,
                               iree::span<const iree_hal_dim_t> shape,
                               iree_hal_buffer_view_t** out_buffer_view) {
    iree_hal_dim_t num_elements = 1;
    for (iree_hal_dim_t dim : shape) {
      num_elements *= dim;
    }
    ASSERT_EQ(contents.size(), num_elements);
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    IREE_ASSERT_OK(iree_hal_buffer_view_allocate_buffer_copy(
        device_, allocator_, shape.size(), shape.data(),
        IREE_HAL_ELEMENT_TYPE_FLOAT_64, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params,
        iree_make_const_byte_span(contents.data(),
                                  contents.size() * sizeof(double)),
        &*out_buffer_view));
  }

  iree_status_t Invoke(const char* function_name) {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
                             check_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                             iree_make_cstring_view(function_name), &function),
                         "exported function '%s' not found", function_name);
    // TODO(#2075): don't directly invoke native functions like this.
    return iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                          /*policy=*/nullptr, inputs_.get(),
                          /*outputs=*/nullptr, iree_allocator_system());
  }

  iree_status_t InvokeValue(const char* function_name,
                            std::vector<iree_vm_value_t> args) {
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), args.size(),
                            iree_allocator_system(), &inputs_));
    for (auto& arg : args) {
      IREE_RETURN_IF_ERROR(iree_vm_list_push_value(inputs_.get(), &arg));
    }
    return Invoke(function_name);
  }

  iree_status_t Invoke(const char* function_name,
                       std::vector<vm::ref<iree_hal_buffer_view_t>> args) {
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), args.size(),
                            iree_allocator_system(), &inputs_));
    iree_vm_ref_t device_ref = iree_hal_device_retain_ref(device_);
    IREE_RETURN_IF_ERROR(
        iree_vm_list_push_ref_move(inputs_.get(), &device_ref));
    for (auto& arg : args) {
      iree_vm_ref_t arg_ref = iree_hal_buffer_view_retain_ref(arg.get());
      IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs_.get(), &arg_ref));
    }
    return Invoke(function_name);
  }

 private:
  static iree_hal_device_t* device_;
  static iree_vm_instance_t* instance_;
  static iree_vm_module_t* check_module_;
  static iree_vm_module_t* hal_module_;

  iree_vm_context_t* context_ = nullptr;
  vm::ref<iree_vm_list_t> inputs_;
  iree_hal_allocator_t* allocator_ = nullptr;
};
iree_hal_device_t* CheckTest::device_ = nullptr;
iree_vm_instance_t* CheckTest::instance_ = nullptr;
iree_vm_module_t* CheckTest::check_module_ = nullptr;
iree_vm_module_t* CheckTest::hal_module_ = nullptr;

TEST_F(CheckTest, ExpectTrueSuccess) {
  IREE_ASSERT_OK(InvokeValue("expect_true", {iree_vm_value_make_i32(1)}));
}

TEST_F(CheckTest, ExpectTrueFailure) {
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(InvokeValue("expect_true", {iree_vm_value_make_i32(0)})),
      "Expected 0 to be nonzero");
}

TEST_F(CheckTest, ExpectFalseSuccess) {
  IREE_ASSERT_OK(InvokeValue("expect_false", {iree_vm_value_make_i32(0)}));
}

TEST_F(CheckTest, ExpectFalseFailure) {
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(InvokeValue("expect_false", {iree_vm_value_make_i32(1)})),
      "Expected 1 to be zero");
}

TEST_F(CheckTest, ExpectFalseNotOneFailure) {
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(InvokeValue("expect_false", {iree_vm_value_make_i32(42)})),
      "Expected 42 to be zero");
}

TEST_F(CheckTest, ExpectAllTrueSuccess) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  IREE_ASSERT_OK(Invoke("expect_all_true", {input_buffer_view}));
}

TEST_F(CheckTest, ExpectAllTrue3DTrueSuccess) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1, 2, 3, 4, 5, 6, 7, 8};
  iree_hal_dim_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  IREE_ASSERT_OK(Invoke("expect_all_true", {input_buffer_view}));
}

TEST_F(CheckTest, ExpectAllTrueFailure) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {0};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_all_true", {input_buffer_view})),
      "element #0 doesn't match");
}

TEST_F(CheckTest, ExpectAllTrueSingleElementFailure) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1, 2, 3, 0, 4};
  iree_hal_dim_t shape[] = {5};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_all_true", {input_buffer_view})),
      "element #3 doesn't match");
}

TEST_F(CheckTest, ExpectAllTrue3DSingleElementFailure) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1, 2, 3, 4, 5, 6, 0, 8};
  iree_hal_dim_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_all_true", {input_buffer_view})),
      "element #6 doesn't match");
}

TEST_F(CheckTest, ExpectEqSameBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  int32_t contents[] = {1};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(
      CreateInt32BufferView(contents, shape, &input_buffer_view));
  IREE_ASSERT_OK(Invoke("expect_eq", {input_buffer_view, input_buffer_view}));
}

TEST_F(CheckTest, ExpectEqIdenticalBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t contents[] = {1};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectEqIdentical3DBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t contents[] = {1, 2, 3, 4, 5, 6, 7, 8};
  iree_hal_dim_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectEqDifferentShapeFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t contents[] = {1, 2, 3, 4};
  iree_hal_dim_t lhs_shape[] = {2, 2};
  iree_hal_dim_t rhs_shape[] = {4};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(contents, lhs_shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(contents, rhs_shape, &rhs));
  EXPECT_NONFATAL_FAILURE(IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs})),
                          "Shapes do not match");
}

TEST_F(CheckTest, ExpectEqDifferentElementTypeFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t lhs_contents[] = {1, 2, 3, 4};
  float rhs_contents[] = {1, 2, 3, 4};
  iree_hal_dim_t shape[] = {2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs})),
                          "Element types do not match");
}

TEST_F(CheckTest, ExpectEqDifferentContentsFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t lhs_contents[] = {1};
  int32_t rhs_contents[] = {2};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs})),
                          "Contents does not match");
}

TEST_F(CheckTest, ExpectEqDifferentEverythingFullMessageFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t lhs_contents[] = {1, 2, 3, 4, 5, 6};
  float rhs_contents[] = {1, 2, 3, 42};
  iree_hal_dim_t lhs_shape[] = {2, 3};
  iree_hal_dim_t rhs_shape[] = {2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(lhs_contents, lhs_shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(rhs_contents, rhs_shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs})),
      "Expected equality of these values. Element types do not match."
      " Shapes do not match. Contents does not match.\n"
      "  lhs:\n"
      "    2x3xi32=[1 2 3][4 5 6]\n"
      "  rhs:\n"
      "    2x2xf32=[1 2][3 42]");
}

TEST_F(CheckTest, ExpectEqDifferentContents3DFullMessageFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  int32_t lhs_contents[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t rhs_contents[] = {1, 2, 3, 42, 5, 6, 7, 8};
  iree_hal_dim_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateInt32BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_eq", {lhs, rhs})),
      "Expected equality of these values. Contents does not match.\n"
      "  lhs:\n"
      "    2x2x2xi32=[[1 2][3 4]][[5 6][7 8]]\n"
      "  rhs:\n"
      "    2x2x2xi32=[[1 2][3 42]][[5 6][7 8]]");
}

TEST_F(CheckTest, ExpectAlmostEqSameBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> input_buffer_view;
  float contents[] = {1};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(contents, shape, &input_buffer_view));
  IREE_ASSERT_OK(
      Invoke("expect_almost_eq", {input_buffer_view, input_buffer_view}));
}

TEST_F(CheckTest, ExpectAlmostEqIdenticalBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  float contents[] = {1};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectAlmostEqNearIdenticalBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  float lhs_contents[] = {1.0f, 1.99999f, 0.00001f, 4.0f};
  float rhs_contents[] = {1.00001f, 2.0f, 0.0f, 4.0f};
  iree_hal_dim_t shape[] = {4};
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(rhs_contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectAlmostEqIdentical3DBufferSuccess) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  float contents[] = {1, 2, 3, 4, 5, 6, 7, 8};
  iree_hal_dim_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectAlmostEqDifferentShapeFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  float contents[] = {1, 2, 3, 4};
  iree_hal_dim_t lhs_shape[] = {2, 2};
  iree_hal_dim_t rhs_shape[] = {4};
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(contents, lhs_shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(contents, rhs_shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs})),
      "Shapes do not match");
}

TEST_F(CheckTest, ExpectAlmostEqSmallerLhsElementCountFailure) {
  vm::ref<iree_hal_buffer_view_t> smaller;
  vm::ref<iree_hal_buffer_view_t> bigger;
  float smaller_contents[] = {1, 2};
  float bigger_contents[] = {1, 2, 3, 4};
  iree_hal_dim_t smaller_shape[] = {2};
  iree_hal_dim_t bigger_shape[] = {4};
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(smaller_contents, smaller_shape, &smaller));
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(bigger_contents, bigger_shape, &bigger));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {smaller, bigger})),
      "Shapes do not match");
}

TEST_F(CheckTest, ExpectAlmostEqSmallerRhsElementCountFailure) {
  vm::ref<iree_hal_buffer_view_t> smaller;
  vm::ref<iree_hal_buffer_view_t> bigger;
  float smaller_contents[] = {1, 2};
  float bigger_contents[] = {1, 2, 3, 4};
  iree_hal_dim_t smaller_shape[] = {2};
  iree_hal_dim_t bigger_shape[] = {4};
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(smaller_contents, smaller_shape, &smaller));
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(bigger_contents, bigger_shape, &bigger));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {bigger, smaller})),
      "Shapes do not match");
}

TEST_F(CheckTest, ExpectAlmostEqDifferentElementTypeFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  double lhs_contents[] = {1, 2, 3, 4};
  float rhs_contents[] = {1, 2, 3, 4};
  iree_hal_dim_t shape[] = {2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateFloat64BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs})),
      "Element types do not match");
}

TEST_F(CheckTest, ExpectAlmostEqDifferentContentsFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  float lhs_contents[] = {1};
  float rhs_contents[] = {2};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs})),
      "Contents does not match");
}

TEST_F(CheckTest, ExpectAlmostEqDifferentEverythingFullMessageFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  double lhs_contents[] = {1, 2, 3, 4, 5, 6};
  float rhs_contents[] = {1, 2, 3, 42};
  iree_hal_dim_t lhs_shape[] = {2, 3};
  iree_hal_dim_t rhs_shape[] = {2, 2};
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat64BufferView(lhs_contents, lhs_shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(
      CreateFloat32BufferView(rhs_contents, rhs_shape, &rhs));
  // Note no comment on contents. Cannot compare different shapes and element
  // types.
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs})),
      "Expected near equality of these values. Element types do not match."
      " Shapes do not match.\n"
      "  lhs:\n"
      "    2x3xf64=[1 2 3][4 5 6]\n"
      "  rhs:\n"
      "    2x2xf32=[1 2][3 42]");
}

TEST_F(CheckTest, ExpectAlmostEqDifferentContents3DFullMessageFailure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  float lhs_contents[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float rhs_contents[] = {1, 2, 3, 42, 5, 6, 7, 8};
  iree_hal_dim_t shape[] = {2, 2, 2};
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat32BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs})),
      "Expected near equality of these values. Contents does not match.\n"
      "  lhs:\n"
      "    2x2x2xf32=[[1 2][3 4]][[5 6][7 8]]\n"
      "  rhs:\n"
      "    2x2x2xf32=[[1 2][3 42]][[5 6][7 8]]");
}

TEST_F(CheckTest, ExpectAlmostEqIdenticalBufferF16Success) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  uint16_t contents[] = {iree_math_f32_to_f16(1.f)};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(CreateFloat16BufferView(contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat16BufferView(contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectAlmostEqNearIdenticalBufferF16Success) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  uint16_t lhs_contents[] = {
      iree_math_f32_to_f16(1.0f), iree_math_f32_to_f16(1.99999f),
      iree_math_f32_to_f16(0.00001f), iree_math_f32_to_f16(4.0f)};
  uint16_t rhs_contents[] = {
      iree_math_f32_to_f16(1.00001f), iree_math_f32_to_f16(2.0f),
      iree_math_f32_to_f16(0.0f), iree_math_f32_to_f16(4.0f)};
  iree_hal_dim_t shape[] = {4};
  ASSERT_NO_FATAL_FAILURE(CreateFloat16BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat16BufferView(rhs_contents, shape, &rhs));
  IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs}));
}

TEST_F(CheckTest, ExpectAlmostEqDifferentContentsF16Failure) {
  vm::ref<iree_hal_buffer_view_t> lhs;
  vm::ref<iree_hal_buffer_view_t> rhs;
  uint16_t lhs_contents[] = {iree_math_f32_to_f16(1.f)};
  uint16_t rhs_contents[] = {iree_math_f32_to_f16(2.f)};
  iree_hal_dim_t shape[] = {1};
  ASSERT_NO_FATAL_FAILURE(CreateFloat16BufferView(lhs_contents, shape, &lhs));
  ASSERT_NO_FATAL_FAILURE(CreateFloat16BufferView(rhs_contents, shape, &rhs));
  EXPECT_NONFATAL_FAILURE(
      IREE_ASSERT_OK(Invoke("expect_almost_eq", {lhs, rhs})),
      "Contents does not match");
}
}  // namespace
}  // namespace iree
