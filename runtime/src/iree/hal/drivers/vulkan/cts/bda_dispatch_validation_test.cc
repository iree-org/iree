// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan BDA-specific validation coverage. These cases assert failures at the
// HAL boundary before malformed pointer tables can reach the device.

#include <cstdint>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

class BdaDispatchValidationTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));
    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_const_byte_span_t constants() const {
    return iree_make_const_byte_span(constant_data_, sizeof(constant_data_));
  }

  iree_status_t CreateInputOutputBuffers(
      iree_hal_buffer_t** out_input_buffer,
      iree_hal_buffer_t** out_output_buffer) {
    *out_input_buffer = nullptr;
    *out_output_buffer = nullptr;
    const uint32_t input_data[4] = {1, 2, 3, 4};
    IREE_RETURN_IF_ERROR(CreateDeviceBufferWithData(
        input_data, sizeof(input_data), out_input_buffer));
    iree_status_t status =
        CreateZeroedDeviceBuffer(sizeof(input_data), out_output_buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_buffer_release(*out_input_buffer);
      *out_input_buffer = nullptr;
    }
    return status;
  }

  static constexpr uint32_t constant_data_[2] = {3, 10};

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsBindingCountMismatch) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable_, /*export_ordinal=*/0,
          iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferDispatchRejectsBindingCountMismatch) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(Status(iree_hal_command_buffer_dispatch(
                  command_buffer, executable_, /*entry_point=*/0,
                  iree_hal_make_static_dispatch_config(1, 1, 1), constants(),
                  bindings, IREE_HAL_DISPATCH_FLAG_NONE)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest, QueueDispatchRejectsEmptyBindingRange) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, /*length=*/0),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  EXPECT_THAT(
      Status(iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable_, /*export_ordinal=*/0,
          iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
          IREE_HAL_DISPATCH_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

TEST_P(BdaDispatchValidationTest,
       CommandBufferExecuteRejectsEmptyBindingRange) {
  iree_hal_buffer_t* input_buffer = nullptr;
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(CreateInputOutputBuffers(&input_buffer, &output_buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, /*length=*/0),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants(), bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_THAT(
      Status(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), command_buffer,
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE)),
      StatusIs(StatusCode::kInvalidArgument));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(BdaDispatchValidationTest);

}  // namespace iree::hal::cts
