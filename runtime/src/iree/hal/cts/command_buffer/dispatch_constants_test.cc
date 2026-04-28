// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL command buffer dispatch with push constants.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class DispatchConstantsTest : public CtsTestBase<> {
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
    executable_params.executable_data = executable_data(
        iree_make_cstring_view("command_buffer_dispatch_constants_test.bin"));
    executable_params.constant_count = 0;
    executable_params.constants = nullptr;

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

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(DispatchConstantsTest, DispatchWithDispatchConstants) {
  // Create output buffer.
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), &output_buffer));

  // Set up bindings for direct or indirect recording mode.
  iree_hal_buffer_ref_t binding_refs[1];
  iree_hal_buffer_binding_t binding_table_values[1];
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  switch (recording_mode()) {
    case RecordingMode::kDirect:
      binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/output_buffer,
          /*offset=*/iree_hal_buffer_byte_offset(output_buffer),
          /*length=*/iree_hal_buffer_byte_length(output_buffer),
      };
      break;
    case RecordingMode::kIndirect:
      binding_table.count = IREE_ARRAYSIZE(binding_refs);
      binding_table.bindings = binding_table_values;
      binding_table_values[0] = {
          /*buffer=*/output_buffer,
          /*offset=*/iree_hal_buffer_byte_offset(output_buffer),
          /*length=*/iree_hal_buffer_byte_length(output_buffer),
      };
      binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/nullptr,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(output_buffer),
      };
      break;
  }
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_table.count, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  std::vector<uint32_t> constant_data{11, 22, 33, 44};
  iree_const_byte_span_t constants = iree_make_const_byte_span(
      constant_data.data(), constant_data.size() * sizeof(constant_data[0]));

  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH |
          IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
      /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer, binding_table));

  std::vector<uint32_t> output_data(4);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, output_buffer, /*source_offset=*/0,
      /*target_buffer=*/output_data.data(),
      /*data_length=*/output_data.size() * sizeof(uint32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  EXPECT_THAT(output_data, ContainerEq(constant_data));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
}

CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(DispatchConstantsTest);

}  // namespace iree::hal::cts
