// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for dispatching to multiple entry points within a single executable.
// Verifies that the backend correctly maps ordinal indices to different
// kernel functions and that multiple dispatches from the same executable
// produce correct results.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class DispatchMultiEntrypointTest : public CtsTestBase<> {
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
        "command_buffer_dispatch_multi_entrypoint_test.bin"));
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

// Dispatches two entry points from the same executable:
//   entry_point=0 (negate): output_negate[i] = -input[i]
//   entry_point=1 (double_it): output_double[i] = input[i] * 2
// With input = [1, 2, 3, 4]:
//   output_negate = [-1, -2, -3, -4]
//   output_double = [2, 4, 6, 8]
TEST_P(DispatchMultiEntrypointTest, NegateAndDouble) {
  // Create shared input buffer: [1, 2, 3, 4].
  iree_hal_buffer_t* input_buffer = nullptr;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(uint32_t), &input_buffer);
  }

  // Create two output buffers (zeroed).
  iree_hal_buffer_t* output_negate = nullptr;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), &output_negate);
  iree_hal_buffer_t* output_double = nullptr;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), &output_double);

  // Set up bindings for two dispatches.
  //
  // Direct mode: each dispatch references buffers directly.
  // Indirect mode: binding table has 3 slots:
  //   slot 0 = input (shared)
  //   slot 1 = output_negate (dispatch 0)
  //   slot 2 = output_double (dispatch 1)
  iree_hal_buffer_ref_t negate_binding_refs[2];
  iree_hal_buffer_ref_t double_binding_refs[2];
  iree_hal_buffer_binding_t binding_table_values[3];
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();

  switch (recording_mode()) {
    case RecordingMode::kDirect:
      negate_binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/input_buffer,
          /*offset=*/iree_hal_buffer_byte_offset(input_buffer),
          /*length=*/iree_hal_buffer_byte_length(input_buffer),
      };
      negate_binding_refs[1] = {
          /*binding=*/1,
          /*buffer_slot=*/0,
          /*buffer=*/output_negate,
          /*offset=*/iree_hal_buffer_byte_offset(output_negate),
          /*length=*/iree_hal_buffer_byte_length(output_negate),
      };
      double_binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/input_buffer,
          /*offset=*/iree_hal_buffer_byte_offset(input_buffer),
          /*length=*/iree_hal_buffer_byte_length(input_buffer),
      };
      double_binding_refs[1] = {
          /*binding=*/1,
          /*buffer_slot=*/0,
          /*buffer=*/output_double,
          /*offset=*/iree_hal_buffer_byte_offset(output_double),
          /*length=*/iree_hal_buffer_byte_length(output_double),
      };
      break;
    case RecordingMode::kIndirect:
      binding_table.count = IREE_ARRAYSIZE(binding_table_values);
      binding_table.bindings = binding_table_values;
      binding_table_values[0] = {
          /*buffer=*/input_buffer,
          /*offset=*/iree_hal_buffer_byte_offset(input_buffer),
          /*length=*/iree_hal_buffer_byte_length(input_buffer),
      };
      binding_table_values[1] = {
          /*buffer=*/output_negate,
          /*offset=*/iree_hal_buffer_byte_offset(output_negate),
          /*length=*/iree_hal_buffer_byte_length(output_negate),
      };
      binding_table_values[2] = {
          /*buffer=*/output_double,
          /*offset=*/iree_hal_buffer_byte_offset(output_double),
          /*length=*/iree_hal_buffer_byte_length(output_double),
      };
      negate_binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/nullptr,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(input_buffer),
      };
      negate_binding_refs[1] = {
          /*binding=*/1,
          /*buffer_slot=*/1,
          /*buffer=*/nullptr,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(output_negate),
      };
      double_binding_refs[0] = {
          /*binding=*/0,
          /*buffer_slot=*/0,
          /*buffer=*/nullptr,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(input_buffer),
      };
      double_binding_refs[1] = {
          /*binding=*/1,
          /*buffer_slot=*/2,
          /*buffer=*/nullptr,
          /*offset=*/0,
          /*length=*/iree_hal_buffer_byte_length(output_double),
      };
      break;
  }

  iree_hal_buffer_ref_list_t negate_bindings = {
      /*.count=*/IREE_ARRAYSIZE(negate_binding_refs),
      /*.values=*/negate_binding_refs,
  };
  iree_hal_buffer_ref_list_t double_bindings = {
      /*.count=*/IREE_ARRAYSIZE(double_binding_refs),
      /*.values=*/double_binding_refs,
  };

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_table.count, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  // Dispatch entry_point=0 (negate).
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), negate_bindings,
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

  // Dispatch entry_point=1 (double_it).
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/1,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), double_bindings,
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

  // Read back and verify negate output: [-1, -2, -3, -4].
  std::vector<int32_t> negate_data(4);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, output_negate, /*source_offset=*/0,
      /*target_buffer=*/negate_data.data(),
      /*data_length=*/negate_data.size() * sizeof(int32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(negate_data, ContainerEq(std::vector<int32_t>{-1, -2, -3, -4}));

  // Read back and verify double output: [2, 4, 6, 8].
  std::vector<uint32_t> double_data(4);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, output_double, /*source_offset=*/0,
      /*target_buffer=*/double_data.data(),
      /*data_length=*/double_data.size() * sizeof(uint32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  EXPECT_THAT(double_data, ContainerEq(std::vector<uint32_t>{2, 4, 6, 8}));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_double);
  iree_hal_buffer_release(output_negate);
  iree_hal_buffer_release(input_buffer);
}

CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(DispatchMultiEntrypointTest);

}  // namespace iree::hal::cts
