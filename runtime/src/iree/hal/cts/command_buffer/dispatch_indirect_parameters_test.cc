// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL command buffer dispatches whose workgroup counts are read from
// indirect parameter buffers.

#include <cstdint>
#include <numeric>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class DispatchIndirectParametersTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kDispatchedWorkgroupCount = 4;
  static constexpr iree_host_size_t kOutputElementCount = 32;
  static constexpr iree_device_size_t kOutputByteLength =
      kOutputElementCount * sizeof(uint32_t);
  static constexpr iree_device_size_t kParameterByteLength =
      3 * sizeof(uint32_t);
  static constexpr uint32_t kSentinelValue = 0xCDCDCDCDu;

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    {
      iree_hal_executable_params_t params;
      iree_hal_executable_params_initialize(&params);
      params.caching_mode =
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
      params.executable_format = iree_make_cstring_view(executable_format());
      params.executable_data = executable_data(iree_make_cstring_view(
          "command_buffer_dispatch_multi_workgroup_test.bin"));
      IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
          executable_cache_, &params, &workgroup_id_executable_));
    }

    {
      iree_hal_executable_params_t params;
      iree_hal_executable_params_initialize(&params);
      params.caching_mode =
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
      params.executable_format = iree_make_cstring_view(executable_format());
      params.executable_data = executable_data(iree_make_cstring_view(
          "command_buffer_dispatch_indirect_parameters_test.bin"));
      IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
          executable_cache_, &params, &parameter_producer_executable_));
    }
  }

  void TearDown() override {
    iree_hal_executable_release(parameter_producer_executable_);
    parameter_producer_executable_ = nullptr;
    iree_hal_executable_release(workgroup_id_executable_);
    workgroup_id_executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  void CreateIndirectParameterBuffer(iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, kParameterByteLength, out_buffer));
  }

  void CreateSentinelOutputBuffer(iree_hal_buffer_t** out_buffer) {
    CreateFilledDeviceBuffer(kOutputByteLength, kSentinelValue, out_buffer);
  }

  iree_host_size_t binding_capacity() const {
    return recording_mode() == RecordingMode::kIndirect ? 2 : 0;
  }

  iree_hal_buffer_ref_t OutputRef(iree_hal_buffer_t* output_buffer) const {
    if (recording_mode() == RecordingMode::kIndirect) {
      return iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                               kOutputByteLength);
    }
    return iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                    kOutputByteLength);
  }

  iree_hal_buffer_ref_t ParameterRef(
      iree_hal_buffer_t* parameter_buffer,
      iree_device_size_t parameter_ref_length = kParameterByteLength) const {
    if (recording_mode() == RecordingMode::kIndirect) {
      return iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/0,
                                               parameter_ref_length);
    }
    return iree_hal_make_buffer_ref(parameter_buffer, /*offset=*/0,
                                    parameter_ref_length);
  }

  iree_hal_buffer_binding_table_t BindingTable(
      iree_hal_buffer_binding_t* bindings, iree_hal_buffer_t* output_buffer,
      iree_hal_buffer_t* parameter_buffer) const {
    if (recording_mode() == RecordingMode::kDirect) {
      return iree_hal_buffer_binding_table_empty();
    }
    bindings[0] = {
        /*buffer=*/output_buffer,
        /*offset=*/0,
        /*length=*/kOutputByteLength,
    };
    bindings[1] = {
        /*buffer=*/parameter_buffer,
        /*offset=*/0,
        /*length=*/kParameterByteLength,
    };
    return iree_hal_buffer_binding_table_t{
        /*count=*/2,
        /*bindings=*/bindings,
    };
  }

  iree_hal_dispatch_config_t IndirectDispatchConfig(
      iree_hal_buffer_t* parameter_buffer,
      iree_device_size_t parameter_ref_length = kParameterByteLength) const {
    iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
        /*workgroup_count_x=*/kOutputElementCount, /*workgroup_count_y=*/1,
        /*workgroup_count_z=*/1);
    config.workgroup_count_ref =
        ParameterRef(parameter_buffer, parameter_ref_length);
    return config;
  }

  void RecordWorkgroupIdDispatch(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_t* output_buffer, iree_hal_buffer_t* parameter_buffer,
      iree_hal_dispatch_flags_t flags,
      iree_device_size_t parameter_ref_length = kParameterByteLength) {
    iree_hal_buffer_ref_t binding_refs[1] = {OutputRef(output_buffer)};
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer, workgroup_id_executable_, /*entry_point=*/0,
        IndirectDispatchConfig(parameter_buffer, parameter_ref_length),
        iree_const_byte_span_empty(), bindings, flags));
  }

  void RecordFullExecutionBarrier(iree_hal_command_buffer_t* command_buffer) {
    IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
        command_buffer,
        /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH |
            IREE_HAL_EXECUTION_STAGE_TRANSFER |
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
            IREE_HAL_EXECUTION_STAGE_DISPATCH |
            IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
        /*memory_barriers=*/nullptr,
        /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  }

  void SubmitAndCheck(iree_hal_command_buffer_t* command_buffer,
                      iree_hal_buffer_t* output_buffer,
                      iree_hal_buffer_t* parameter_buffer) {
    iree_hal_buffer_binding_t binding_table_values[2];
    iree_hal_buffer_binding_table_t binding_table =
        BindingTable(binding_table_values, output_buffer, parameter_buffer);
    IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer, binding_table));

    std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
    std::vector<uint32_t> expected(kOutputElementCount, kSentinelValue);
    std::iota(expected.begin(), expected.begin() + kDispatchedWorkgroupCount,
              0u);
    EXPECT_THAT(output_data, ContainerEq(expected));
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* workgroup_id_executable_ = nullptr;
  iree_hal_executable_t* parameter_producer_executable_ = nullptr;
};

TEST_P(DispatchIndirectParametersTest, StaticParametersFromQueueUpdate) {
  Ref<iree_hal_buffer_t> output_buffer;
  CreateSentinelOutputBuffer(output_buffer.out());

  Ref<iree_hal_buffer_t> parameter_buffer;
  CreateIndirectParameterBuffer(parameter_buffer.out());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  RecordWorkgroupIdDispatch(command_buffer, output_buffer, parameter_buffer,
                            IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const uint32_t parameter_data[3] = {
      kDispatchedWorkgroupCount,
      1,
      1,
  };
  SemaphoreList update_signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, update_signal,
      parameter_data, /*source_offset=*/0, parameter_buffer,
      /*target_offset=*/0, sizeof(parameter_data), IREE_HAL_UPDATE_FLAG_NONE));

  iree_hal_buffer_binding_t binding_table_values[2];
  iree_hal_buffer_binding_table_t binding_table =
      BindingTable(binding_table_values, output_buffer, parameter_buffer);
  SemaphoreList execute_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, update_signal, execute_signal,
      command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  std::vector<uint32_t> expected(kOutputElementCount, kSentinelValue);
  std::iota(expected.begin(), expected.begin() + kDispatchedWorkgroupCount, 0u);
  EXPECT_THAT(output_data, ContainerEq(expected));
}

TEST_P(DispatchIndirectParametersTest, WholeBufferParameterRef) {
  Ref<iree_hal_buffer_t> output_buffer;
  CreateSentinelOutputBuffer(output_buffer.out());

  Ref<iree_hal_buffer_t> parameter_buffer;
  CreateIndirectParameterBuffer(parameter_buffer.out());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  RecordWorkgroupIdDispatch(command_buffer, output_buffer, parameter_buffer,
                            IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                            IREE_HAL_WHOLE_BUFFER);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const uint32_t parameter_data[3] = {
      kDispatchedWorkgroupCount,
      1,
      1,
  };
  SemaphoreList update_signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, update_signal,
      parameter_data, /*source_offset=*/0, parameter_buffer,
      /*target_offset=*/0, sizeof(parameter_data), IREE_HAL_UPDATE_FLAG_NONE));

  iree_hal_buffer_binding_t binding_table_values[2];
  iree_hal_buffer_binding_table_t binding_table =
      BindingTable(binding_table_values, output_buffer, parameter_buffer);
  SemaphoreList execute_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, update_signal, execute_signal,
      command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  std::vector<uint32_t> expected(kOutputElementCount, kSentinelValue);
  std::iota(expected.begin(), expected.begin() + kDispatchedWorkgroupCount, 0u);
  EXPECT_THAT(output_data, ContainerEq(expected));
}

TEST_P(DispatchIndirectParametersTest, DynamicParametersFromUpdate) {
  Ref<iree_hal_buffer_t> output_buffer;
  CreateSentinelOutputBuffer(output_buffer.out());

  Ref<iree_hal_buffer_t> parameter_buffer;
  CreateIndirectParameterBuffer(parameter_buffer.out());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t parameter_data[3] = {
      kDispatchedWorkgroupCount,
      1,
      1,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, parameter_data, /*source_offset=*/0,
      ParameterRef(parameter_buffer), IREE_HAL_UPDATE_FLAG_NONE));
  RecordFullExecutionBarrier(command_buffer);
  RecordWorkgroupIdDispatch(command_buffer, output_buffer, parameter_buffer,
                            IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SubmitAndCheck(command_buffer, output_buffer, parameter_buffer);
}

TEST_P(DispatchIndirectParametersTest, DynamicParametersFromFill) {
  Ref<iree_hal_buffer_t> output_buffer;
  CreateSentinelOutputBuffer(output_buffer.out());

  Ref<iree_hal_buffer_t> parameter_buffer;
  CreateIndirectParameterBuffer(parameter_buffer.out());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t one = 1;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, ParameterRef(parameter_buffer), &one, sizeof(one),
      IREE_HAL_FILL_FLAG_NONE));
  RecordFullExecutionBarrier(command_buffer);
  RecordWorkgroupIdDispatch(command_buffer, output_buffer, parameter_buffer,
                            IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_buffer_binding_t binding_table_values[2];
  iree_hal_buffer_binding_table_t binding_table =
      BindingTable(binding_table_values, output_buffer, parameter_buffer);
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer, binding_table));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  std::vector<uint32_t> expected(kOutputElementCount, kSentinelValue);
  expected[0] = 0;
  EXPECT_THAT(output_data, ContainerEq(expected));
}

TEST_P(DispatchIndirectParametersTest, DynamicParametersFromDispatch) {
  Ref<iree_hal_buffer_t> output_buffer;
  CreateSentinelOutputBuffer(output_buffer.out());

  Ref<iree_hal_buffer_t> parameter_buffer;
  CreateIndirectParameterBuffer(parameter_buffer.out());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity(), command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  {
    iree_hal_buffer_ref_t binding_refs[1] = {ParameterRef(parameter_buffer)};
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer, parameter_producer_executable_, /*entry_point=*/0,
        iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  }
  RecordFullExecutionBarrier(command_buffer);
  RecordWorkgroupIdDispatch(command_buffer, output_buffer, parameter_buffer,
                            IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SubmitAndCheck(command_buffer, output_buffer, parameter_buffer);
}

CTS_REGISTER_EXECUTABLE_COMMAND_BUFFER_TEST_SUITE(
    DispatchIndirectParametersTest);

}  // namespace iree::hal::cts
