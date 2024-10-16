// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_DISPATCH_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_DISPATCH_TEST_H_

#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer_view_util.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::cts {
using ::testing::ContainerEq;

class DeviceGroupCommandBufferDispatchTest : public CTSTestBase<> {
 protected:
  void PrepareAbsExecutable() {
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"),
        iree_loop_inline(&loop_status_), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(get_test_executable_format());
    executable_params.executable_data = get_test_executable_data(
        iree_make_cstring_view("command_buffer_dispatch_test.bin"));

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void CleanupExecutable() {
    iree_hal_executable_release(executable_);
    iree_hal_executable_cache_release(executable_cache_);
    IREE_ASSERT_OK(loop_status_);
  }

  iree_status_t loop_status_ = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache_ = NULL;
  iree_hal_executable_t* executable_ = NULL;
};

// Dispatches absf(x) on a two subranges (elements 8-9 and 16-17) of a 32
// element input buffer. input_buffer  = [-2.5 -2.5 -2.5 -2.5, ...]
// output_buffer = [-9.0  -9.0, -9.0, -9.0, 2.5, 2.5, -9.0, ......, 2.5, 2.5]
TEST_F(DeviceGroupCommandBufferDispatchTest,
       DispatchSameCommandListInTwoPlaces) {
  PrepareAbsExecutable();

  // Create input buffer.
  iree_hal_buffer_t* input_buffer = NULL;
  CreateFilledDeviceBuffer<float>(32 * sizeof(float), -2.5f, &input_buffer);

  // Create output buffer.
  iree_hal_buffer_t* output_buffer = NULL;
  CreateFilledDeviceBuffer<float>(32 * sizeof(float), -9.0f, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_binding_t binding_table_values_gpu0[2];
  iree_hal_buffer_binding_t binding_table_values_gpu1[2];
  iree_hal_buffer_binding_table_t binding_table_gpu0{
      .count = IREE_ARRAYSIZE(binding_refs),
      .bindings = binding_table_values_gpu0};
  iree_hal_buffer_binding_table_t binding_table_gpu1{
      .count = IREE_ARRAYSIZE(binding_refs),
      .bindings = binding_table_values_gpu1};

  binding_refs[0] = {
      /*binding=*/0,
      /*buffer_slot=*/0,
      /*buffer=*/NULL,
      /*offset=*/0,
      /*length=*/2 * sizeof(float),
  };
  binding_refs[1] = {
      /*binding=*/1,
      /*buffer_slot=*/1,
      /*buffer=*/NULL,
      /*offset=*/0,
      /*length=*/2 * sizeof(float),
  };

  binding_table_values_gpu0[0] = {
      /*buffer=*/input_buffer,
      /*offset=*/8 * sizeof(float),
      /*length=*/2 * sizeof(float),
  };
  binding_table_values_gpu0[1] = {
      /*buffer=*/output_buffer,
      /*offset=*/8 * sizeof(float),
      /*length=*/2 * sizeof(float),
  };

  binding_table_values_gpu1[0] = {
      /*buffer=*/input_buffer,
      /*offset=*/16 * sizeof(float),
      /*length=*/2 * sizeof(float),
  };
  binding_table_values_gpu1[1] = {
      /*buffer=*/output_buffer,
      /*offset=*/16 * sizeof(float),
      /*length=*/2 * sizeof(float),
  };

  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_table_gpu0.count, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  uint32_t workgroup_count[3] = {1, 1, 1};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0, workgroup_count,
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH |
          IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
      /*memory_barriers=*/NULL,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/NULL));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // One signal semaphore from 0 -> 1.
  iree_hal_semaphore_t* gpu0_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &gpu0_semaphore));

  // One signal semaphore from 0 -> 1.
  iree_hal_semaphore_t* gpu1_semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device_, 0ull, IREE_HAL_SEMAPHORE_FLAG_NONE, &gpu1_semaphore));

  uint64_t target_payload_value = 1ull;
  iree_hal_semaphore_list_t gpu0_semaphore_list = {
      /*count=*/1,
      /*semaphores=*/&gpu0_semaphore,
      /*payload_values=*/&target_payload_value,
  };

  iree_hal_semaphore_list_t gpu1_semaphore_list = {
      /*count=*/1,
      /*semaphores=*/&gpu1_semaphore,
      /*payload_values=*/&target_payload_value,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, 0x1, iree_hal_semaphore_list_empty(), gpu0_semaphore_list, 1,
      &command_buffer, &binding_table_gpu0));

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, 0x2, iree_hal_semaphore_list_empty(), gpu1_semaphore_list, 1,
      &command_buffer, &binding_table_gpu1));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(gpu0_semaphore, target_payload_value,
                                         iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(gpu1_semaphore, target_payload_value,
                                         iree_infinite_timeout()));

  std::vector<float> output_values(32);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, output_buffer,
      /*source_offset=*/0, output_values.data(),
      sizeof(float) * output_values.size(),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  std::vector<float> reference(32);
  for (size_t i = 0; i < 32; ++i) {
    if (i >= 8 && i <= 9 || i >= 16 && i <= 17) {
      reference[i] = 2.5;
      continue;
    }
    reference[i] = -9.0;
  }
  EXPECT_THAT(output_values, ContainerEq(reference));

  iree_hal_semaphore_release(gpu0_semaphore);
  iree_hal_semaphore_release(gpu1_semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_release(input_buffer);
  CleanupExecutable();
}

}  // namespace iree::hal::cts

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_DISPATCH_TEST_H_
