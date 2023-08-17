// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_PUSH_CONSTANTS_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_PUSH_CONSTANTS_TEST_H_

#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

using ::testing::ContainerEq;

class command_buffer_push_constants_test : public CtsTestBase {
 protected:
  void PrepareExecutable() {
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"),
        iree_loop_inline(&loop_status_), &executable_cache_));

    iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] =
        {
            {
                0,
                IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                IREE_HAL_DESCRIPTOR_FLAG_NONE,
            },
        };
    IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
        device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_FLAG_NONE,
        IREE_ARRAYSIZE(descriptor_set_layout_bindings),
        descriptor_set_layout_bindings, &descriptor_set_layout_));
    IREE_ASSERT_OK(iree_hal_pipeline_layout_create(
        device_, /*push_constants=*/4, /*set_layout_count=*/1,
        &descriptor_set_layout_, &pipeline_layout_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(get_test_executable_format());
    executable_params.executable_data = get_test_executable_data(
        iree_make_cstring_view("command_buffer_push_constants_test.bin"));
    executable_params.pipeline_layout_count = 1;
    executable_params.pipeline_layouts = &pipeline_layout_;
    // No executable-level "specialization constants" (not to be confused with
    // per-dispatch varying "push constants").
    executable_params.constant_count = 0;
    executable_params.constants = NULL;

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void CleanupExecutable() {
    iree_hal_executable_release(executable_);
    iree_hal_pipeline_layout_release(pipeline_layout_);
    iree_hal_descriptor_set_layout_release(descriptor_set_layout_);
    iree_hal_executable_cache_release(executable_cache_);
    IREE_ASSERT_OK(loop_status_);
  }

  iree_status_t loop_status_ = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache_ = NULL;
  iree_hal_descriptor_set_layout_t* descriptor_set_layout_ = NULL;
  iree_hal_pipeline_layout_t* pipeline_layout_ = NULL;
  iree_hal_executable_t* executable_ = NULL;
};

TEST_P(command_buffer_push_constants_test, DispatchWithPushConstants) {
  ASSERT_NO_FATAL_FAILURE(PrepareExecutable());

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  // Create output buffer.
  iree_hal_buffer_params_t output_params = {0};
  output_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  output_params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                        IREE_HAL_BUFFER_USAGE_TRANSFER |
                        IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* output_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, output_params, 4 * sizeof(uint32_t), &output_buffer));

  iree_hal_descriptor_set_binding_t descriptor_set_bindings[] = {
      {
          /*binding=*/0,
          /*buffer_slot=*/0,
          output_buffer,
          iree_hal_buffer_byte_offset(output_buffer),
          iree_hal_buffer_byte_length(output_buffer),
      },
  };

  IREE_ASSERT_OK(iree_hal_command_buffer_push_descriptor_set(
      command_buffer, pipeline_layout_, /*set=*/0,
      IREE_ARRAYSIZE(descriptor_set_bindings), descriptor_set_bindings));

  std::vector<uint32_t> push_constants{11, 22, 33, 44};
  IREE_ASSERT_OK(iree_hal_command_buffer_push_constants(
      command_buffer, pipeline_layout_, /*offset=*/0, push_constants.data(),
      push_constants.size() * sizeof(uint32_t)));

  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0,
      /*workgroup_x=*/1, /*workgroup_y=*/1, /*workgroup_z=*/1));
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

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  std::vector<uint32_t> output_data(4);
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device_, output_buffer, /*source_offset=*/0,
      /*target_buffer=*/output_data.data(),
      /*data_length=*/output_data.size() * sizeof(uint32_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  EXPECT_THAT(output_data, ContainerEq(push_constants));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  CleanupExecutable();
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_PUSH_CONSTANTS_TEST_H_
