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
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

class command_buffer_dispatch_test : public CtsTestBase {
 protected:
  void PrepareAbsExecutable() {
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] =
        {
            {0, IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {1, IREE_HAL_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        };
    IREE_ASSERT_OK(iree_hal_descriptor_set_layout_create(
        device_, IREE_HAL_DESCRIPTOR_SET_LAYOUT_USAGE_TYPE_PUSH_ONLY,
        IREE_ARRAYSIZE(descriptor_set_layout_bindings),
        descriptor_set_layout_bindings, &descriptor_set_layout_));
    IREE_ASSERT_OK(iree_hal_executable_layout_create(
        device_, /*push_constants=*/0, /*set_layout_count=*/1,
        &descriptor_set_layout_, &executable_layout_));

    iree_hal_executable_params_t executable_params;
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(get_test_executable_format());
    executable_params.executable_data = get_test_executable_data(
        iree_make_cstring_view("command_buffer_dispatch_test.bin"));
    executable_params.executable_layout_count = 1;
    executable_params.executable_layouts = &executable_layout_;

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void CleanupExecutable() {
    iree_hal_executable_release(executable_);
    iree_hal_executable_layout_release(executable_layout_);
    iree_hal_descriptor_set_layout_release(descriptor_set_layout_);
    iree_hal_executable_cache_release(executable_cache_);
  }

  iree_hal_executable_cache_t* executable_cache_ = NULL;
  iree_hal_descriptor_set_layout_t* descriptor_set_layout_ = NULL;
  iree_hal_executable_layout_t* executable_layout_ = NULL;
  iree_hal_executable_t* executable_ = NULL;
};

TEST_P(command_buffer_dispatch_test, DispatchAbs) {
  PrepareAbsExecutable();

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  // Create input and output buffers.
  iree_hal_buffer_params_t input_params = {0};
  input_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  input_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_view_t* input_buffer_view = NULL;
  float input_data[1] = {-2.5f};
  IREE_ASSERT_OK(iree_hal_buffer_view_allocate_buffer(
      device_allocator_, /*shape=*/NULL,
      /*shape_rank=*/0, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, input_params,
      iree_make_const_byte_span((void*)input_data, sizeof(input_data)),
      &input_buffer_view));
  iree_hal_buffer_params_t output_params = {0};
  output_params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  output_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* output_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, output_params, sizeof(float),
      iree_const_byte_span_empty(), &output_buffer));

  iree_hal_descriptor_set_binding_t descriptor_set_bindings[] = {
      {/*binding=*/0, iree_hal_buffer_view_buffer(input_buffer_view),
       /*offset=*/0, iree_hal_buffer_view_byte_length(input_buffer_view)},
      {/*binding=*/1, output_buffer, iree_hal_buffer_byte_offset(output_buffer),
       iree_hal_buffer_byte_length(output_buffer)},
  };

  IREE_ASSERT_OK(iree_hal_command_buffer_push_descriptor_set(
      command_buffer, executable_layout_, /*set=*/0,
      IREE_ARRAYSIZE(descriptor_set_bindings), descriptor_set_bindings));

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

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                            command_buffer));

  float out_value = 0.0f;
  IREE_ASSERT_OK(iree_hal_buffer_read_data(output_buffer, /*source_offset=*/0,
                                           &out_value, sizeof(out_value)));
  EXPECT_EQ(2.5f, out_value);

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_buffer);
  iree_hal_buffer_view_release(input_buffer_view);
  CleanupExecutable();
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_DISPATCH_TEST_H_
