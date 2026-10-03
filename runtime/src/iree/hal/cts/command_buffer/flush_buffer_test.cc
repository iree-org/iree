// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class CommandBufferFlushBufferTest : public CtsTestBase<> {
 protected:
  // Fills the whole buffer with |pattern| and then flushes the range
  // [flush_offset, flush_offset + flush_length) using the current recording
  // mode. Flushes only perform cache maintenance so the buffer contents must be
  // left untouched by them.
  // In direct mode: uses inline buffer references (binding_capacity = 0).
  // In indirect mode: uses binding table slots (binding_capacity = 1).
  // Results are written to |out_data|, which is resized to |buffer_size|.
  void RunFillThenFlushTest(iree_device_size_t buffer_size, uint32_t pattern,
                            iree_device_size_t flush_offset,
                            iree_device_size_t flush_length,
                            std::vector<uint8_t>& out_data) {
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, &device_buffer));

    const bool indirect = recording_mode() == RecordingMode::kIndirect;
    const iree_host_size_t binding_capacity = indirect ? 1 : 0;

    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

    iree_hal_buffer_ref_t fill_ref;
    iree_hal_buffer_ref_t flush_ref;
    if (indirect) {
      fill_ref = iree_hal_make_indirect_buffer_ref(/*binding=*/0, 0,
                                                   buffer_size);
      flush_ref = iree_hal_make_indirect_buffer_ref(/*binding=*/0,
                                                    flush_offset, flush_length);
    } else {
      fill_ref = iree_hal_make_buffer_ref(device_buffer, 0, buffer_size);
      flush_ref =
          iree_hal_make_buffer_ref(device_buffer, flush_offset, flush_length);
    }

    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, fill_ref, &pattern, sizeof(pattern),
        IREE_HAL_FILL_FLAG_NONE));
    // Flushes do not establish execution dependencies so the fill must be
    // ordered before the flush explicitly.
    IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
        command_buffer,
        /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_TRANSFER |
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
            IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
        /*memory_barriers=*/nullptr,
        /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
    IREE_ASSERT_OK(
        iree_hal_command_buffer_flush_buffer(command_buffer, flush_ref));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    if (indirect) {
      const iree_hal_buffer_binding_t bindings[] = {
          {device_buffer, 0, IREE_HAL_WHOLE_BUFFER},
      };
      IREE_ASSERT_OK(SubmitCommandBufferAndWait(
          command_buffer,
          iree_hal_buffer_binding_table_t{IREE_ARRAYSIZE(bindings), bindings}));
    } else {
      IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
    }

    out_data.resize(buffer_size);
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device_, device_buffer, /*source_offset=*/0, out_data.data(),
        buffer_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(device_buffer);
  }
};

TEST_P(CommandBufferFlushBufferTest, FlushWholeBuffer) {
  const iree_device_size_t buffer_size = 16;
  const uint32_t pattern = 0xCAFEF00Du;
  std::vector<uint8_t> actual_data;
  RunFillThenFlushTest(buffer_size, pattern, /*flush_offset=*/0,
                       /*flush_length=*/buffer_size, actual_data);
  EXPECT_THAT(actual_data,
              ContainerEq(MakeFilledBytes(buffer_size, 0, buffer_size, pattern,
                                          sizeof(pattern))));
}

TEST_P(CommandBufferFlushBufferTest, FlushSubrange) {
  const iree_device_size_t buffer_size = 32;
  const uint32_t pattern = 0x12345678u;
  std::vector<uint8_t> actual_data;
  RunFillThenFlushTest(buffer_size, pattern, /*flush_offset=*/8,
                       /*flush_length=*/12, actual_data);
  EXPECT_THAT(actual_data,
              ContainerEq(MakeFilledBytes(buffer_size, 0, buffer_size, pattern,
                                          sizeof(pattern))));
}

TEST_P(CommandBufferFlushBufferTest, FlushZeroLength) {
  const iree_device_size_t buffer_size = 16;
  const uint32_t pattern = 0xA5A55A5Au;
  std::vector<uint8_t> actual_data;
  RunFillThenFlushTest(buffer_size, pattern, /*flush_offset=*/4,
                       /*flush_length=*/0, actual_data);
  EXPECT_THAT(actual_data,
              ContainerEq(MakeFilledBytes(buffer_size, 0, buffer_size, pattern,
                                          sizeof(pattern))));
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferFlushBufferTest);

}  // namespace iree::hal::cts
