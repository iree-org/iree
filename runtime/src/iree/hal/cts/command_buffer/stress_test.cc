// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stress tests for rapid repeated command buffer submission.
//
// These tests exercise the recording item pool cycling and multi-worker
// drainer protocol under sustained load. The iteration counts are chosen
// to trigger races that only manifest after many rapid submit/complete
// cycles — the kind of workload benchmark tools produce.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::Each;
using ::testing::ElementsAreArray;

class CommandBufferStressTest : public CtsTestBase<> {};

// Submits 200 one-shot fill command buffers in a tight loop, each writing
// a different pattern. This stresses the recording item pool: each submit
// acquires a pool item, workers drain it, the item is released back to the
// pool, and the next submit reuses it. With 3+ workers, the drainer
// protocol's generation-tagged atomics are exercised under contention.
TEST_P(CommandBufferStressTest, RapidFillSubmit) {
  const iree_device_size_t buffer_size = 4096;
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, buffer.out()));

  for (int i = 0; i < 200; ++i) {
    uint32_t pattern = (uint32_t)(0xBEEF0000 | i);

    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, iree_hal_make_buffer_ref(buffer, 0, buffer_size),
        &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer))
        << "iteration " << i;
  }

  // Verify the last pattern was written.
  auto data = ReadBufferData<uint32_t>(buffer);
  uint32_t expected = 0xBEEF0000 | 199;
  EXPECT_THAT(data, Each(expected));
}

// Same as RapidFillSubmit but with a copy operation that uses both source
// and target buffers, exercising the binding resolution path.
TEST_P(CommandBufferStressTest, RapidCopySubmit) {
  const iree_device_size_t buffer_size = 4096;
  Ref<iree_hal_buffer_t> source;
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, source.out()));
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, target.out()));

  for (int i = 0; i < 200; ++i) {
    // Fill source with pattern via host mapping.
    uint32_t pattern = (uint32_t)(0xCAFE0000 | i);
    {
      iree_hal_buffer_mapping_t mapping;
      IREE_ASSERT_OK(iree_hal_buffer_map_range(
          source, IREE_HAL_MAPPING_MODE_SCOPED,
          IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0, buffer_size, &mapping));
      uint32_t* data = (uint32_t*)mapping.contents.data;
      for (iree_device_size_t j = 0; j < buffer_size / sizeof(uint32_t); ++j) {
        data[j] = pattern;
      }
      IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
    }

    // Copy source→target via command buffer.
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        command_buffer, iree_hal_make_buffer_ref(source, 0, buffer_size),
        iree_hal_make_buffer_ref(target, 0, buffer_size),
        IREE_HAL_COPY_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer))
        << "iteration " << i;
  }

  auto data = ReadBufferData<uint32_t>(target);
  uint32_t expected = 0xCAFE0000 | 199;
  EXPECT_THAT(data, Each(expected));
}

// Records enough packet-producing commands in one reusable command buffer to
// force block splitting in drivers with bounded recording blocks. The public
// behavior is still one queue_execute whose signal is published only after the
// final RETURN is reached.
TEST_P(CommandBufferStressTest, LargeFillCommandBuffer) {
  static constexpr uint32_t kFillCount = 512;
  const iree_device_size_t buffer_size = kFillCount * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(buffer_size, buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  std::vector<uint32_t> expected(kFillCount);
  for (uint32_t i = 0; i < kFillCount; ++i) {
    expected[i] = 0xBD370000u | i;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer,
        iree_hal_make_buffer_ref(buffer, i * sizeof(uint32_t),
                                 sizeof(uint32_t)),
        &expected[i], sizeof(expected[i]), IREE_HAL_FILL_FLAG_NONE));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  auto data = ReadBufferData<uint32_t>(buffer);
  EXPECT_THAT(data, ElementsAreArray(expected));
}

CTS_REGISTER_TEST_SUITE(CommandBufferStressTest);

}  // namespace iree::hal::cts
