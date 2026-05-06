// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanQueueDescriptorCacheTest : public CtsTestBase<> {};

TEST_P(VulkanQueueDescriptorCacheTest, DeferredUnalignedFillsExceedOneBlock) {
  constexpr iree_host_size_t kSubmissionCount = 5000;
  constexpr uint8_t kPattern = 0x5A;

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kSubmissionCount, target_buffer.out()));

  SemaphoreList gate(device_, {0}, {1});
  std::vector<uint64_t> initial_values(kSubmissionCount, 0);
  std::vector<uint64_t> payload_values(kSubmissionCount, 1);
  SemaphoreList signals(device_, std::move(initial_values),
                        std::move(payload_values));

  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &signals.semaphores[i],
        .payload_values = &signals.payload_values[i],
    };
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, gate, signal_list,
        target_buffer.get(), i, /*length=*/1, &kPattern, sizeof(kPattern),
        IREE_HAL_FILL_FLAG_NONE));
  }

  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate.semaphores[0], 1, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint8_t> bytes =
      ReadBufferBytes(target_buffer.get(), /*offset=*/0, kSubmissionCount);
  ASSERT_EQ(kSubmissionCount, bytes.size());
  for (uint8_t byte : bytes) {
    EXPECT_EQ(kPattern, byte);
  }
}

CTS_REGISTER_TEST_SUITE(VulkanQueueDescriptorCacheTest);

}  // namespace iree::hal::cts
