// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanAllocatorGuardrailTest : public CtsTestBase<> {};

namespace {

constexpr iree_device_size_t kMaxSparseProbeAllocationSize =
    64ull * 1024ull * 1024ull;

}  // namespace

TEST_P(VulkanAllocatorGuardrailTest, QueueAllocaAcceptsSparseSizedAllocation) {
  iree_host_size_t heap_count = 0;
  iree_status_t count_status = iree_hal_allocator_query_memory_heaps(
      device_allocator_, /*capacity=*/0, /*heaps=*/NULL, &heap_count);
  if (iree_status_is_out_of_range(count_status)) {
    iree_status_free(count_status);
  } else {
    IREE_ASSERT_OK(count_status);
  }
  ASSERT_NE(0u, heap_count);

  std::vector<iree_hal_allocator_memory_heap_t> heaps(heap_count);
  IREE_ASSERT_OK(iree_hal_allocator_query_memory_heaps(
      device_allocator_, heaps.size(), heaps.data(), &heap_count));
  heaps.resize(heap_count);

  iree_hal_buffer_params_t params = {0};
  iree_device_size_t allocation_size = 0;
  for (const auto& heap : heaps) {
    if (!iree_all_bits_set(heap.allowed_usage,
                           IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      continue;
    }
    if (iree_device_size_checked_add(heap.max_allocation_size, 1,
                                     &allocation_size)) {
      params.type = heap.type;
      params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
      break;
    }
  }
  if (allocation_size == 0) {
    GTEST_SKIP() << "No finite Vulkan allocation limit to probe";
  }
  if (allocation_size > kMaxSparseProbeAllocationSize) {
    GTEST_SKIP() << "Sparse queue_alloca probe would allocate "
                 << allocation_size << " bytes";
  }

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      alloca_signal, /*pool=*/NULL, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &raw_buffer));
  buffer.reset(raw_buffer);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const uint32_t pattern = 0x1234CAFEu;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal,
      buffer.get(), /*target_offset=*/0, sizeof(pattern), &pattern,
      sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint8_t> data =
      ReadBufferBytes(buffer.get(), /*offset=*/0, sizeof(pattern));
  uint32_t readback = 0;
  memcpy(&readback, data.data(), sizeof(readback));
  EXPECT_EQ(pattern, readback);

  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_signal, dealloca_signal,
      buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

CTS_REGISTER_TEST_SUITE(VulkanAllocatorGuardrailTest);

}  // namespace iree::hal::cts
