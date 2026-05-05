// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanAllocatorGuardrailTest : public CtsTestBase<> {};

TEST_P(VulkanAllocatorGuardrailTest, QueueAllocaRejectsSparseSizedAllocation) {
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

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_semaphore_list_empty(), /*pool=*/NULL, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME, &buffer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  iree_hal_buffer_release(buffer);
}

CTS_REGISTER_TEST_SUITE(VulkanAllocatorGuardrailTest);

}  // namespace iree::hal::cts
