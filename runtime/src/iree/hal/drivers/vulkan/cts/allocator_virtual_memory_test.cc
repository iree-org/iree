// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanVirtualMemoryTest : public CtsTestBase<> {};

static iree_hal_buffer_params_t DeviceLocalVirtualMemoryParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return params;
}

static iree_device_size_t QueryRecommendedPageSize(
    iree_hal_allocator_t* allocator, iree_hal_buffer_params_t params) {
  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_EXPECT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, params, &minimum_page_size, &recommended_page_size));
  EXPECT_NE(0u, minimum_page_size);
  EXPECT_NE(0u, recommended_page_size);
  EXPECT_GE(recommended_page_size, minimum_page_size);
  return recommended_page_size;
}

class VirtualBufferRef {
 public:
  explicit VirtualBufferRef(iree_hal_allocator_t* allocator)
      : allocator_(allocator) {}
  ~VirtualBufferRef() { reset(); }

  VirtualBufferRef(const VirtualBufferRef&) = delete;
  VirtualBufferRef& operator=(const VirtualBufferRef&) = delete;

  iree_hal_buffer_t* get() const { return buffer_; }
  iree_hal_buffer_t** out() { return &buffer_; }

  void reset() {
    if (buffer_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_virtual_memory_release(allocator_, buffer_));
      buffer_ = nullptr;
    }
  }

 private:
  iree_hal_allocator_t* allocator_ = nullptr;
  iree_hal_buffer_t* buffer_ = nullptr;
};

class PhysicalMemoryRef {
 public:
  explicit PhysicalMemoryRef(iree_hal_allocator_t* allocator)
      : allocator_(allocator) {}
  ~PhysicalMemoryRef() { reset(); }

  PhysicalMemoryRef(const PhysicalMemoryRef&) = delete;
  PhysicalMemoryRef& operator=(const PhysicalMemoryRef&) = delete;

  iree_hal_physical_memory_t* get() const { return memory_; }
  iree_hal_physical_memory_t** out() { return &memory_; }

  void reset() {
    if (memory_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_physical_memory_free(allocator_, memory_));
      memory_ = nullptr;
    }
  }

 private:
  iree_hal_allocator_t* allocator_ = nullptr;
  iree_hal_physical_memory_t* memory_ = nullptr;
};

TEST_P(VulkanVirtualMemoryTest, ReserveMapTransferUnmap) {
  if (!iree_hal_allocator_supports_virtual_memory(device_allocator_)) {
    GTEST_SKIP() << "Vulkan sparse virtual memory is not available";
  }

  iree_hal_buffer_params_t params = DeviceLocalVirtualMemoryParams();
  iree_device_size_t recommended_page_size =
      QueryRecommendedPageSize(device_allocator_, params);
  ASSERT_NE(0u, recommended_page_size);

  VirtualBufferRef virtual_buffer(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      device_allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, recommended_page_size,
      virtual_buffer.out()));
  ASSERT_TRUE(virtual_buffer.get());
  EXPECT_EQ(recommended_page_size,
            iree_hal_buffer_byte_length(virtual_buffer.get()));

  PhysicalMemoryRef physical_memory(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      device_allocator_, params, recommended_page_size, iree_allocator_system(),
      physical_memory.out()));
  ASSERT_TRUE(physical_memory.get());

  iree_status_t protect_status = iree_hal_allocator_virtual_memory_protect(
      device_allocator_, virtual_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, protect_status);

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      device_allocator_, virtual_buffer.get(), /*virtual_offset=*/0,
      physical_memory.get(), /*physical_offset=*/0, recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      device_allocator_, virtual_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));

  constexpr iree_device_size_t kTouchedSize = 256;
  ASSERT_GE(recommended_page_size, kTouchedSize);
  const uint32_t pattern = 0x7A6B5C4Du;
  SemaphoreList empty_wait;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal,
      virtual_buffer.get(), /*target_offset=*/0, kTouchedSize, &pattern,
      sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_buffer_t> readback_buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kTouchedSize, readback_buffer.out()));
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_signal, copy_signal,
      virtual_buffer.get(), /*source_offset=*/0, readback_buffer.get(),
      /*target_offset=*/0, kTouchedSize, IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint8_t> bytes =
      ReadBufferBytes(readback_buffer.get(), /*offset=*/0, kTouchedSize);
  ASSERT_EQ(kTouchedSize, bytes.size());
  for (iree_host_size_t i = 0; i < bytes.size(); i += sizeof(pattern)) {
    uint32_t value = 0;
    memcpy(&value, bytes.data() + i, sizeof(value));
    EXPECT_EQ(pattern, value);
  }

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, virtual_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size));
}

TEST_P(VulkanVirtualMemoryTest, MappingsMustBeUnmappedBeforeReleaseOrFree) {
  if (!iree_hal_allocator_supports_virtual_memory(device_allocator_)) {
    GTEST_SKIP() << "Vulkan sparse virtual memory is not available";
  }

  iree_hal_buffer_params_t params = DeviceLocalVirtualMemoryParams();
  iree_device_size_t recommended_page_size =
      QueryRecommendedPageSize(device_allocator_, params);
  ASSERT_NE(0u, recommended_page_size);

  iree_device_size_t reservation_size = 0;
  ASSERT_TRUE(iree_device_size_checked_mul(recommended_page_size, 3,
                                           &reservation_size));

  VirtualBufferRef virtual_buffer(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      device_allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, reservation_size,
      virtual_buffer.out()));
  ASSERT_TRUE(virtual_buffer.get());

  PhysicalMemoryRef physical_memory(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      device_allocator_, params, reservation_size, iree_allocator_system(),
      physical_memory.out()));
  ASSERT_TRUE(physical_memory.get());

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      device_allocator_, virtual_buffer.get(), /*virtual_offset=*/0,
      physical_memory.get(), /*physical_offset=*/0, reservation_size));

  iree_status_t release_status = iree_hal_allocator_virtual_memory_release(
      device_allocator_, virtual_buffer.get());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, release_status);

  iree_status_t free_status = iree_hal_allocator_physical_memory_free(
      device_allocator_, physical_memory.get());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, free_status);

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, virtual_buffer.get(), recommended_page_size,
      recommended_page_size));

  free_status = iree_hal_allocator_physical_memory_free(device_allocator_,
                                                        physical_memory.get());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, free_status);

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, virtual_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, virtual_buffer.get(),
      /*virtual_offset=*/2 * recommended_page_size, recommended_page_size));
}

CTS_REGISTER_TEST_SUITE(VulkanVirtualMemoryTest);

}  // namespace iree::hal::cts
