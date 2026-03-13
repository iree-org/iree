// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <thread>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

// RAII wrapper for iree_hal_semaphore_list_t that manages retain/release.
// Duplicated from queue_host_call_test.cc — could be factored into a shared
// header if more queue tests need it.
struct SemaphoreList {
  SemaphoreList() = default;
  SemaphoreList(iree_hal_device_t* device, std::vector<uint64_t> initial_values,
                std::vector<uint64_t> desired_values) {
    for (size_t i = 0; i < initial_values.size(); ++i) {
      iree_hal_semaphore_t* semaphore = NULL;
      IREE_EXPECT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_AFFINITY_ANY, initial_values[i],
          IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore));
      semaphores.push_back(semaphore);
    }
    payload_values = desired_values;
    assert(semaphores.size() == payload_values.size());
  }

  SemaphoreList(const iree_hal_semaphore_list_t& list) {
    semaphores.reserve(list.count);
    payload_values.reserve(list.count);
    for (iree_host_size_t i = 0; i < list.count; ++i) {
      semaphores.push_back(list.semaphores[i]);
      payload_values.push_back(list.payload_values[i]);
    }
    iree_hal_semaphore_list_retain(*this);
  }

  SemaphoreList(const SemaphoreList& other) {
    semaphores = other.semaphores;
    payload_values = other.payload_values;
    iree_hal_semaphore_list_retain(*this);
  }

  SemaphoreList& operator=(const SemaphoreList& other) {
    if (this != &other) {
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      semaphores = other.semaphores;
      payload_values = other.payload_values;
      iree_hal_semaphore_list_retain(*this);
    }
    return *this;
  }

  SemaphoreList(SemaphoreList&& other) noexcept
      : semaphores(std::move(other.semaphores)),
        payload_values(std::move(other.payload_values)) {
    other.semaphores.clear();
    other.payload_values.clear();
  }

  SemaphoreList& operator=(SemaphoreList&& other) noexcept {
    if (this != &other) {
      iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
      semaphores = std::move(other.semaphores);
      payload_values = std::move(other.payload_values);
      other.semaphores.clear();
      other.payload_values.clear();
    }
    return *this;
  }

  ~SemaphoreList() {
    iree_hal_semaphore_list_release((iree_hal_semaphore_list_t)(*this));
  }

  operator iree_hal_semaphore_list_t() {
    iree_hal_semaphore_list_t list;
    list.count = semaphores.size();
    list.semaphores = semaphores.data();
    list.payload_values = payload_values.data();
    return list;
  }

  std::vector<iree_hal_semaphore_t*> semaphores;
  std::vector<uint64_t> payload_values;
};

class QueueAllocaTest : public CtsTestBase<> {};

// Allocates a buffer with no wait semaphores. Verifies the buffer is returned
// immediately and can be used after the signal semaphore fires.
TEST_P(QueueAllocaTest, BasicAlloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 1024;

  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT, params,
      allocation_size, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // Wait for the allocation to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify buffer metadata.
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), allocation_size);

  // Write a pattern and read it back to verify the buffer is usable.
  uint8_t pattern = 0xAB;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, allocation_size, &pattern,
                                          sizeof(pattern)));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0,
                                           allocation_size, &mapping));

  for (iree_device_size_t i = 0; i < allocation_size; ++i) {
    ASSERT_EQ(mapping.contents.data[i], 0xAB) << "Mismatch at byte " << i;
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(buffer);
}

// Allocates a buffer that waits on a semaphore before committing. Signals the
// wait semaphore from a background thread to verify the async path.
TEST_P(QueueAllocaTest, AllocaWithWaitSemaphores) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 512;

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT, params,
      allocation_size, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // Signal the wait semaphore from a background thread after a short delay.
  std::thread waker([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    IREE_EXPECT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list));
  });

  // Wait for the allocation to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify the buffer is usable after the signal.
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), allocation_size);

  uint8_t pattern = 0xCD;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, allocation_size, &pattern,
                                          sizeof(pattern)));

  waker.join();
  iree_hal_buffer_release(buffer);
}

// Verifies the full alloca → use → dealloca lifecycle.
TEST_P(QueueAllocaTest, AllocaDeallocaCycle) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 2048;

  // Phase 1: alloca → fill → dealloca.
  {
    SemaphoreList alloca_signal(device_, {0}, {1});
    SemaphoreList dealloca_signal(device_, {0}, {1});

    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

    iree_hal_buffer_t* buffer = NULL;
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, allocation_size,
        IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
    ASSERT_NE(buffer, nullptr);

    // Wait for alloca, fill the buffer.
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        alloca_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

    uint32_t pattern = 0xDEADBEEF;
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, allocation_size,
                                            &pattern, sizeof(pattern)));

    // Verify the fill.
    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
        sizeof(uint32_t), &mapping));
    uint32_t readback = 0;
    memcpy(&readback, mapping.contents.data, sizeof(readback));
    EXPECT_EQ(readback, 0xDEADBEEF);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

    // Dealloca: wait on alloca completion, signal dealloca completion.
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
        buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(dealloca_signal,
                                                iree_make_timeout_ms(5000),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_release(buffer);
  }

  // Phase 2: alloca again to verify the system handles repeated cycles.
  {
    SemaphoreList alloca_signal(device_, {0}, {1});
    SemaphoreList dealloca_signal(device_, {0}, {1});

    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

    iree_hal_buffer_t* buffer = NULL;
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, allocation_size,
        IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
    ASSERT_NE(buffer, nullptr);

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        alloca_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

    uint32_t pattern = 0xCAFEBABE;
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, allocation_size,
                                            &pattern, sizeof(pattern)));

    iree_hal_buffer_mapping_t mapping;
    IREE_ASSERT_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
        sizeof(uint32_t), &mapping));
    uint32_t readback = 0;
    memcpy(&readback, mapping.contents.data, sizeof(readback));
    EXPECT_EQ(readback, 0xCAFEBABE);
    IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
        buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(dealloca_signal,
                                                iree_make_timeout_ms(5000),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_release(buffer);
  }
}

// Verifies returned buffer metadata matches the requested parameters.
TEST_P(QueueAllocaTest, BufferMetadata) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, IREE_HAL_ALLOCATOR_POOL_DEFAULT, params,
      allocation_size, IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // Buffer metadata is available immediately (before the backing is committed).
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), allocation_size);
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                IREE_HAL_BUFFER_USAGE_MAPPING));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE));

  // The buffer must have the ASYNCHRONOUS placement flag for dealloca routing.
  EXPECT_TRUE(
      iree_all_bits_set(iree_hal_buffer_allocation_placement(buffer).flags,
                        IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS));

  // Wait for completion and clean up.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_make_timeout_ms(5000),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Dealloca to clean up the transient buffer.
  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, signal_semaphore_list,
      dealloca_signal, buffer, IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
}

// Verifies that accessing a decommitted buffer (after dealloca) fails with
// FAILED_PRECONDITION. This validates the transient buffer's decommit behavior.
TEST_P(QueueAllocaTest, DeallocaReleasesMemory) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 256;

  SemaphoreList alloca_signal(device_, {0}, {1});
  SemaphoreList dealloca_signal(device_, {0}, {1});

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;

  SemaphoreList empty_wait;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify the buffer works before dealloca.
  uint8_t pattern = 0xFF;
  IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, allocation_size, &pattern,
                                          sizeof(pattern)));

  // Dealloca: wait on alloca completion.
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
      buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  // Accessing the buffer after dealloca must fail. The transient buffer's
  // backing has been decommitted — any map attempt should return
  // FAILED_PRECONDITION.
  iree_hal_buffer_mapping_t mapping;
  EXPECT_THAT(Status(iree_hal_buffer_map_range(
                  buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                  IREE_HAL_MEMORY_ACCESS_READ, 0, allocation_size, &mapping)),
              StatusIs(StatusCode::kFailedPrecondition));

  iree_hal_buffer_release(buffer);
}

// Chains alloca → dealloca via semaphores without explicit host-side waits
// between them. Verifies the queue ordering handles the dependency correctly.
TEST_P(QueueAllocaTest, ChainedAllocaDealloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 1024;

  // Semaphore 0→1: alloca signal / dealloca wait.
  // Semaphore 0→1: dealloca signal.
  SemaphoreList alloca_signal(device_, {0}, {1});
  SemaphoreList dealloca_signal(device_, {0}, {1});

  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_MAPPING |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

  SemaphoreList empty_wait;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // Chain dealloca to wait on alloca completion — no host wait in between.
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
      buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

  // Wait for the entire chain to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
}

CTS_REGISTER_TEST_SUITE(QueueAllocaTest);

}  // namespace iree::hal::cts
