// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/io/file_handle.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

namespace {

// Uses queue 0 explicitly so backend pool queries are scoped to one queue and
// one physical memory domain, even on multi-queue/multi-device backends.
constexpr iree_hal_queue_affinity_t kQueueAffinity0 =
    ((iree_hal_queue_affinity_t)1ull) << 0;
constexpr iree_hal_queue_affinity_t kQueueAffinity1 =
    ((iree_hal_queue_affinity_t)1ull) << 1;

iree_hal_buffer_params_t MakeQueueAllocaBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return params;
}

}  // namespace

class QueueAllocaTest : public CtsTestBase<> {
 protected:
  iree_status_t HasQueueAffinity(iree_hal_queue_affinity_t queue_affinity,
                                 bool* out_has_queue_affinity) {
    *out_has_queue_affinity = false;
    iree_hal_queue_pool_backend_t backend = {0};
    iree_status_t status = iree_hal_device_query_queue_pool_backend(
        device_, queue_affinity, &backend);
    if (iree_status_is_invalid_argument(status)) {
      iree_status_free(status);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(status);
    *out_has_queue_affinity = true;
    return iree_ok_status();
  }

  iree_status_t CreateExplicitPassthroughPool(iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {0};
    IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
        device_, kQueueAffinity0, &backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete backend bundle");
    }
    iree_hal_passthrough_pool_options_t options = {0};
    return iree_hal_passthrough_pool_create(options, backend.slab_provider,
                                            backend.notification,
                                            iree_allocator_system(), out_pool);
  }

  iree_status_t CreateExplicitFixedBlockPool(
      iree_device_size_t block_size, iree_hal_pool_epoch_query_t epoch_query,
      iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {0};
    IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
        device_, kQueueAffinity0, &backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete backend bundle");
    }
    iree_hal_fixed_block_pool_options_t options = {};
    options.block_allocator_options.block_size = block_size;
    options.block_allocator_options.block_count = 1;
    options.block_allocator_options.frontier_capacity = 2;
    return iree_hal_fixed_block_pool_create(options, backend.slab_provider,
                                            backend.notification, epoch_query,
                                            iree_allocator_system(), out_pool);
  }

  iree_status_t CreateExplicitTLSFPool(iree_device_size_t range_length,
                                       iree_hal_pool_epoch_query_t epoch_query,
                                       iree_hal_pool_t** out_pool) {
    iree_hal_queue_pool_backend_t backend = {0};
    IREE_RETURN_IF_ERROR(iree_hal_device_query_queue_pool_backend(
        device_, kQueueAffinity0, &backend));
    if (!backend.slab_provider || !backend.notification) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "queue pool backend query returned an incomplete backend bundle");
    }
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = range_length;
    options.tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
    options.tlsf_options.frontier_capacity = 2;
    return iree_hal_tlsf_pool_create(options, backend.slab_provider,
                                     backend.notification, epoch_query,
                                     iree_allocator_system(), out_pool);
  }

  void QueueWriteAndWait(iree_hal_queue_affinity_t queue_affinity,
                         iree_hal_buffer_t* source_buffer,
                         iree_device_size_t source_offset,
                         iree_hal_file_t* target_file, uint64_t target_offset,
                         iree_device_size_t length) {
    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, queue_affinity, empty_wait, write_signal, source_buffer,
        source_offset, target_file, target_offset, length,
        IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        write_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void FillBufferAndWait(iree_hal_queue_affinity_t queue_affinity,
                         iree_hal_buffer_t* target_buffer,
                         iree_device_size_t target_offset,
                         iree_device_size_t length, const void* pattern,
                         iree_host_size_t pattern_length) {
    SemaphoreList empty_wait;
    SemaphoreList fill_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, queue_affinity, empty_wait, fill_signal, target_buffer,
        target_offset, length, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  std::vector<uint8_t> ReadBufferBytes(iree_hal_queue_affinity_t queue_affinity,
                                       iree_hal_buffer_t* buffer,
                                       iree_device_size_t offset,
                                       iree_device_size_t length) {
    std::vector<uint8_t> data(length);
    if (iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
        iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED)) {
      IREE_EXPECT_OK(
          iree_hal_buffer_map_read(buffer, offset, data.data(), length));
      return data;
    }

    iree_io_file_handle_t* handle = NULL;
    IREE_EXPECT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(data.data(), length),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    if (!handle) return data;

    iree_hal_file_t* file = NULL;
    IREE_EXPECT_OK(iree_hal_file_import(
        device_, queue_affinity, IREE_HAL_MEMORY_ACCESS_WRITE, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
    iree_io_file_handle_release(handle);
    if (!file) return data;

    QueueWriteAndWait(queue_affinity, buffer, offset, file, 0, length);
    iree_hal_file_release(file);
    return data;
  }

  template <typename T>
  T ReadBufferValue(iree_hal_queue_affinity_t queue_affinity,
                    iree_hal_buffer_t* buffer, iree_device_size_t offset = 0) {
    T value = 0;
    std::vector<uint8_t> data =
        ReadBufferBytes(queue_affinity, buffer, offset, sizeof(T));
    if (data.size() == sizeof(T)) {
      memcpy(&value, data.data(), sizeof(T));
    }
    return value;
  }

  template <typename T>
  std::vector<T> ReadBufferData(iree_hal_queue_affinity_t queue_affinity,
                                iree_hal_buffer_t* buffer,
                                iree_device_size_t offset = 0) {
    iree_device_size_t byte_length =
        iree_hal_buffer_byte_length(buffer) - offset;
    std::vector<T> values(byte_length / sizeof(T));
    std::vector<uint8_t> data = ReadBufferBytes(queue_affinity, buffer, offset,
                                                values.size() * sizeof(T));
    if (data.size() == values.size() * sizeof(T)) {
      memcpy(values.data(), data.data(), data.size());
    }
    return values;
  }
};

// Allocates a buffer with no wait semaphores. Verifies the buffer is returned
// immediately and can be used after the signal semaphore fires.
TEST_P(QueueAllocaTest, BasicAlloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 1024;

  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, /*pool=*/NULL, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // Wait for the allocation to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify buffer metadata.
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), allocation_size);

  // Write a pattern and read it back to verify the buffer is usable.
  uint8_t pattern = 0xAB;
  FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                    &pattern, sizeof(pattern));

  std::vector<uint8_t> data =
      ReadBufferBytes(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size);
  for (iree_device_size_t i = 0; i < allocation_size; ++i) {
    ASSERT_EQ(data[i], 0xAB) << "Mismatch at byte " << i;
  }

  iree_hal_buffer_release(buffer);
}

// Allocates and deallocates through an explicit caller-provided pool created
// from the device's backend bundle instead of relying on the default pool.
TEST_P(QueueAllocaTest, ExplicitPassthroughPoolAllocaDealloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 1024;

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitPassthroughPool(pool.out()));

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, alloca_signal, pool.get(),
      MakeQueueAllocaBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      buffer.out()));
  ASSERT_NE(buffer.get(), nullptr);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t pattern = 0xA11CA7EDu;
  FillBufferAndWait(kQueueAffinity0, buffer.get(), 0, allocation_size, &pattern,
                    sizeof(pattern));
  uint32_t readback = ReadBufferValue<uint32_t>(kQueueAffinity0, buffer.get());
  EXPECT_EQ(readback, pattern);

  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, alloca_signal, dealloca_signal, buffer.get(),
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

// Allocates from an explicit suballocating pool, uses the transient buffer in a
// queue-ordered transfer chain, and deallocates after the final transfer. This
// exercises the production pool API without requiring an async pool wait.
TEST_P(QueueAllocaTest, ExplicitTLSFPoolTransferAllocaDealloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitTLSFPool(
      allocation_size, iree_hal_pool_epoch_query_null(), pool.out()));

  Ref<iree_hal_buffer_t> source_buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, alloca_signal, pool.get(),
      MakeQueueAllocaBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      source_buffer.out()));
  ASSERT_NE(source_buffer.get(), nullptr);

  Ref<iree_hal_buffer_t> target_buffer;
  CreateZeroedDeviceBuffer(allocation_size, target_buffer.out());

  SemaphoreList fill_signal(device_, {0}, {1});
  uint32_t pattern = 0x711FA110u;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, kQueueAffinity0, alloca_signal, fill_signal, source_buffer.get(),
      0, allocation_size, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, kQueueAffinity0, fill_signal, copy_signal, source_buffer.get(),
      0, target_buffer.get(), 0, allocation_size, IREE_HAL_COPY_FLAG_NONE));

  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, copy_signal, dealloca_signal,
      source_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const std::vector<uint32_t> data =
      ReadBufferData<uint32_t>(kQueueAffinity0, target_buffer.get());
  for (uint32_t value : data) {
    EXPECT_EQ(value, pattern);
  }
}

// Reuses a one-block explicit pool across two queues. The fixed-block pool is
// created without a host epoch query so the second queue must import the first
// queue's death frontier as a queue-owned hidden dependency instead of proving
// reuse safe via a host-side completion query.
TEST_P(QueueAllocaTest, ExplicitFixedBlockPoolCrossQueueWaitFrontier) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  bool has_queue1 = false;
  IREE_ASSERT_OK(HasQueueAffinity(kQueueAffinity1, &has_queue1));
  if (!has_queue1) {
    GTEST_SKIP() << "backend exposes fewer than two explicit queue affinities";
    return;
  }

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      allocation_size, iree_hal_pool_epoch_query_null(), pool.out()));

  iree_hal_buffer_params_t queue0_params = MakeQueueAllocaBufferParams();
  queue0_params.queue_affinity = kQueueAffinity0;
  Ref<iree_hal_buffer_t> queue0_buffer;
  SemaphoreList empty_wait;
  SemaphoreList queue0_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, queue0_alloca_signal, pool.get(),
      queue0_params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      queue0_buffer.out()));
  ASSERT_NE(queue0_buffer.get(), nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_alloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  SemaphoreList queue0_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, queue0_alloca_signal, queue0_dealloca_signal,
      queue0_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  queue0_buffer.reset();

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool.get(), &stats);
  EXPECT_EQ(stats.wait_count, 0u);

  iree_hal_buffer_params_t queue1_params = MakeQueueAllocaBufferParams();
  queue1_params.queue_affinity = kQueueAffinity1;
  Ref<iree_hal_buffer_t> queue1_buffer;
  SemaphoreList queue1_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity1, empty_wait, queue1_alloca_signal, pool.get(),
      queue1_params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER, queue1_buffer.out()));
  ASSERT_NE(queue1_buffer.get(), nullptr);

  iree_hal_pool_query_stats(pool.get(), &stats);
  EXPECT_EQ(stats.wait_count, 1u);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue1_alloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t pattern = 0xB10CADA0u;
  FillBufferAndWait(kQueueAffinity1, queue1_buffer.get(), 0, sizeof(pattern),
                    &pattern, sizeof(pattern));

  SemaphoreList queue1_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity1, queue1_alloca_signal, queue1_dealloca_signal,
      queue1_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue1_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

// Reuses a one-block explicit pool across two queues without a host wait
// between the releasing dealloca and the next alloca. Depending on scheduling,
// the dealloca may still be in flight or may have already published its death
// frontier; both cases must leave the second allocation usable.
TEST_P(QueueAllocaTest, ExplicitFixedBlockPoolPendingDeallocaWaitFrontier) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  bool has_queue1 = false;
  IREE_ASSERT_OK(HasQueueAffinity(kQueueAffinity1, &has_queue1));
  if (!has_queue1) {
    GTEST_SKIP() << "backend exposes fewer than two explicit queue affinities";
    return;
  }

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      allocation_size, iree_hal_pool_epoch_query_null(), pool.out()));

  iree_hal_buffer_params_t queue0_params = MakeQueueAllocaBufferParams();
  queue0_params.queue_affinity = kQueueAffinity0;
  Ref<iree_hal_buffer_t> queue0_buffer;
  SemaphoreList empty_wait;
  SemaphoreList queue0_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, queue0_alloca_signal, pool.get(),
      queue0_params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      queue0_buffer.out()));
  ASSERT_NE(queue0_buffer.get(), nullptr);

  SemaphoreList queue0_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, queue0_alloca_signal, queue0_dealloca_signal,
      queue0_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));

  iree_hal_buffer_params_t queue1_params = MakeQueueAllocaBufferParams();
  queue1_params.queue_affinity = kQueueAffinity1;
  Ref<iree_hal_buffer_t> queue1_buffer;
  SemaphoreList queue1_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity1, empty_wait, queue1_alloca_signal, pool.get(),
      queue1_params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER, queue1_buffer.out()));
  ASSERT_NE(queue1_buffer.get(), nullptr);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue1_alloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  queue0_buffer.reset();

  uint32_t pattern = 0xD0A110CAu;
  FillBufferAndWait(kQueueAffinity1, queue1_buffer.get(), 0, sizeof(pattern),
                    &pattern, sizeof(pattern));

  SemaphoreList queue1_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity1, queue1_alloca_signal, queue1_dealloca_signal,
      queue1_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue1_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

// Verifies that a waitable recycled block is not silently turned into a pool
// notification wait when the caller has not permitted hidden pool-frontier
// dependencies.
TEST_P(QueueAllocaTest, ExplicitFixedBlockPoolRequiresWaitFrontierFlag) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  bool has_queue1 = false;
  IREE_ASSERT_OK(HasQueueAffinity(kQueueAffinity1, &has_queue1));
  if (!has_queue1) {
    GTEST_SKIP() << "backend exposes fewer than two explicit queue affinities";
    return;
  }

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      allocation_size, iree_hal_pool_epoch_query_null(), pool.out()));

  iree_hal_buffer_params_t queue0_params = MakeQueueAllocaBufferParams();
  queue0_params.queue_affinity = kQueueAffinity0;
  Ref<iree_hal_buffer_t> queue0_buffer;
  SemaphoreList empty_wait;
  SemaphoreList queue0_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, queue0_alloca_signal, pool.get(),
      queue0_params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      queue0_buffer.out()));
  ASSERT_NE(queue0_buffer.get(), nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_alloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  SemaphoreList queue0_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, queue0_alloca_signal, queue0_dealloca_signal,
      queue0_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  queue0_buffer.reset();

  iree_hal_buffer_params_t queue1_params = MakeQueueAllocaBufferParams();
  queue1_params.queue_affinity = kQueueAffinity1;
  iree_hal_buffer_t* queue1_buffer = nullptr;
  SemaphoreList queue1_alloca_signal(device_, {0}, {1});
  iree_status_t alloca_status = iree_hal_device_queue_alloca(
      device_, kQueueAffinity1, empty_wait, queue1_alloca_signal, pool.get(),
      queue1_params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &queue1_buffer);
  EXPECT_THAT(Status(std::move(alloca_status)),
              StatusIs(StatusCode::kResourceExhausted));
  EXPECT_EQ(queue1_buffer, nullptr);
  iree_hal_buffer_release(queue1_buffer);
}

// Uses a one-range explicit TLSF pool across two queues. This exercises the
// generic suballocator used for variable-sized default pools. TLSF skips stale
// cross-queue blocks and grows instead of returning hidden wait-frontier
// reservations; fixed-block tests below cover notification-driven retry.
TEST_P(QueueAllocaTest, ExplicitTLSFPoolCrossQueueStaleBlockGrows) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  bool has_queue1 = false;
  IREE_ASSERT_OK(HasQueueAffinity(kQueueAffinity1, &has_queue1));
  if (!has_queue1) {
    GTEST_SKIP() << "backend exposes fewer than two explicit queue affinities";
    return;
  }

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitTLSFPool(
      allocation_size, iree_hal_pool_epoch_query_null(), pool.out()));

  iree_hal_buffer_params_t queue0_params = MakeQueueAllocaBufferParams();
  queue0_params.queue_affinity = kQueueAffinity0;
  Ref<iree_hal_buffer_t> queue0_buffer;
  SemaphoreList empty_wait;
  SemaphoreList queue0_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, queue0_alloca_signal, pool.get(),
      queue0_params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      queue0_buffer.out()));
  ASSERT_NE(queue0_buffer.get(), nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_alloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  SemaphoreList queue0_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, queue0_alloca_signal, queue0_dealloca_signal,
      queue0_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue0_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  queue0_buffer.reset();

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool.get(), &stats);
  EXPECT_EQ(stats.wait_count, 0u);

  iree_hal_buffer_params_t queue1_params = MakeQueueAllocaBufferParams();
  queue1_params.queue_affinity = kQueueAffinity1;
  Ref<iree_hal_buffer_t> queue1_buffer;
  SemaphoreList queue1_alloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity1, empty_wait, queue1_alloca_signal, pool.get(),
      queue1_params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER, queue1_buffer.out()));
  ASSERT_NE(queue1_buffer.get(), nullptr);

  iree_hal_pool_query_stats(pool.get(), &stats);
  EXPECT_EQ(stats.wait_count, 0u);
  EXPECT_GT(stats.slab_count, 1u);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue1_alloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t pattern = 0x715FADA0u;
  FillBufferAndWait(kQueueAffinity1, queue1_buffer.get(), 0, sizeof(pattern),
                    &pattern, sizeof(pattern));

  SemaphoreList queue1_dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity1, queue1_alloca_signal, queue1_dealloca_signal,
      queue1_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(queue1_dealloca_signal,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

// Reuses a one-block explicit pool after a second queue_alloca observes pool
// exhaustion. The second allocation must park on the pool's release
// notification, retry after the first dealloca releases its reservation, and
// then complete through the normal queue signal path.
TEST_P(QueueAllocaTest, ExplicitFixedBlockPoolNotificationRetry) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(CreateExplicitFixedBlockPool(
      allocation_size, iree_hal_pool_epoch_query_null(), pool.out()));

  Ref<iree_hal_buffer_t> buffer0;
  SemaphoreList empty_wait;
  SemaphoreList alloca0_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, alloca0_signal, pool.get(),
      MakeQueueAllocaBufferParams(), allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      buffer0.out()));
  ASSERT_NE(buffer0.get(), nullptr);
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca0_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_buffer_t> buffer1;
  SemaphoreList alloca1_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, kQueueAffinity0, empty_wait, alloca1_signal, pool.get(),
      MakeQueueAllocaBufferParams(), allocation_size,
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER, buffer1.out()));
  ASSERT_NE(buffer1.get(), nullptr);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool.get(), &stats);
  EXPECT_GE(stats.exhausted_count, 1u);
  EXPECT_FALSE(iree_hal_semaphore_list_poll(alloca1_signal));

  SemaphoreList dealloca0_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, alloca0_signal, dealloca0_signal, buffer0.get(),
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca1_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca0_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  buffer0.reset();

  uint32_t pattern = 0xB10CA110u;
  FillBufferAndWait(kQueueAffinity0, buffer1.get(), 0, sizeof(pattern),
                    &pattern, sizeof(pattern));

  SemaphoreList dealloca1_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, kQueueAffinity0, alloca1_signal, dealloca1_signal, buffer1.get(),
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca1_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

// Allocates a buffer that waits on a semaphore before committing.
TEST_P(QueueAllocaTest, AllocaWithWaitSemaphores) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 512;

  SemaphoreList wait_semaphore_list(device_, {0}, {1});
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, /*pool=*/NULL, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // The backing is not committed until the wait semaphore fires.
  EXPECT_FALSE(iree_hal_semaphore_list_poll(signal_semaphore_list));
  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(wait_semaphore_list,
                                                /*frontier=*/NULL));

  // Wait for the allocation to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify the buffer is usable after the signal.
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), allocation_size);

  uint8_t pattern = 0xCD;
  FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                    &pattern, sizeof(pattern));

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

    iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

    iree_hal_buffer_t* buffer = NULL;
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        /*pool=*/NULL, params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
        &buffer));
    ASSERT_NE(buffer, nullptr);

    // Wait for alloca, fill the buffer.
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    uint32_t pattern = 0xDEADCAFE;
    FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                      &pattern, sizeof(pattern));

    uint32_t readback =
        ReadBufferValue<uint32_t>(IREE_HAL_QUEUE_AFFINITY_ANY, buffer);
    EXPECT_EQ(readback, 0xDEADCAFE);

    // Dealloca: wait on alloca completion, signal dealloca completion.
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
        buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_release(buffer);
  }

  // Phase 2: alloca again to verify the system handles repeated cycles.
  {
    SemaphoreList alloca_signal(device_, {0}, {1});
    SemaphoreList dealloca_signal(device_, {0}, {1});

    iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

    iree_hal_buffer_t* buffer = NULL;
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        /*pool=*/NULL, params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
        &buffer));
    ASSERT_NE(buffer, nullptr);

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    uint32_t pattern = 0xCAFEF00D;
    FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                      &pattern, sizeof(pattern));

    uint32_t readback =
        ReadBufferValue<uint32_t>(IREE_HAL_QUEUE_AFFINITY_ANY, buffer);
    EXPECT_EQ(readback, 0xCAFEF00D);

    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
        buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_release(buffer);
  }
}

// Verifies returned buffer metadata matches the requested parameters.
TEST_P(QueueAllocaTest, BufferMetadata) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 4096;

  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, /*pool=*/NULL, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  // Buffer metadata is available immediately (before the backing is committed).
  EXPECT_GE(iree_hal_buffer_byte_length(buffer), allocation_size);
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE));

  // The buffer must have the ASYNCHRONOUS placement flag for dealloca routing.
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  EXPECT_EQ(placement.device, device_);
  EXPECT_TRUE(iree_all_bits_set(placement.flags,
                                IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS));
  EXPECT_FALSE(iree_hal_queue_affinity_is_empty(placement.queue_affinity));
  EXPECT_EQ(iree_hal_queue_affinity_count(placement.queue_affinity), 1u);

  // Wait for completion and clean up.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Dealloca to clean up the transient buffer.
  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, signal_semaphore_list,
      dealloca_signal, buffer, IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
}

// Verifies that accessing a decommitted buffer after dealloca fails.
TEST_P(QueueAllocaTest, DeallocaReleasesMemory) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 256;

  SemaphoreList alloca_signal(device_, {0}, {1});
  SemaphoreList dealloca_signal(device_, {0}, {1});

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  SemaphoreList empty_wait;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      /*pool=*/NULL, params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer));
  ASSERT_NE(buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify the buffer works before dealloca.
  uint8_t pattern = 0xFF;
  FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                    &pattern, sizeof(pattern));

  // Dealloca: wait on alloca completion.
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
      buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  SemaphoreList post_dealloca_signal(device_, {0}, {1});
  iree_status_t fill_status = iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, post_dealloca_signal,
      buffer, 0, allocation_size, &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE);
  if (iree_status_is_ok(fill_status)) {
    EXPECT_THAT(Status(iree_hal_semaphore_list_wait(post_dealloca_signal,
                                                    iree_infinite_timeout(),
                                                    IREE_ASYNC_WAIT_FLAG_NONE)),
                StatusIs(StatusCode::kFailedPrecondition));
  } else {
    EXPECT_THAT(Status(std::move(fill_status)),
                StatusIs(StatusCode::kInvalidArgument));
  }

  iree_hal_buffer_release(buffer);
}

// A dealloca whose wait dependency has already failed must not leave the
// buffer marked as deallocation-queued when no dealloca work was submitted.
TEST_P(QueueAllocaTest, FailedDeallocaWaitDoesNotDealloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 256;

  SemaphoreList alloca_signal(device_, {0}, {1});
  SemaphoreList failed_wait(device_, {0}, {1});
  SemaphoreList failed_dealloca_signal(device_, {0}, {1});

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  SemaphoreList empty_wait;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      /*pool=*/NULL, params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer));
  ASSERT_NE(buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint8_t pattern = 0xAB;
  FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                    &pattern, sizeof(pattern));

  iree_hal_semaphore_fail(
      failed_wait.semaphores[0],
      iree_make_status(IREE_STATUS_CANCELLED, "dealloca wait failed"));

  iree_status_t dealloca_status = iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, failed_wait, failed_dealloca_signal,
      buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  if (iree_status_is_ok(dealloca_status)) {
    EXPECT_THAT(Status(iree_hal_semaphore_list_wait(failed_dealloca_signal,
                                                    iree_infinite_timeout(),
                                                    IREE_ASYNC_WAIT_FLAG_NONE)),
                StatusIs(StatusCode::kCancelled));
  } else {
    EXPECT_THAT(Status(std::move(dealloca_status)),
                StatusIs(StatusCode::kCancelled));
  }

  pattern = 0xCD;
  FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                    &pattern, sizeof(pattern));

  SemaphoreList cleanup_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, cleanup_signal, buffer,
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      cleanup_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

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

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  SemaphoreList empty_wait;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      /*pool=*/NULL, params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &buffer));
  ASSERT_NE(buffer, nullptr);

  // Chain dealloca to wait on alloca completion — no host wait in between.
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
      buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

  // Wait for the entire chain to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
}

// Repeats alloca -> dealloca chains through the device default pool. This keeps
// the test generic while still exercising default-pool reuse in drivers backed
// by frontier-aware suballocating pools.
TEST_P(QueueAllocaTest, DefaultPoolRepeatedChainedAllocaDealloca) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 1024;
  const uint32_t iteration_count = 32;

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  for (uint32_t i = 0; i < iteration_count; ++i) {
    SemaphoreList alloca_signal(device_, {0}, {1});
    SemaphoreList dealloca_signal(device_, {0}, {1});

    SemaphoreList empty_wait;
    iree_hal_buffer_t* buffer = NULL;
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        /*pool=*/NULL, params, allocation_size, IREE_HAL_ALLOCA_FLAG_NONE,
        &buffer));
    ASSERT_NE(buffer, nullptr);

    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, dealloca_signal,
        buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    iree_hal_buffer_release(buffer);
  }
}

// Verifies PREFER_ORIGIN reroutes dealloca to the queue affinity recorded in
// the buffer placement instead of requiring the caller to repeat that queue
// affinity manually.
TEST_P(QueueAllocaTest, DeallocaPrefersOriginPlacement) {
  IREE_TRACE_SCOPE();

  SemaphoreList alloca_signal(device_, {0}, {1});
  SemaphoreList dealloca_signal(device_, {0}, {1});

  iree_hal_buffer_params_t params = MakeQueueAllocaBufferParams();

  SemaphoreList empty_wait;
  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      /*pool=*/NULL, params, /*allocation_size=*/1024,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, /*queue_affinity=*/0, alloca_signal, dealloca_signal, buffer,
      IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_release(buffer);
}

// Allocates with zero access flags (as the HAL module does) and verifies the
// buffer is usable after canonicalization promotes access to ALL.
TEST_P(QueueAllocaTest, ZeroAccessFlagsCanonicalized) {
  IREE_TRACE_SCOPE();

  const iree_device_size_t allocation_size = 256;

  SemaphoreList wait_semaphore_list;
  SemaphoreList signal_semaphore_list(device_, {0}, {1});

  // The HAL module constructs params with only .type and .usage, leaving
  // .access = 0. The driver must canonicalize this to MEMORY_ACCESS_ALL.
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, /*pool=*/NULL, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));
  ASSERT_NE(buffer, nullptr);

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal_semaphore_list,
                                              iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Fill and readback: exercises DISCARD_WRITE and READ access on the buffer.
  uint32_t pattern = 0x12345678;
  FillBufferAndWait(IREE_HAL_QUEUE_AFFINITY_ANY, buffer, 0, allocation_size,
                    &pattern, sizeof(pattern));

  uint32_t readback =
      ReadBufferValue<uint32_t>(IREE_HAL_QUEUE_AFFINITY_ANY, buffer);
  EXPECT_EQ(readback, 0x12345678);

  iree_hal_buffer_release(buffer);
}

CTS_REGISTER_TEST_SUITE(QueueAllocaTest);

}  // namespace iree::hal::cts
