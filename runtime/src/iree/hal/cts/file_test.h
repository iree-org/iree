// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_FILE_TEST_H_
#define IREE_HAL_CTS_FILE_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

using ::testing::ContainerEq;

namespace {
constexpr iree_device_size_t kMinimumAlignment = 128;
}  // namespace

class file_test : public CtsTestBase {
 protected:
  void CreatePatternedDeviceBuffer(iree_device_size_t buffer_size,
                                   uint8_t pattern,
                                   iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment = kMinimumAlignment;
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_), params, buffer_size,
        &device_buffer));

    iree_hal_transfer_command_t transfer_command;
    memset(&transfer_command, 0, sizeof(transfer_command));
    transfer_command.type = IREE_HAL_TRANSFER_COMMAND_TYPE_FILL;
    transfer_command.fill.target_buffer = device_buffer;
    transfer_command.fill.target_offset = 0;
    transfer_command.fill.length = buffer_size;
    transfer_command.fill.pattern = &pattern;
    transfer_command.fill.pattern_length = sizeof(pattern);
    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_CHECK_OK(iree_hal_create_transfer_command_buffer(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_QUEUE_AFFINITY_ANY, 1, &transfer_command, &command_buffer));
    IREE_CHECK_OK(SubmitCommandBufferAndWait(command_buffer));
    iree_hal_command_buffer_release(command_buffer);

    *out_buffer = device_buffer;
  }

  void CreatePatternedMemoryFile(iree_hal_memory_access_t access,
                                 iree_device_size_t file_size, uint8_t pattern,
                                 iree_hal_file_t** out_file) {
    void* file_contents = NULL;
    IREE_CHECK_OK(iree_allocator_malloc_aligned(iree_allocator_system(),
                                                file_size, kMinimumAlignment, 0,
                                                (void**)&file_contents));
    memset(file_contents, pattern, file_size);

    iree_io_file_handle_release_callback_t release_callback = {
        +[](void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
          iree_allocator_free_aligned(iree_allocator_system(), user_data);
        },
        file_contents,
    };
    iree_io_file_handle_t* handle = NULL;
    IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(file_contents, file_size), release_callback,
        iree_allocator_system(), &handle));
    IREE_CHECK_OK(iree_hal_file_import(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, access, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file));
    iree_io_file_handle_release(handle);
  }
};

// Reads the entire file into a buffer and check the contents match.
TEST_P(file_test, ReadEntireFile) {
  iree_device_size_t file_size = 128;
  iree_hal_file_t* file = NULL;
  CreatePatternedMemoryFile(IREE_HAL_MEMORY_ACCESS_READ, file_size, 0xDEu,
                            &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0xCD, &buffer);

  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(iree_hal_semaphore_create(device_, 0ull, &semaphore));
  iree_hal_fence_t* wait_fence = NULL;
  IREE_ASSERT_OK(iree_hal_fence_create_at(
      semaphore, 1ull, iree_allocator_system(), &wait_fence));
  iree_hal_fence_t* signal_fence = NULL;
  IREE_ASSERT_OK(iree_hal_fence_create_at(
      semaphore, 2ull, iree_allocator_system(), &signal_fence));

  // NOTE: synchronously executing here so start with the wait signaled.
  // We should be able to make this async in the future.
  IREE_ASSERT_OK(iree_hal_fence_signal(wait_fence));

  IREE_ASSERT_OK(iree_hal_device_queue_read(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_fence_semaphore_list(wait_fence),
      iree_hal_fence_semaphore_list(signal_fence), /*source_file=*/file,
      /*source_offset=*/0, /*target_buffer=*/buffer, /*target_offset=*/0,
      /*length=*/file_size, /*flags=*/0));

  IREE_ASSERT_OK(iree_hal_fence_wait(signal_fence, iree_infinite_timeout()));
  iree_hal_fence_release(wait_fence);
  iree_hal_fence_release(signal_fence);
  iree_hal_semaphore_release(semaphore);

  std::vector<uint8_t> reference_buffer(file_size);
  memset(reference_buffer.data(), 0xDEu, file_size);
  std::vector<uint8_t> actual_data(file_size);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, buffer, /*source_offset=*/0,
      /*target_buffer=*/actual_data.data(),
      /*data_length=*/file_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_FILE_TEST_H_
