// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::cts {

namespace {

struct FileHandleDeleter {
  void operator()(iree_io_file_handle_t* handle) const {
    iree_io_file_handle_release(handle);
  }
};

using FileHandlePtr = std::unique_ptr<iree_io_file_handle_t, FileHandleDeleter>;

}  // namespace

class VulkanFileTest : public CtsTestBase<> {
 protected:
  void ImportNativeFile(const std::vector<uint8_t>& file_contents,
                        const char* temp_file_stem,
                        Ref<iree_hal_file_t>* out_file) {
    iree::testing::TempFilePath path(temp_file_stem);
    IREE_ASSERT_OK(iree_io_file_contents_write(
        path.path_view(),
        iree_make_const_byte_span(file_contents.data(), file_contents.size()),
        iree_allocator_system()));

    iree_io_file_handle_t* raw_handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
        path.path_view(), iree_allocator_system(), &raw_handle));
    FileHandlePtr handle(raw_handle);

    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(iree_hal_file_import(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        handle.get(), IREE_HAL_EXTERNAL_FILE_FLAG_NONE, file.out()));
    *out_file = std::move(file);
  }

  void NativeFileRoundTrip(iree_device_size_t file_length,
                           const char* temp_file_stem) {
    const std::vector<uint8_t> file_contents(file_length, 0);
    const std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(file_length);

    Ref<iree_hal_file_t> file;
    ImportNativeFile(file_contents, temp_file_stem, &file);

    Ref<iree_hal_buffer_t> source_buffer;
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        source_contents.data(), source_contents.size(), source_buffer.out()));
    Ref<iree_hal_buffer_t> target_buffer;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(file_length, target_buffer.out()));

    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    SemaphoreList read_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, write_signal,
        source_buffer.get(), /*source_offset=*/0, file.get(),
        /*target_offset=*/0, file_length, IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_device_queue_read(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, write_signal, read_signal,
        file.get(), /*source_offset=*/0, target_buffer.get(),
        /*target_offset=*/0, file_length, IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    const std::vector<uint8_t> actual_contents =
        ReadBufferBytes(target_buffer.get(), /*offset=*/0, file_length);
    ASSERT_EQ(actual_contents.size(), source_contents.size());
    EXPECT_EQ(0, std::memcmp(actual_contents.data(), source_contents.data(),
                             source_contents.size()));
  }
};

TEST_P(VulkanFileTest, NativeFileLargeRangeRoundTrip) {
  const iree_device_size_t file_length = 256 * 1024 + 31;
  NativeFileRoundTrip(file_length, "iree_vulkan_hal_cts_native_file");
}

TEST_P(VulkanFileTest, NativeFileRoundTripExceedsStagingRing) {
  const iree_device_size_t file_length = 20 * 1024 * 1024 + 31;
  NativeFileRoundTrip(file_length, "iree_vulkan_hal_cts_native_file_staging");
}

TEST_P(VulkanFileTest, NativeFileShortReadExceedsStagingRingFails) {
  const iree_device_size_t import_length = 20 * 1024 * 1024 + 31;
  const iree_device_size_t actual_length = 4 * 1024 * 1024 + 17;
  const std::vector<uint8_t> import_contents(import_length, 0);
  const std::vector<uint8_t> actual_contents =
      MakeDeterministicBytes(actual_length);

  iree::testing::TempFilePath path("iree_vulkan_hal_cts_native_file_short");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(import_contents.data(), import_contents.size()),
      iree_allocator_system()));

  iree_io_file_handle_t* raw_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), iree_allocator_system(), &raw_handle));
  FileHandlePtr handle(raw_handle);

  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(iree_hal_file_import(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle.get(),
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, file.out()));

  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(actual_contents.data(), actual_contents.size()),
      iree_allocator_system()));

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(import_length, target_buffer.out()));

  SemaphoreList empty_wait;
  SemaphoreList read_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, read_signal, file.get(),
      /*source_offset=*/0, target_buffer.get(), /*target_offset=*/0,
      import_length, IREE_HAL_READ_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_semaphore_list_wait(read_signal, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(VulkanFileTest, {"file_io"}, {});

}  // namespace iree::hal::cts
