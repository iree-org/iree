// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_handle.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#endif

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {
constexpr iree_device_size_t kMinimumAlignment = 128;
}  // namespace

class FileTest : public CtsTestBase<> {
 protected:
  void TearDown() override {
    for (const auto& path : temp_paths_) {
#if defined(IREE_PLATFORM_WINDOWS)
      DeleteFileA(path.c_str());
#else
      unlink(path.c_str());
#endif
    }
    CtsTestBase::TearDown();
  }

  void CreatePatternedDeviceBuffer(iree_device_size_t buffer_size,
                                   uint8_t pattern,
                                   iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment = kMinimumAlignment;
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator_, params, buffer_size, &device_buffer));

    iree_hal_transfer_command_t transfer_command;
    memset(&transfer_command, 0, sizeof(transfer_command));
    transfer_command.type = IREE_HAL_TRANSFER_COMMAND_TYPE_FILL;
    transfer_command.fill.target_buffer = device_buffer;
    transfer_command.fill.target_offset = 0;
    transfer_command.fill.length = buffer_size;
    transfer_command.fill.pattern = &pattern;
    transfer_command.fill.pattern_length = sizeof(pattern);
    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_create_transfer_command_buffer(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_QUEUE_AFFINITY_ANY, 1, &transfer_command, &command_buffer));
    IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
    iree_hal_command_buffer_release(command_buffer);

    *out_buffer = device_buffer;
  }

  void CreatePatternedMemoryFile(iree_hal_memory_access_t access,
                                 iree_device_size_t file_size, uint8_t pattern,
                                 iree_hal_file_t** out_file) {
    void* file_contents = NULL;
    IREE_ASSERT_OK(iree_allocator_malloc_aligned(iree_allocator_system(),
                                                 file_size, kMinimumAlignment,
                                                 0, (void**)&file_contents));
    memset(file_contents, pattern, file_size);

    iree_io_file_handle_release_callback_t release_callback = {
        +[](void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
          iree_allocator_free_aligned(iree_allocator_system(), user_data);
        },
        file_contents,
    };
    iree_io_file_handle_t* handle = NULL;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(file_contents, file_size), release_callback,
        iree_allocator_system(), &handle));
    IREE_ASSERT_OK(iree_hal_file_import(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, access, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file));
    iree_io_file_handle_release(handle);
  }

  // Creates a temp file on disk with |data|, returns the path.
  // The file is closed after writing.
  std::string CreateTempFileWithContents(const void* data, size_t length) {
#if defined(IREE_PLATFORM_WINDOWS)
    char temp_dir[MAX_PATH] = {0};
    GetTempPathA(MAX_PATH, temp_dir);
    char temp_path[MAX_PATH] = {0};
    GetTempFileNameA(temp_dir, "cts", 0, temp_path);
    HANDLE handle = CreateFileA(temp_path, GENERIC_WRITE, 0, NULL,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    EXPECT_NE(handle, INVALID_HANDLE_VALUE);
    DWORD bytes_written = 0;
    WriteFile(handle, data, (DWORD)length, &bytes_written, NULL);
    CloseHandle(handle);
    temp_paths_.push_back(temp_path);
    return temp_path;
#else
    char temp_path[] = "/tmp/iree_hal_cts_file_XXXXXX";
    int fd = mkstemp(temp_path);
    EXPECT_GE(fd, 0);
    if (fd >= 0) {
      ssize_t written = write(fd, data, length);
      EXPECT_EQ(written, static_cast<ssize_t>(length));
      close(fd);
    }
    temp_paths_.push_back(temp_path);
    return temp_path;
#endif
  }

  // Creates a temp file with a repeating byte pattern.
  std::string CreatePatternedTempFile(size_t size, uint8_t pattern) {
    std::vector<uint8_t> data(size, pattern);
    return CreateTempFileWithContents(data.data(), data.size());
  }

  // Imports an on-disk file as a HAL file via its path.
  void ImportFdFile(const std::string& path, iree_hal_memory_access_t access,
                    iree_hal_file_t** out_file) {
    iree_io_file_mode_t mode = IREE_IO_FILE_MODE_READ;
    if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      mode |= IREE_IO_FILE_MODE_WRITE;
    }
    iree_io_file_handle_t* handle = NULL;
    IREE_ASSERT_OK(
        iree_io_file_handle_open(mode, iree_make_cstring_view(path.c_str()),
                                 iree_allocator_system(), &handle));
    IREE_ASSERT_OK(iree_hal_file_import(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, access, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file));
    iree_io_file_handle_release(handle);
  }

  // Submits a queue_read and waits for completion.
  void QueueReadAndWait(iree_hal_file_t* source_file, uint64_t source_offset,
                        iree_hal_buffer_t* target_buffer,
                        iree_device_size_t target_offset,
                        iree_device_size_t length) {
    iree_hal_semaphore_t* semaphore = NULL;
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    iree_hal_fence_t* wait_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 1ull, iree_allocator_system(), &wait_fence));
    iree_hal_fence_t* signal_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 2ull, iree_allocator_system(), &signal_fence));
    IREE_ASSERT_OK(iree_hal_fence_signal(wait_fence));

    IREE_ASSERT_OK(iree_hal_device_queue_read(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), source_file, source_offset,
        target_buffer, target_offset, length, IREE_HAL_READ_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_fence_wait(signal_fence, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_fence_release(wait_fence);
    iree_hal_fence_release(signal_fence);
    iree_hal_semaphore_release(semaphore);
  }

  // Submits a queue_write and waits for completion.
  void QueueWriteAndWait(iree_hal_buffer_t* source_buffer,
                         iree_device_size_t source_offset,
                         iree_hal_file_t* target_file, uint64_t target_offset,
                         iree_device_size_t length) {
    iree_hal_semaphore_t* semaphore = NULL;
    IREE_ASSERT_OK(
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                  IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    iree_hal_fence_t* wait_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 1ull, iree_allocator_system(), &wait_fence));
    iree_hal_fence_t* signal_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 2ull, iree_allocator_system(), &signal_fence));
    IREE_ASSERT_OK(iree_hal_fence_signal(wait_fence));

    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), source_buffer,
        source_offset, target_file, target_offset, length,
        IREE_HAL_WRITE_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_fence_wait(signal_fence, iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE));
    iree_hal_fence_release(wait_fence);
    iree_hal_fence_release(signal_fence);
    iree_hal_semaphore_release(semaphore);
  }

  // Reads buffer contents back to host for verification.
  std::vector<uint8_t> ReadBufferContents(iree_hal_buffer_t* buffer,
                                          iree_device_size_t offset,
                                          iree_device_size_t length) {
    std::vector<uint8_t> data(length);
    IREE_EXPECT_OK(iree_hal_device_transfer_d2h(
        device_, buffer, offset, data.data(), length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    return data;
  }

  std::vector<std::string> temp_paths_;
};

//===----------------------------------------------------------------------===//
// Memory file tests (storage_buffer -> queue_copy path)
//===----------------------------------------------------------------------===//

TEST_P(FileTest, MemoryFileReadEntire) {
  iree_device_size_t file_size = 128;
  iree_hal_file_t* file = NULL;
  CreatePatternedMemoryFile(IREE_HAL_MEMORY_ACCESS_READ, file_size, 0xDEu,
                            &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0xCD, &buffer);

  QueueReadAndWait(file, 0, buffer, 0, file_size);

  std::vector<uint8_t> expected(file_size, 0xDE);
  EXPECT_THAT(ReadBufferContents(buffer, 0, file_size), ContainerEq(expected));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

TEST_P(FileTest, MemoryFileWriteEntire) {
  iree_device_size_t file_size = 128;
  iree_hal_file_t* file = NULL;
  CreatePatternedMemoryFile(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file_size,
      0x00u, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0xAB, &buffer);

  QueueWriteAndWait(buffer, 0, file, 0, file_size);

  // Verify via synchronous read: re-import the same memory as readable.
  // The storage_buffer should reflect the written data.
  iree_hal_buffer_t* storage = iree_hal_file_storage_buffer(file);
  ASSERT_NE(storage, nullptr);
  std::vector<uint8_t> expected(file_size, 0xAB);
  EXPECT_THAT(ReadBufferContents(storage, 0, file_size), ContainerEq(expected));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

//===----------------------------------------------------------------------===//
// FD file tests (proactor async I/O path)
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

TEST_P(FileTest, FdFileReadEntire) {
  const iree_device_size_t file_size = 4096;
  std::string path = CreatePatternedTempFile(file_size, 0xBE);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);

  QueueReadAndWait(file, 0, buffer, 0, file_size);

  std::vector<uint8_t> expected(file_size, 0xBE);
  EXPECT_THAT(ReadBufferContents(buffer, 0, file_size), ContainerEq(expected));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

TEST_P(FileTest, FdFileReadSubrange) {
  const iree_device_size_t file_size = 4096;
  std::vector<uint8_t> file_data(file_size);
  for (size_t i = 0; i < file_size; ++i) {
    file_data[i] = static_cast<uint8_t>(i & 0xFF);
  }
  std::string path =
      CreateTempFileWithContents(file_data.data(), file_data.size());

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);

  // Read 256 bytes from offset 512 into buffer offset 128.
  const iree_device_size_t read_offset = 512;
  const iree_device_size_t buffer_offset = 128;
  const iree_device_size_t read_length = 256;
  QueueReadAndWait(file, read_offset, buffer, buffer_offset, read_length);

  std::vector<uint8_t> expected(file_data.begin() + read_offset,
                                file_data.begin() + read_offset + read_length);
  EXPECT_THAT(ReadBufferContents(buffer, buffer_offset, read_length),
              ContainerEq(expected));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

TEST_P(FileTest, FdFileWriteEntire) {
  const iree_device_size_t file_size = 4096;
  std::string path = CreatePatternedTempFile(file_size, 0x00);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
               &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0xCA, &buffer);

  QueueWriteAndWait(buffer, 0, file, 0, file_size);

  // Verify: read the file back from disk.
  iree_hal_file_release(file);
  file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* readback = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &readback);
  QueueReadAndWait(file, 0, readback, 0, file_size);

  std::vector<uint8_t> expected(file_size, 0xCA);
  EXPECT_THAT(ReadBufferContents(readback, 0, file_size),
              ContainerEq(expected));

  iree_hal_buffer_release(readback);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

TEST_P(FileTest, FdFileReadRangeValidation) {
  const iree_device_size_t file_size = 128;
  std::string path = CreatePatternedTempFile(file_size, 0xAA);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(256, 0x00, &buffer);

  // Read past EOF: should fail at submit time with OUT_OF_RANGE.
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  iree_hal_fence_t* wait_fence = NULL;
  IREE_ASSERT_OK(iree_hal_fence_create_at(
      semaphore, 1ull, iree_allocator_system(), &wait_fence));
  iree_hal_fence_t* signal_fence = NULL;
  IREE_ASSERT_OK(iree_hal_fence_create_at(
      semaphore, 2ull, iree_allocator_system(), &signal_fence));
  IREE_ASSERT_OK(iree_hal_fence_signal(wait_fence));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_device_queue_read(
                            device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                            iree_hal_fence_semaphore_list(wait_fence),
                            iree_hal_fence_semaphore_list(signal_fence), file,
                            /*source_offset=*/0, buffer, /*target_offset=*/0,
                            /*length=*/256, IREE_HAL_READ_FLAG_NONE));

  iree_hal_fence_release(wait_fence);
  iree_hal_fence_release(signal_fence);
  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

#endif  // IREE_FILE_IO_ENABLE

CTS_REGISTER_TEST_SUITE(FileTest);

}  // namespace iree::hal::cts
