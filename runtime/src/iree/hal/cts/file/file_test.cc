// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
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
constexpr iree_device_size_t kLargeFdFileSize = 20 * 1024 * 1024;

uint8_t PatternByte(size_t index) {
  return static_cast<uint8_t>((index * 7 + 13) & 0xFF);
}

std::vector<uint8_t> MakePatternData(size_t size) {
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = PatternByte(i);
  }
  return data;
}
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

    FillDeviceBufferRange(device_buffer, 0, buffer_size, pattern);
    *out_buffer = device_buffer;
  }

  void FillDeviceBufferRange(iree_hal_buffer_t* buffer,
                             iree_device_size_t offset,
                             iree_device_size_t length, uint8_t pattern) {
    if (length == 0) return;
    SemaphoreList empty_wait;
    SemaphoreList fill_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal, buffer,
        offset, length, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
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
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
#if defined(IREE_PLATFORM_WINDOWS)
    char temp_dir[MAX_PATH] = {0};
    GetTempPathA(MAX_PATH, temp_dir);
    char temp_path[MAX_PATH] = {0};
    GetTempFileNameA(temp_dir, "cts", 0, temp_path);
    HANDLE handle = CreateFileA(temp_path, GENERIC_WRITE, 0, NULL,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    EXPECT_NE(handle, INVALID_HANDLE_VALUE);
    size_t total_written = 0;
    if (handle != INVALID_HANDLE_VALUE) {
      while (total_written < length) {
        const size_t remaining = length - total_written;
        const DWORD chunk_length =
            static_cast<DWORD>(std::min<size_t>(remaining, UINT32_MAX));
        DWORD bytes_written = 0;
        const BOOL did_write = WriteFile(handle, bytes + total_written,
                                         chunk_length, &bytes_written, NULL);
        EXPECT_TRUE(did_write);
        if (!did_write) break;
        EXPECT_GT(bytes_written, 0u);
        if (bytes_written == 0) break;
        total_written += bytes_written;
      }
      CloseHandle(handle);
    }
    EXPECT_EQ(total_written, length);
    temp_paths_.push_back(temp_path);
    return temp_path;
#else
    char temp_path[] = "/tmp/iree_hal_cts_file_XXXXXX";
    int fd = mkstemp(temp_path);
    EXPECT_GE(fd, 0);
    if (fd >= 0) {
      size_t total_written = 0;
      while (total_written < length) {
        ssize_t written =
            write(fd, bytes + total_written, length - total_written);
        if (written < 0 && errno == EINTR) continue;
        EXPECT_GT(written, 0);
        if (written <= 0) break;
        total_written += static_cast<size_t>(written);
      }
      EXPECT_EQ(total_written, length);
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

  // Reads exactly |length| bytes from a temp file on disk.
  std::vector<uint8_t> ReadTempFileContents(const std::string& path,
                                            size_t length) {
    std::vector<uint8_t> data(length);
    std::ifstream file(path, std::ios::binary);
    EXPECT_TRUE(file.good());
    if (file.good()) {
      file.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
      EXPECT_EQ(file.gcount(), static_cast<std::streamsize>(data.size()));
    }
    return data;
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
    if (iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
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
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_WRITE,
        handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
    iree_io_file_handle_release(handle);
    if (!file) return data;

    QueueWriteAndWait(buffer, offset, file, 0, length);

    iree_hal_file_release(file);
    return data;
  }

  void ExpectByteRangeRepeated(const std::vector<uint8_t>& data, size_t offset,
                               size_t length, uint8_t pattern) {
    ASSERT_LE(offset, data.size());
    ASSERT_LE(length, data.size() - offset);
    for (size_t i = 0; i < length; ++i) {
      if (data[offset + i] != pattern) {
        ADD_FAILURE() << "byte mismatch at offset " << (offset + i)
                      << ": expected 0x" << std::hex
                      << static_cast<int>(pattern) << ", got 0x"
                      << static_cast<int>(data[offset + i]);
        return;
      }
    }
  }

  void ExpectByteRangeMatches(const std::vector<uint8_t>& data, size_t offset,
                              const std::vector<uint8_t>& expected) {
    ASSERT_LE(offset, data.size());
    ASSERT_LE(expected.size(), data.size() - offset);
    if (!expected.empty()) {
      EXPECT_EQ(memcmp(data.data() + offset, expected.data(), expected.size()),
                0);
    }
  }

  void ExpectBufferRangeMatches(iree_hal_buffer_t* buffer,
                                iree_device_size_t offset,
                                const std::vector<uint8_t>& expected) {
    std::vector<uint8_t> contents =
        ReadBufferContents(buffer, offset, expected.size());
    ASSERT_EQ(contents.size(), expected.size());
    ExpectByteRangeMatches(contents, 0, expected);
  }

  void ExpectBufferRangeRepeated(iree_hal_buffer_t* buffer,
                                 iree_device_size_t offset,
                                 iree_device_size_t length, uint8_t pattern) {
    std::vector<uint8_t> contents = ReadBufferContents(buffer, offset, length);
    ASSERT_EQ(contents.size(), length);
    ExpectByteRangeRepeated(contents, 0, contents.size(), pattern);
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

//===----------------------------------------------------------------------===//
// Memory file ordering and pipeline tests
//===----------------------------------------------------------------------===//

// Reads from a memory file into a buffer, then writes from that buffer to
// a different memory file. Verifies queue ordering between read and write
// via semaphore chaining (no host waits between operations).
TEST_P(FileTest, MemoryFileChainedReadWrite) {
  const iree_device_size_t file_size = 256;

  // Source file: patterned data.
  iree_hal_file_t* source_file = NULL;
  CreatePatternedMemoryFile(IREE_HAL_MEMORY_ACCESS_READ, file_size, 0xEEu,
                            &source_file);

  // Target file: zeroed.
  iree_hal_file_t* target_file = NULL;
  CreatePatternedMemoryFile(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file_size,
      0x00u, &target_file);

  // Intermediate buffer.
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);

  // Chain: read source→buffer (signal@1), write buffer→target (wait@1,
  // signal@2). Single semaphore with advancing timeline.
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  // Read: wait@0 (immediate), signal@1.
  {
    iree_hal_fence_t* wait_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 0ull, iree_allocator_system(), &wait_fence));
    iree_hal_fence_t* signal_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 1ull, iree_allocator_system(), &signal_fence));
    IREE_ASSERT_OK(iree_hal_device_queue_read(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), source_file, 0, buffer, 0,
        file_size, IREE_HAL_READ_FLAG_NONE));
    iree_hal_fence_release(wait_fence);
    iree_hal_fence_release(signal_fence);
  }

  // Write: wait@1, signal@2.
  {
    iree_hal_fence_t* wait_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 1ull, iree_allocator_system(), &wait_fence));
    iree_hal_fence_t* signal_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 2ull, iree_allocator_system(), &signal_fence));
    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), buffer, 0, target_file, 0,
        file_size, IREE_HAL_WRITE_FLAG_NONE));
    iree_hal_fence_release(wait_fence);
    iree_hal_fence_release(signal_fence);
  }

  // Only wait on the final signal — ordering must be correct for data to
  // flow from source through buffer to target.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  // Verify the target file received the source data.
  iree_hal_buffer_t* target_storage = iree_hal_file_storage_buffer(target_file);
  ASSERT_NE(target_storage, nullptr);
  std::vector<uint8_t> expected(file_size, 0xEE);
  EXPECT_THAT(ReadBufferContents(target_storage, 0, file_size),
              ContainerEq(expected));

  iree_hal_semaphore_release(semaphore);
  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target_file);
  iree_hal_file_release(source_file);
}

// Reads a subrange from a memory file into a buffer at an offset.
TEST_P(FileTest, MemoryFileReadSubrange) {
  const iree_device_size_t file_size = 512;
  const iree_device_size_t read_offset = 128;
  const iree_device_size_t buffer_offset = 64;
  const iree_device_size_t read_length = 256;

  // Create a file with sequential byte values.
  void* file_contents = NULL;
  IREE_ASSERT_OK(iree_allocator_malloc_aligned(iree_allocator_system(),
                                               file_size, kMinimumAlignment, 0,
                                               &file_contents));
  for (size_t i = 0; i < file_size; ++i) {
    static_cast<uint8_t*>(file_contents)[i] = static_cast<uint8_t>(i & 0xFF);
  }
  iree_io_file_handle_release_callback_t release_callback = {
      +[](void* user_data, iree_io_file_handle_primitive_t) {
        iree_allocator_free_aligned(iree_allocator_system(), user_data);
      },
      file_contents,
  };
  iree_io_file_handle_t* handle = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ, iree_make_byte_span(file_contents, file_size),
      release_callback, iree_allocator_system(), &handle));
  iree_hal_file_t* file = NULL;
  IREE_ASSERT_OK(iree_hal_file_import(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                      IREE_HAL_MEMORY_ACCESS_READ, handle,
                                      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(handle);

  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);

  QueueReadAndWait(file, read_offset, buffer, buffer_offset, read_length);

  // Verify the read subrange.
  std::vector<uint8_t> expected(read_length);
  for (size_t i = 0; i < read_length; ++i) {
    expected[i] = static_cast<uint8_t>((read_offset + i) & 0xFF);
  }
  EXPECT_THAT(ReadBufferContents(buffer, buffer_offset, read_length),
              ContainerEq(expected));

  // Verify buffer regions outside the read are untouched.
  std::vector<uint8_t> before(buffer_offset, 0x00);
  EXPECT_THAT(ReadBufferContents(buffer, 0, buffer_offset),
              ContainerEq(before));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Writes buffer contents to a memory file at an offset, verifying only the
// target region is modified.
TEST_P(FileTest, MemoryFileWriteSubrange) {
  const iree_device_size_t file_size = 256;
  const iree_device_size_t write_offset = 64;
  const iree_device_size_t write_length = 128;

  iree_hal_file_t* file = NULL;
  CreatePatternedMemoryFile(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file_size,
      0x00u, &file);

  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(write_length, 0xBB, &buffer);

  QueueWriteAndWait(buffer, 0, file, write_offset, write_length);

  iree_hal_buffer_t* storage = iree_hal_file_storage_buffer(file);
  ASSERT_NE(storage, nullptr);

  // Verify the written region.
  std::vector<uint8_t> expected_written(write_length, 0xBB);
  EXPECT_THAT(ReadBufferContents(storage, write_offset, write_length),
              ContainerEq(expected_written));

  // Verify boundaries are untouched.
  std::vector<uint8_t> expected_zero_before(write_offset, 0x00);
  EXPECT_THAT(ReadBufferContents(storage, 0, write_offset),
              ContainerEq(expected_zero_before));
  std::vector<uint8_t> expected_zero_after(
      file_size - write_offset - write_length, 0x00);
  EXPECT_THAT(ReadBufferContents(storage, write_offset + write_length,
                                 file_size - write_offset - write_length),
              ContainerEq(expected_zero_after));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Reads from a memory file, modifies the buffer (fill), then writes back.
// Verifies the read-modify-write pattern that parameter loading uses.
TEST_P(FileTest, MemoryFileReadModifyWrite) {
  const iree_device_size_t file_size = 128;

  // Source file with known data.
  iree_hal_file_t* source = NULL;
  CreatePatternedMemoryFile(IREE_HAL_MEMORY_ACCESS_READ, file_size, 0xAAu,
                            &source);

  // Target file for modified data.
  iree_hal_file_t* target = NULL;
  CreatePatternedMemoryFile(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file_size,
      0x00u, &target);

  // Read from source into buffer.
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);
  QueueReadAndWait(source, 0, buffer, 0, file_size);

  // Modify the first half of the buffer via queue_fill.
  {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    uint8_t pattern = 0xFF;
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, buffer, 0,
        file_size / 2, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Write modified buffer to target file.
  QueueWriteAndWait(buffer, 0, target, 0, file_size);

  // Verify: first half should be 0xFF (modified), second half 0xAA (from
  // source).
  iree_hal_buffer_t* target_storage = iree_hal_file_storage_buffer(target);
  ASSERT_NE(target_storage, nullptr);

  std::vector<uint8_t> first_half(file_size / 2, 0xFF);
  EXPECT_THAT(ReadBufferContents(target_storage, 0, file_size / 2),
              ContainerEq(first_half));
  std::vector<uint8_t> second_half(file_size / 2, 0xAA);
  EXPECT_THAT(ReadBufferContents(target_storage, file_size / 2, file_size / 2),
              ContainerEq(second_half));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target);
  iree_hal_file_release(source);
}

//===----------------------------------------------------------------------===//
// FD file ordering and pipeline tests
//===----------------------------------------------------------------------===//

#if IREE_FILE_IO_ENABLE

// Writes a subrange to an FD file, verifying only the target region is
// modified.
TEST_P(FileTest, FdFileWriteSubrange) {
  const iree_device_size_t file_size = 4096;
  const iree_device_size_t write_offset = 1024;
  const iree_device_size_t write_length = 512;
  std::string path = CreatePatternedTempFile(file_size, 0x00);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
               &file);

  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(write_length, 0xDD, &buffer);

  QueueWriteAndWait(buffer, 0, file, write_offset, write_length);
  iree_hal_file_release(file);
  iree_hal_buffer_release(buffer);

  // Re-read the entire file and verify.
  file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);
  QueueReadAndWait(file, 0, buffer, 0, file_size);

  // Before the write region: zeros.
  std::vector<uint8_t> before(write_offset, 0x00);
  EXPECT_THAT(ReadBufferContents(buffer, 0, write_offset), ContainerEq(before));

  // The write region: 0xDD.
  std::vector<uint8_t> written(write_length, 0xDD);
  EXPECT_THAT(ReadBufferContents(buffer, write_offset, write_length),
              ContainerEq(written));

  // After the write region: zeros.
  std::vector<uint8_t> after(file_size - write_offset - write_length, 0x00);
  EXPECT_THAT(ReadBufferContents(buffer, write_offset + write_length,
                                 file_size - write_offset - write_length),
              ContainerEq(after));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Reads from one FD file, writes to another FD file. Tests the full
// file→buffer→file pipeline through the FD (async I/O) path.
TEST_P(FileTest, FdFileChainedReadWrite) {
  const iree_device_size_t file_size = 4096;
  std::string source_path = CreatePatternedTempFile(file_size, 0xCC);
  std::string target_path = CreatePatternedTempFile(file_size, 0x00);

  iree_hal_file_t* source_file = NULL;
  ImportFdFile(source_path, IREE_HAL_MEMORY_ACCESS_READ, &source_file);
  iree_hal_file_t* target_file = NULL;
  ImportFdFile(target_path,
               IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
               &target_file);

  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);

  // Chain read→write via semaphore timeline.
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  // Read: signal@1.
  {
    iree_hal_fence_t* wait_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 0ull, iree_allocator_system(), &wait_fence));
    iree_hal_fence_t* signal_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 1ull, iree_allocator_system(), &signal_fence));
    IREE_ASSERT_OK(iree_hal_device_queue_read(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), source_file, 0, buffer, 0,
        file_size, IREE_HAL_READ_FLAG_NONE));
    iree_hal_fence_release(wait_fence);
    iree_hal_fence_release(signal_fence);
  }

  // Write: wait@1, signal@2.
  {
    iree_hal_fence_t* wait_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 1ull, iree_allocator_system(), &wait_fence));
    iree_hal_fence_t* signal_fence = NULL;
    IREE_ASSERT_OK(iree_hal_fence_create_at(
        semaphore, 2ull, iree_allocator_system(), &signal_fence));
    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), buffer, 0, target_file, 0,
        file_size, IREE_HAL_WRITE_FLAG_NONE));
    iree_hal_fence_release(wait_fence);
    iree_hal_fence_release(signal_fence);
  }

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      semaphore, 2ull, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(semaphore);
  iree_hal_file_release(target_file);
  iree_hal_file_release(source_file);
  iree_hal_buffer_release(buffer);

  // Verify: read target file back and check contents.
  target_file = NULL;
  ImportFdFile(target_path, IREE_HAL_MEMORY_ACCESS_READ, &target_file);
  buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);
  QueueReadAndWait(target_file, 0, buffer, 0, file_size);

  std::vector<uint8_t> expected(file_size, 0xCC);
  EXPECT_THAT(ReadBufferContents(buffer, 0, file_size), ContainerEq(expected));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(target_file);
}

// Reads from an FD file, modifies the buffer, writes back to the same file.
// This is the read-modify-write pattern for mutable parameter files.
TEST_P(FileTest, FdFileReadModifyWrite) {
  const iree_device_size_t file_size = 4096;
  std::string path = CreatePatternedTempFile(file_size, 0x11);

  // Read the file.
  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);
  QueueReadAndWait(file, 0, buffer, 0, file_size);
  iree_hal_file_release(file);

  // Modify: overwrite second half with 0xFF via queue_fill.
  {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    uint8_t pattern = 0xFF;
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, buffer,
        file_size / 2, file_size / 2, &pattern, sizeof(pattern),
        IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Write back to the same file.
  file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
               &file);
  QueueWriteAndWait(buffer, 0, file, 0, file_size);
  iree_hal_file_release(file);
  iree_hal_buffer_release(buffer);

  // Verify: re-read and check both halves.
  file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);
  QueueReadAndWait(file, 0, buffer, 0, file_size);

  std::vector<uint8_t> first_half(file_size / 2, 0x11);
  EXPECT_THAT(ReadBufferContents(buffer, 0, file_size / 2),
              ContainerEq(first_half));
  std::vector<uint8_t> second_half(file_size / 2, 0xFF);
  EXPECT_THAT(ReadBufferContents(buffer, file_size / 2, file_size / 2),
              ContainerEq(second_half));

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Reads a large file to exercise chunked I/O paths in implementations with
// bounded staging.
TEST_P(FileTest, FdFileLargeRead) {
  const iree_device_size_t file_size = kLargeFdFileSize;
  std::vector<uint8_t> file_data = MakePatternData(file_size);
  std::string path = CreateTempFileWithContents(file_data.data(), file_size);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0x00, &buffer);

  QueueReadAndWait(file, 0, buffer, 0, file_size);

  auto contents = ReadBufferContents(buffer, 0, file_size);
  ASSERT_EQ(contents.size(), file_data.size());
  EXPECT_EQ(contents[0], file_data[0]);
  EXPECT_EQ(contents[1000], file_data[1000]);
  EXPECT_EQ(contents[file_size / 2], file_data[file_size / 2]);
  EXPECT_EQ(contents[file_size - 1], file_data[file_size - 1]);
  ExpectByteRangeMatches(contents, 0, file_data);

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Reads a large subrange into a non-zero target offset, verifying both chunk
// boundary math and that target padding is untouched.
TEST_P(FileTest, FdFileLargeReadSubrangeWithPadding) {
  const iree_device_size_t file_size = kLargeFdFileSize;
  const iree_device_size_t source_offset = 1237;
  const iree_device_size_t transfer_length = file_size - source_offset - 8191;
  const iree_device_size_t target_offset = 4099;
  const iree_device_size_t target_suffix_length = 3071;
  const iree_device_size_t buffer_size =
      target_offset + transfer_length + target_suffix_length;
  std::vector<uint8_t> file_data = MakePatternData(file_size);
  std::string path = CreateTempFileWithContents(file_data.data(), file_size);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ, &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(buffer_size, 0x5A, &buffer);

  QueueReadAndWait(file, source_offset, buffer, target_offset, transfer_length);

  ExpectBufferRangeRepeated(buffer, 0, target_offset, 0x5A);
  std::vector<uint8_t> expected(
      file_data.begin() + static_cast<size_t>(source_offset),
      file_data.begin() + static_cast<size_t>(source_offset + transfer_length));
  ExpectBufferRangeMatches(buffer, target_offset, expected);
  ExpectBufferRangeRepeated(buffer, target_offset + transfer_length,
                            target_suffix_length, 0x5A);

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Writes a large file to exercise chunked I/O paths in implementations with
// bounded staging.
TEST_P(FileTest, FdFileLargeWrite) {
  const iree_device_size_t file_size = kLargeFdFileSize;
  std::string path = CreatePatternedTempFile(file_size, 0x00);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
               &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(file_size, 0xA7, &buffer);

  QueueWriteAndWait(buffer, 0, file, 0, file_size);

  std::vector<uint8_t> contents = ReadTempFileContents(path, file_size);
  ASSERT_EQ(contents.size(), file_size);
  EXPECT_EQ(contents[0], 0xA7);
  EXPECT_EQ(contents[1000], 0xA7);
  EXPECT_EQ(contents[file_size / 2], 0xA7);
  EXPECT_EQ(contents[file_size - 1], 0xA7);
  ExpectByteRangeRepeated(contents, 0, contents.size(), 0xA7);

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

// Writes a large subrange from a non-zero source offset, verifying both chunk
// boundary math and that file padding is untouched.
TEST_P(FileTest, FdFileLargeWriteSubrangeWithPadding) {
  const iree_device_size_t file_size = kLargeFdFileSize;
  const iree_device_size_t target_offset = 2049;
  const iree_device_size_t transfer_length = file_size - target_offset - 4096;
  const iree_device_size_t source_offset = 3073;
  const iree_device_size_t source_suffix_length = 4091;
  const iree_device_size_t buffer_size =
      source_offset + transfer_length + source_suffix_length;
  std::string path = CreatePatternedTempFile(file_size, 0x11);

  iree_hal_file_t* file = NULL;
  ImportFdFile(path, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
               &file);
  iree_hal_buffer_t* buffer = NULL;
  CreatePatternedDeviceBuffer(buffer_size, 0x3C, &buffer);
  FillDeviceBufferRange(buffer, source_offset, transfer_length, 0xE6);

  QueueWriteAndWait(buffer, source_offset, file, target_offset,
                    transfer_length);

  std::vector<uint8_t> contents = ReadTempFileContents(path, file_size);
  ASSERT_EQ(contents.size(), file_size);
  ExpectByteRangeRepeated(contents, 0, target_offset, 0x11);
  ExpectByteRangeRepeated(contents, target_offset, transfer_length, 0xE6);
  ExpectByteRangeRepeated(contents, target_offset + transfer_length,
                          file_size - target_offset - transfer_length, 0x11);

  iree_hal_buffer_release(buffer);
  iree_hal_file_release(file);
}

#endif  // IREE_FILE_IO_ENABLE

CTS_REGISTER_TEST_SUITE(FileTest);

}  // namespace iree::hal::cts
