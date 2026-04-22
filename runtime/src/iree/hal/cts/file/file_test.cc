// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_handle.h"

#if IREE_FILE_IO_ENABLE
#if defined(IREE_PLATFORM_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS
#endif  // IREE_FILE_IO_ENABLE

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {

constexpr iree_device_size_t kFileAlignment = 128;

enum class FileProviderKind {
  kMemory,
  kFd,
};

struct FileProvider {
  // Human-readable provider name used in scoped test traces.
  const char* name;
  // Backing implementation shape exercised by the provider.
  FileProviderKind kind;
};

const FileProvider kFileProviders[] = {
    {
        "memory_file",
        FileProviderKind::kMemory,
    },
    {
        "fd_file",
        FileProviderKind::kFd,
    },
};

void ReleaseAlignedAllocation(void* user_data,
                              iree_io_file_handle_primitive_t primitive) {
  (void)primitive;
  iree_allocator_free_aligned(iree_allocator_system(), user_data);
}

#if IREE_FILE_IO_ENABLE
int FileDescriptorFromStream(FILE* stream) {
#if defined(IREE_PLATFORM_WINDOWS)
  return _fileno(stream);
#else
  return fileno(stream);
#endif  // IREE_PLATFORM_WINDOWS
}
#endif  // IREE_FILE_IO_ENABLE

}  // namespace

class FileTest : public CtsTestBase<> {
 protected:
  class TestFile {
   public:
    TestFile() = default;
    ~TestFile() { Reset(); }

    TestFile(TestFile&& other) noexcept { MoveFrom(std::move(other)); }

    TestFile& operator=(TestFile&& other) noexcept {
      if (this != &other) {
        Reset();
        MoveFrom(std::move(other));
      }
      return *this;
    }

    TestFile(const TestFile&) = delete;
    TestFile& operator=(const TestFile&) = delete;

    iree_hal_file_t* get() const { return file_; }
    iree_hal_file_t** out_file() { return &file_; }

    void SetMemoryContents(uint8_t* memory_contents) {
      memory_contents_ = memory_contents;
    }

    void SetStream(FILE* stream) {
#if IREE_FILE_IO_ENABLE
      stream_ = stream;
#else
      (void)stream;
#endif  // IREE_FILE_IO_ENABLE
    }

    std::vector<uint8_t> ReadContents(iree_device_size_t offset,
                                      iree_device_size_t length) const {
      std::vector<uint8_t> contents(static_cast<size_t>(length));
      if (memory_contents_) {
        std::memcpy(contents.data(), memory_contents_ + offset,
                    static_cast<size_t>(length));
        return contents;
      }

#if IREE_FILE_IO_ENABLE
      if (stream_) {
        EXPECT_EQ(0, std::fflush(stream_));
        EXPECT_EQ(0, std::fseek(stream_, static_cast<long>(offset), SEEK_SET));
        const size_t bytes_read = std::fread(
            contents.data(), 1, static_cast<size_t>(length), stream_);
        EXPECT_EQ(static_cast<size_t>(length), bytes_read);
        EXPECT_EQ(0, std::ferror(stream_));
        return contents;
      }
#endif  // IREE_FILE_IO_ENABLE

      ADD_FAILURE() << "test file has no inspectable backing storage";
      return contents;
    }

    void Reset() {
      iree_hal_file_release(file_);
      file_ = nullptr;
#if IREE_FILE_IO_ENABLE
      if (stream_) {
        std::fclose(stream_);
        stream_ = nullptr;
      }
#endif  // IREE_FILE_IO_ENABLE
      memory_contents_ = nullptr;
    }

   private:
    void MoveFrom(TestFile&& other) {
      file_ = other.file_;
      other.file_ = nullptr;
      memory_contents_ = other.memory_contents_;
      other.memory_contents_ = nullptr;
#if IREE_FILE_IO_ENABLE
      stream_ = other.stream_;
      other.stream_ = nullptr;
#endif  // IREE_FILE_IO_ENABLE
    }

    // Imported HAL file under test.
    iree_hal_file_t* file_ = nullptr;
    // Host allocation backing memory_file providers, unowned.
    uint8_t* memory_contents_ = nullptr;
#if IREE_FILE_IO_ENABLE
    // Temporary stream holding fd_file provider contents.
    FILE* stream_ = nullptr;
#endif  // IREE_FILE_IO_ENABLE
  };

  template <typename Body>
  void ForEachAvailableProvider(Body body) {
    int available_count = 0;
    for (const auto& provider : kFileProviders) {
      SCOPED_TRACE(provider.name);
      if (!ProviderAvailable(provider)) continue;
      ++available_count;
      body(provider);
    }
    EXPECT_GT(available_count, 0);
  }

  bool ProviderAvailable(const FileProvider& provider) {
    if (provider.kind == FileProviderKind::kMemory) return true;
#if IREE_FILE_IO_ENABLE
    return true;
#else
    return false;
#endif  // IREE_FILE_IO_ENABLE
  }

  bool TryCreateTestFile(const FileProvider& provider,
                         iree_hal_memory_access_t access,
                         const std::vector<uint8_t>& initial_contents,
                         TestFile* out_file) {
    iree_status_t status =
        CreateTestFileStatus(provider, access, initial_contents, out_file);
    if (ProviderSkipped(provider, status)) return false;
    if (!iree_status_is_ok(status)) {
      IREE_EXPECT_OK(status);
      return false;
    }
    return true;
  }

  Ref<iree_hal_buffer_t> CreateBufferWithData(
      const std::vector<uint8_t>& contents) {
    Ref<iree_hal_buffer_t> buffer;
    CreateDeviceBufferWithData(contents.data(),
                               static_cast<iree_device_size_t>(contents.size()),
                               buffer.out());
    return buffer;
  }

  Ref<iree_hal_buffer_t> CreateZeroedBuffer(iree_device_size_t length) {
    Ref<iree_hal_buffer_t> buffer;
    CreateZeroedDeviceBuffer(length, buffer.out());
    return buffer;
  }

  void QueueReadAndWait(iree_hal_file_t* source_file,
                        iree_device_size_t source_offset,
                        iree_hal_buffer_t* target_buffer,
                        iree_device_size_t target_offset,
                        iree_device_size_t length) {
    SemaphoreList empty_wait;
    SemaphoreList read_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_read(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, read_signal,
        source_file, source_offset, target_buffer, target_offset, length,
        IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void QueueWriteAndWait(iree_hal_buffer_t* source_buffer,
                         iree_device_size_t source_offset,
                         iree_hal_file_t* target_file,
                         iree_device_size_t target_offset,
                         iree_device_size_t length) {
    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, write_signal,
        source_buffer, source_offset, target_file, target_offset, length,
        IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        write_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void QueueWriteThenRead(iree_hal_buffer_t* source_buffer,
                          iree_device_size_t source_offset,
                          iree_hal_file_t* file, iree_device_size_t file_offset,
                          iree_hal_buffer_t* target_buffer,
                          iree_device_size_t target_offset,
                          iree_device_size_t length) {
    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    SemaphoreList read_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_write(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, write_signal,
        source_buffer, source_offset, file, file_offset, length,
        IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_device_queue_read(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, write_signal, read_signal, file,
        file_offset, target_buffer, target_offset, length,
        IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

 private:
  iree_status_t CreateTestFileStatus(
      const FileProvider& provider, iree_hal_memory_access_t access,
      const std::vector<uint8_t>& initial_contents, TestFile* out_file) {
    out_file->Reset();
    switch (provider.kind) {
      case FileProviderKind::kMemory:
        return CreateMemoryTestFile(access, initial_contents, out_file);
      case FileProviderKind::kFd:
        return CreateFdTestFile(access, initial_contents, out_file);
    }
    return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
  }

  bool ProviderSkipped(const FileProvider& provider, iree_status_t status) {
    if (iree_status_is_ok(status)) return false;
    const iree_status_code_t code = iree_status_code(status);
    if (provider.kind == FileProviderKind::kFd &&
        (code == IREE_STATUS_UNAVAILABLE ||
         code == IREE_STATUS_UNIMPLEMENTED)) {
      iree_status_free(status);
      return true;
    }
    return false;
  }

  iree_status_t CreateMemoryTestFile(
      iree_hal_memory_access_t access,
      const std::vector<uint8_t>& initial_contents, TestFile* out_file) {
    void* file_contents = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
        iree_allocator_system(), initial_contents.size(), kFileAlignment, 0,
        &file_contents));
    std::memcpy(file_contents, initial_contents.data(),
                initial_contents.size());

    iree_io_file_handle_release_callback_t release_callback = {
        ReleaseAlignedAllocation,
        file_contents,
    };
    iree_io_file_handle_t* handle = nullptr;
    iree_status_t status = iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(file_contents, initial_contents.size()),
        release_callback, iree_allocator_system(), &handle);
    if (iree_status_is_ok(status)) {
      status = iree_hal_file_import(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, access, handle,
          IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file->out_file());
    }
    if (iree_status_is_ok(status)) {
      out_file->SetMemoryContents(static_cast<uint8_t*>(file_contents));
    } else if (!handle) {
      iree_io_file_handle_primitive_t primitive = {};
      release_callback.fn(release_callback.user_data, primitive);
    }
    iree_io_file_handle_release(handle);
    return status;
  }

  iree_status_t CreateFdTestFile(iree_hal_memory_access_t access,
                                 const std::vector<uint8_t>& initial_contents,
                                 TestFile* out_file) {
#if IREE_FILE_IO_ENABLE
    FILE* stream = std::tmpfile();
    if (!stream) {
      return iree_status_from_code(iree_status_code_from_errno(errno));
    }
    const size_t bytes_written = std::fwrite(initial_contents.data(), 1,
                                             initial_contents.size(), stream);
    if (bytes_written != initial_contents.size() || std::fflush(stream) != 0) {
      const iree_status_code_t status_code = iree_status_code_from_errno(errno);
      std::fclose(stream);
      return iree_status_from_code(status_code);
    }

    const int fd = FileDescriptorFromStream(stream);
    if (fd == -1) {
      const iree_status_code_t status_code = iree_status_code_from_errno(errno);
      std::fclose(stream);
      return iree_status_from_code(status_code);
    }

    iree_io_file_handle_t* handle = nullptr;
    iree_status_t status = iree_io_file_handle_open_fd(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS,
        fd, iree_allocator_system(), &handle);
    if (iree_status_is_ok(status)) {
      status = iree_hal_file_import(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, access, handle,
          IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file->out_file());
    }
    iree_io_file_handle_release(handle);

    if (iree_status_is_ok(status)) {
      out_file->SetStream(stream);
    } else {
      std::fclose(stream);
    }
    return status;
#else
    (void)access;
    (void)initial_contents;
    (void)out_file;
    return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
#endif  // IREE_FILE_IO_ENABLE
  }
};

TEST_P(FileTest, ReadEntireFile) {
  ForEachAvailableProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 128;
    const std::vector<uint8_t> file_contents(file_length, 0xDEu);
    TestFile file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ, file_contents,
                           &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(file_length);
    QueueReadAndWait(file.get(), /*source_offset=*/0, target_buffer.get(),
                     /*target_offset=*/0, file_length);

    EXPECT_THAT(ReadBufferBytes(target_buffer.get(), /*offset=*/0, file_length),
                ContainerEq(file_contents));
  });
}

TEST_P(FileTest, ReadSubrangeWithOffsets) {
  ForEachAvailableProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 512;
    const iree_device_size_t target_length = 384;
    const iree_device_size_t source_offset = 97;
    const iree_device_size_t target_offset = 13;
    const iree_device_size_t read_length = 211;
    const std::vector<uint8_t> file_contents =
        MakeDeterministicBytes(file_length);
    TestFile file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ, file_contents,
                           &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(target_length);
    QueueReadAndWait(file.get(), source_offset, target_buffer.get(),
                     target_offset, read_length);

    std::vector<uint8_t> expected(static_cast<size_t>(target_length), 0);
    std::copy(file_contents.begin() + source_offset,
              file_contents.begin() + source_offset + read_length,
              expected.begin() + target_offset);
    EXPECT_THAT(
        ReadBufferBytes(target_buffer.get(), /*offset=*/0, target_length),
        ContainerEq(expected));
  });
}

TEST_P(FileTest, WriteSubrangeWithOffsets) {
  ForEachAvailableProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 512;
    const iree_device_size_t source_length = 384;
    const iree_device_size_t source_offset = 23;
    const iree_device_size_t target_offset = 41;
    const iree_device_size_t write_length = 307;
    std::vector<uint8_t> file_contents = MakeDeterministicBytes(file_length);
    std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(source_length);
    TestFile file;
    if (!TryCreateTestFile(
            provider,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
            file_contents, &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> source_buffer =
        CreateBufferWithData(source_contents);
    QueueWriteAndWait(source_buffer.get(), source_offset, file.get(),
                      target_offset, write_length);

    std::copy(source_contents.begin() + source_offset,
              source_contents.begin() + source_offset + write_length,
              file_contents.begin() + target_offset);
    EXPECT_THAT(file.ReadContents(/*offset=*/0, file_length),
                ContainerEq(file_contents));
  });
}

TEST_P(FileTest, ChainedWriteThenReadWithSemaphores) {
  ForEachAvailableProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 768;
    const iree_device_size_t source_length = 512;
    const iree_device_size_t target_length = 512;
    const iree_device_size_t source_offset = 19;
    const iree_device_size_t file_offset = 123;
    const iree_device_size_t target_offset = 37;
    const iree_device_size_t transfer_length = 333;
    const std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(source_length);
    const std::vector<uint8_t> file_contents(file_length, 0);
    TestFile file;
    if (!TryCreateTestFile(
            provider,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
            file_contents, &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> source_buffer =
        CreateBufferWithData(source_contents);
    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(target_length);
    QueueWriteThenRead(source_buffer.get(), source_offset, file.get(),
                       file_offset, target_buffer.get(), target_offset,
                       transfer_length);

    std::vector<uint8_t> expected(static_cast<size_t>(target_length), 0);
    std::copy(source_contents.begin() + source_offset,
              source_contents.begin() + source_offset + transfer_length,
              expected.begin() + target_offset);
    EXPECT_THAT(
        ReadBufferBytes(target_buffer.get(), /*offset=*/0, target_length),
        ContainerEq(expected));
  });
}

TEST_P(FileTest, LargeRangeRoundTrip) {
  ForEachAvailableProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 256 * 1024 + 31;
    const std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(file_length);
    const std::vector<uint8_t> file_contents(file_length, 0);
    TestFile file;
    if (!TryCreateTestFile(
            provider,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
            file_contents, &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> source_buffer =
        CreateBufferWithData(source_contents);
    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(file_length);
    QueueWriteThenRead(source_buffer.get(), /*source_offset=*/0, file.get(),
                       /*file_offset=*/0, target_buffer.get(),
                       /*target_offset=*/0, file_length);

    EXPECT_THAT(ReadBufferBytes(target_buffer.get(), /*offset=*/0, file_length),
                ContainerEq(source_contents));
  });
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(FileTest, {"file_io"}, {});

}  // namespace iree::hal::cts
