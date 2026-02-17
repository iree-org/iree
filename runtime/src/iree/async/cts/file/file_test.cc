// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for async file I/O operations.
//
// Tests cover the full file lifecycle: open, read, write, close. All operations
// use positioned (pread/pwrite) semantics, so concurrent reads/writes at
// different offsets are safe.

#include "iree/async/file.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/operations/file.h"
#include "iree/async/span.h"

// Platform headers for temp file creation.
#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#endif

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Test fixture with temp file helpers
//===----------------------------------------------------------------------===//

class FileTest : public CtsTestBase<> {
 protected:
  void TearDown() override {
    // Clean up any temp files created during the test.
    for (const auto& path : temp_paths_) {
#if defined(IREE_PLATFORM_WINDOWS)
      DeleteFileA(path.c_str());
#else
      unlink(path.c_str());
#endif
    }
    CtsTestBase::TearDown();
  }

  // Creates a temp file with the given contents, returns the path.
  // The file is closed after writing; callers reopen it via file_open or
  // file_import as needed.
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
    char temp_path[] = "/tmp/iree_cts_file_XXXXXX";
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

  // Creates an empty temp file and returns the path.
  std::string CreateEmptyTempFile() {
    return CreateTempFileWithContents("", 0);
  }

  // Opens a temp file for read+write and imports it into the proactor.
  // Returns the imported file handle. The caller must submit a close operation
  // or release the file when done.
  iree_async_file_t* ImportTempFileForReadWrite(const std::string& path) {
#if defined(IREE_PLATFORM_WINDOWS)
    HANDLE handle =
        CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
                    NULL, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    EXPECT_NE(handle, INVALID_HANDLE_VALUE);
    iree_async_primitive_t primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)handle);
#else
    int fd = open(path.c_str(), O_RDWR);
    EXPECT_GE(fd, 0);
    iree_async_primitive_t primitive = iree_async_primitive_from_fd(fd);
#endif
    iree_async_file_t* file = nullptr;
    IREE_EXPECT_OK(iree_async_file_import(proactor_, primitive, &file));
    return file;
  }

  // Opens a temp file for reading only and imports it into the proactor.
  iree_async_file_t* ImportTempFileForRead(const std::string& path) {
#if defined(IREE_PLATFORM_WINDOWS)
    HANDLE handle =
        CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                    OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
    EXPECT_NE(handle, INVALID_HANDLE_VALUE);
    iree_async_primitive_t primitive =
        iree_async_primitive_from_win32_handle((uintptr_t)handle);
#else
    int fd = open(path.c_str(), O_RDONLY);
    EXPECT_GE(fd, 0);
    iree_async_primitive_t primitive = iree_async_primitive_from_fd(fd);
#endif
    iree_async_file_t* file = nullptr;
    IREE_EXPECT_OK(iree_async_file_import(proactor_, primitive, &file));
    return file;
  }

 private:
  std::vector<std::string> temp_paths_;
};

//===----------------------------------------------------------------------===//
// File open tests
//===----------------------------------------------------------------------===//

// Open an existing file for reading via async open operation.
TEST_P(FileTest, OpenExistingFileForRead) {
  const char kTestData[] = "hello async file";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_open_operation_t open_op;
  memset(&open_op, 0, sizeof(open_op));
  open_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_op.path = path.c_str();
  open_op.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  open_op.opened_file = nullptr;

  CompletionTracker tracker;
  open_op.base.completion_fn = CompletionTracker::Callback;
  open_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  ASSERT_NE(open_op.opened_file, nullptr);

  // Clean up: close via release (file was opened, not submitted for close).
  iree_async_file_release(open_op.opened_file);
}

// Open a non-existent file without CREATE flag fails with NOT_FOUND.
TEST_P(FileTest, OpenNonExistentFileFailsNotFound) {
  iree_async_file_open_operation_t open_op;
  memset(&open_op, 0, sizeof(open_op));
  open_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_op.path = "/tmp/iree_cts_file_nonexistent_should_not_exist_12345";
  open_op.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  open_op.opened_file = nullptr;

  CompletionTracker tracker;
  open_op.base.completion_fn = CompletionTracker::Callback;
  open_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, tracker.ConsumeStatus());
  EXPECT_EQ(open_op.opened_file, nullptr);
}

// Open with CREATE flag creates a new file.
TEST_P(FileTest, OpenWithCreateCreatesNewFile) {
  // Use a unique path that doesn't exist.
  std::string path = CreateEmptyTempFile();
  // Remove it so the open+CREATE can recreate it.
#if defined(IREE_PLATFORM_WINDOWS)
  DeleteFileA(path.c_str());
#else
  unlink(path.c_str());
#endif

  iree_async_file_open_operation_t open_op;
  memset(&open_op, 0, sizeof(open_op));
  open_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_OPEN;
  open_op.path = path.c_str();
  open_op.open_flags =
      IREE_ASYNC_FILE_OPEN_FLAG_WRITE | IREE_ASYNC_FILE_OPEN_FLAG_CREATE;
  open_op.opened_file = nullptr;

  CompletionTracker tracker;
  open_op.base.completion_fn = CompletionTracker::Callback;
  open_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &open_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  ASSERT_NE(open_op.opened_file, nullptr);

  iree_async_file_release(open_op.opened_file);
}

//===----------------------------------------------------------------------===//
// File read tests
//===----------------------------------------------------------------------===//

// Read entire file contents at offset 0.
TEST_P(FileTest, ReadEntireFile) {
  const char kTestData[] = "hello async world";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, strlen(kTestData));
  EXPECT_EQ(memcmp(read_buffer, kTestData, strlen(kTestData)), 0);

  iree_async_file_release(file);
}

// Read at a non-zero offset within the file.
TEST_P(FileTest, ReadAtOffset) {
  const char kTestData[] = "abcdefghij";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[5] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 5;  // Start at 'f'.
  read_op.buffer = iree_async_span_from_ptr(read_buffer, 5);
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, 5u);
  EXPECT_EQ(memcmp(read_buffer, "fghij", 5), 0);

  iree_async_file_release(file);
}

// Read past EOF produces a short read (bytes_read < requested).
TEST_P(FileTest, ReadPastEOFShortRead) {
  const char kTestData[] = "short";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[1024] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  // Short read: file has 5 bytes, we requested 1024.
  EXPECT_EQ(read_op.bytes_read, strlen(kTestData));
  EXPECT_EQ(memcmp(read_buffer, kTestData, strlen(kTestData)), 0);

  iree_async_file_release(file);
}

// Read at offset beyond EOF returns 0 bytes (EOF).
TEST_P(FileTest, ReadBeyondEOFReturnsZeroBytes) {
  const char kTestData[] = "data";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 1000;  // Well past the 4-byte file.
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, 0u);

  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// File write tests
//===----------------------------------------------------------------------===//

// Write data to a file and verify by reading it back.
TEST_P(FileTest, WriteAndReadBack) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Write data at offset 0.
  const char kWriteData[] = "written by proactor";
  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 0;
  write_op.buffer = iree_async_span_from_ptr(const_cast<char*>(kWriteData),
                                             strlen(kWriteData));
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(write_tracker.call_count, 1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());
  EXPECT_EQ(write_op.bytes_written, strlen(kWriteData));

  // Read it back at offset 0.
  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(read_tracker.call_count, 1);
  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, strlen(kWriteData));
  EXPECT_EQ(memcmp(read_buffer, kWriteData, strlen(kWriteData)), 0);

  iree_async_file_release(file);
}

// Write at a non-zero offset, then read the full file to verify.
TEST_P(FileTest, WriteAtOffset) {
  const char kInitialData[] = "AAAAABBBBB";
  std::string path =
      CreateTempFileWithContents(kInitialData, strlen(kInitialData));

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Overwrite the "BBBBB" portion (offset 5).
  const char kOverwrite[] = "CCCCC";
  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 5;
  write_op.buffer = iree_async_span_from_ptr(const_cast<char*>(kOverwrite),
                                             strlen(kOverwrite));
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(write_tracker.call_count, 1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());
  EXPECT_EQ(write_op.bytes_written, strlen(kOverwrite));

  // Read full file and verify.
  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(read_tracker.call_count, 1);
  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, 10u);
  EXPECT_EQ(memcmp(read_buffer, "AAAAACCCCC", 10), 0);

  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// File close tests
//===----------------------------------------------------------------------===//

// Async close completes successfully and the callback fires.
TEST_P(FileTest, AsyncClose) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  iree_async_file_close_operation_t close_op;
  memset(&close_op, 0, sizeof(close_op));
  close_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE;
  close_op.file = file;

  CompletionTracker tracker;
  close_op.base.completion_fn = CompletionTracker::Callback;
  close_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());
  // File reference was consumed by the close operation. Do not release.
}

// Write, then close via async close, verify the write persisted.
TEST_P(FileTest, WriteFlushClose) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Write data.
  const char kData[] = "persist me";
  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 0;
  write_op.buffer =
      iree_async_span_from_ptr(const_cast<char*>(kData), strlen(kData));
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());

  // Close the file async.
  iree_async_file_close_operation_t close_op;
  memset(&close_op, 0, sizeof(close_op));
  close_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE;
  close_op.file = file;

  CompletionTracker close_tracker;
  close_op.base.completion_fn = CompletionTracker::Callback;
  close_op.base.user_data = &close_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &close_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));
  IREE_EXPECT_OK(close_tracker.ConsumeStatus());

  // Re-open and verify the data persisted.
  iree_async_file_t* reopened = ImportTempFileForRead(path);
  ASSERT_NE(reopened, nullptr);

  uint8_t read_buffer[64] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = reopened;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));

  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, strlen(kData));
  EXPECT_EQ(memcmp(read_buffer, kData, strlen(kData)), 0);

  iree_async_file_release(reopened);
}

//===----------------------------------------------------------------------===//
// File import and lifecycle tests
//===----------------------------------------------------------------------===//

// Import a file, retain, release -- verify reference counting works.
TEST_P(FileTest, ImportRetainRelease) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Retain bumps refcount.
  iree_async_file_retain(file);

  // First release: refcount goes from 2 to 1.
  iree_async_file_release(file);

  // File should still be usable. Verify with a read (0-byte read at offset 0).
  uint8_t read_buffer[1] = {0};
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer, sizeof(read_buffer));
  read_op.bytes_read = 0;

  CompletionTracker tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1, /*total_budget=*/iree_make_duration_ms(500));
  EXPECT_EQ(tracker.call_count, 1);
  IREE_EXPECT_OK(tracker.ConsumeStatus());

  // Final release destroys.
  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// Concurrent read operations
//===----------------------------------------------------------------------===//

// Multiple reads at different offsets on the same file complete independently.
TEST_P(FileTest, ConcurrentReadsAtDifferentOffsets) {
  const char kTestData[] = "0123456789abcdefghij";
  std::string path = CreateTempFileWithContents(kTestData, strlen(kTestData));

  iree_async_file_t* file = ImportTempFileForRead(path);
  ASSERT_NE(file, nullptr);

  // Set up two concurrent reads at different offsets.
  uint8_t buffer_a[5] = {0};
  uint8_t buffer_b[5] = {0};

  iree_async_file_read_operation_t read_a;
  memset(&read_a, 0, sizeof(read_a));
  read_a.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_a.file = file;
  read_a.offset = 0;
  read_a.buffer = iree_async_span_from_ptr(buffer_a, 5);

  iree_async_file_read_operation_t read_b;
  memset(&read_b, 0, sizeof(read_b));
  read_b.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_b.file = file;
  read_b.offset = 10;
  read_b.buffer = iree_async_span_from_ptr(buffer_b, 5);

  CompletionTracker tracker_a;
  read_a.base.completion_fn = CompletionTracker::Callback;
  read_a.base.user_data = &tracker_a;

  CompletionTracker tracker_b;
  read_b.base.completion_fn = CompletionTracker::Callback;
  read_b.base.user_data = &tracker_b;

  // Submit as a batch.
  iree_async_operation_t* ops[] = {&read_a.base, &read_b.base};
  iree_async_operation_list_t list = {ops, 2};
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, list));

  PollUntil(/*min_completions=*/2, /*total_budget=*/iree_make_duration_ms(500));

  EXPECT_EQ(tracker_a.call_count, 1);
  IREE_EXPECT_OK(tracker_a.ConsumeStatus());
  EXPECT_EQ(read_a.bytes_read, 5u);
  EXPECT_EQ(memcmp(buffer_a, "01234", 5), 0);

  EXPECT_EQ(tracker_b.call_count, 1);
  IREE_EXPECT_OK(tracker_b.ConsumeStatus());
  EXPECT_EQ(read_b.bytes_read, 5u);
  EXPECT_EQ(memcmp(buffer_b, "abcde", 5), 0);

  iree_async_file_release(file);
}

//===----------------------------------------------------------------------===//
// Larger I/O tests
//===----------------------------------------------------------------------===//

// Write and read back a larger buffer to exercise multi-page I/O paths.
TEST_P(FileTest, LargeWriteAndReadBack) {
  std::string path = CreateEmptyTempFile();

  iree_async_file_t* file = ImportTempFileForReadWrite(path);
  ASSERT_NE(file, nullptr);

  // Create a 64KB buffer with a known pattern.
  static constexpr size_t kBufferSize = 64 * 1024;
  std::vector<uint8_t> write_buffer(kBufferSize);
  for (size_t i = 0; i < kBufferSize; ++i) {
    write_buffer[i] = static_cast<uint8_t>(i & 0xFF);
  }

  iree_async_file_write_operation_t write_op;
  memset(&write_op, 0, sizeof(write_op));
  write_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write_op.file = file;
  write_op.offset = 0;
  write_op.buffer = iree_async_span_from_ptr(write_buffer.data(), kBufferSize);
  write_op.bytes_written = 0;

  CompletionTracker write_tracker;
  write_op.base.completion_fn = CompletionTracker::Callback;
  write_op.base.user_data = &write_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write_op.base));
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(write_tracker.call_count, 1);
  IREE_EXPECT_OK(write_tracker.ConsumeStatus());
  EXPECT_EQ(write_op.bytes_written, kBufferSize);

  // Read it back.
  std::vector<uint8_t> read_buffer(kBufferSize, 0);
  iree_async_file_read_operation_t read_op;
  memset(&read_op, 0, sizeof(read_op));
  read_op.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
  read_op.file = file;
  read_op.offset = 0;
  read_op.buffer = iree_async_span_from_ptr(read_buffer.data(), kBufferSize);
  read_op.bytes_read = 0;

  CompletionTracker read_tracker;
  read_op.base.completion_fn = CompletionTracker::Callback;
  read_op.base.user_data = &read_tracker;

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &read_op.base));
  PollUntil(/*min_completions=*/1,
            /*total_budget=*/iree_make_duration_ms(2000));

  EXPECT_EQ(read_tracker.call_count, 1);
  IREE_EXPECT_OK(read_tracker.ConsumeStatus());
  EXPECT_EQ(read_op.bytes_read, kBufferSize);
  EXPECT_EQ(write_buffer, read_buffer);

  iree_async_file_release(file);
}

CTS_REGISTER_TEST_SUITE(FileTest);

}  // namespace iree::async::cts
