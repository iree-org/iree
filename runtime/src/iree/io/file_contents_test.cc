// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/file_contents.h"

#include "iree/base/api.h"

#if IREE_FILE_IO_ENABLE

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS)
#include <signal.h>
#include <sys/resource.h>
#endif

namespace iree {
namespace io {
namespace {

using ::iree::testing::status::StatusIs;

std::string GetUniqueContents(const char* unique_name,
                              iree_host_size_t padded_size) {
  std::string str = std::string("Test with name ") + unique_name + "\n";
  if (str.size() < padded_size) str.resize(padded_size, 0);
  return str;
}

static iree_status_t RejectLargeAllocations(void* self,
                                            iree_allocator_command_t command,
                                            const void* params,
                                            void** inout_ptr) {
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC ||
      command == IREE_ALLOCATOR_COMMAND_REALLOC) {
    const auto* allocation =
        static_cast<const iree_allocator_alloc_params_t*>(params);
    if (allocation->byte_length >= *static_cast<iree_host_size_t*>(self)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "test allocation limit");
    }
  }
  auto system_allocator = iree_allocator_system();
  return system_allocator.ctl(system_allocator.self, command, params,
                              inout_ptr);
}

TEST(FileContents, ReadAllocationFailure) {
  iree::testing::TempFilePath path("iree_file_contents_test");
  iree_host_size_t allocation_limit_in_bytes = 65536;
  auto payload =
      GetUniqueContents("ReadAllocationFailure", allocation_limit_in_bytes);
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(payload.data(), payload.size()),
      iree_allocator_system()));
  iree_allocator_t allocator = {&allocation_limit_in_bytes,
                                RejectLargeAllocations};
  iree_io_file_contents_t* contents = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_io_file_contents_read(path.path_view(), allocator, &contents));
  EXPECT_EQ(contents, nullptr);
  iree_io_file_contents_free(contents);
}

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS)
TEST(FileContents, BufferedWriteFailure) {
  iree::testing::TempFilePath path("iree_file_contents_test");
  struct rlimit old_limit;
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &old_limit), 0);
  auto old_handler = signal(SIGXFSZ, SIG_IGN);
  ASSERT_NE(old_handler, SIG_ERR);
  struct RestoreFileLimit {
    struct rlimit limit;
    void (*handler)(int);
    ~RestoreFileLimit() {
      EXPECT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
      EXPECT_NE(signal(SIGXFSZ, handler), SIG_ERR);
    }
  } restore{old_limit, old_handler};
  struct rlimit limit = old_limit;
  limit.rlim_cur = 0;
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);

  // A short fwrite can succeed into a stdio buffer even when the subsequent
  // flush cannot write to the file. The complete write must report failure.
  uint8_t byte = 1;
  iree_status_t status = iree_io_file_contents_write(
      path.path_view(), iree_make_const_byte_span(&byte, 1),
      iree_allocator_system());
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);
}
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_MACOS

TEST(FileContents, ReadWriteContentsPreload) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 32);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_read(
      path.path_view(), iree_allocator_system(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

TEST(FileContents, ReadWriteContentsMmap) {
  constexpr const char* kUniqueName = "ReadWriteContents";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Generate file contents.
  auto write_contents = GetUniqueContents(kUniqueName, 4096);

  // Write the contents to disk.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(write_contents.data(), write_contents.size()),
      iree_allocator_system()));

  // Read the contents from disk.
  iree_io_file_contents_t* read_contents = NULL;
  IREE_ASSERT_OK(
      iree_io_file_contents_map(path.path_view(), IREE_IO_FILE_ACCESS_READ,
                                iree_allocator_system(), &read_contents));

  // Expect the contents are equal.
  EXPECT_EQ(write_contents.size(), read_contents->const_buffer.data_length);
  EXPECT_EQ(memcmp(write_contents.data(), read_contents->const_buffer.data,
                   read_contents->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read_contents);
}

// Tests that a file opened for reading can be opened again concurrently.
// Validates FILE_SHARE_READ behavior on Windows. Without it, the second open
// fails with ERROR_SHARING_VIOLATION.
TEST(FileContents, ConcurrentReadOpens) {
  constexpr const char* kUniqueName = "ConcurrentReadOpens";
  iree::testing::TempFilePath path("iree_file_contents_test");

  // File must not exist.
  ASSERT_FALSE(path.Exists());

  // Write a file to open.
  auto contents = GetUniqueContents(kUniqueName, 4096);
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system()));

  // Open the file twice for reading. Both should succeed.
  iree_io_file_contents_t* read1 = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(path.path_view(),
                                           IREE_IO_FILE_ACCESS_READ,
                                           iree_allocator_system(), &read1));

  iree_io_file_contents_t* read2 = NULL;
  IREE_ASSERT_OK(iree_io_file_contents_map(path.path_view(),
                                           IREE_IO_FILE_ACCESS_READ,
                                           iree_allocator_system(), &read2));

  // Both should have the same contents.
  EXPECT_EQ(read1->const_buffer.data_length, read2->const_buffer.data_length);
  EXPECT_EQ(memcmp(read1->const_buffer.data, read2->const_buffer.data,
                   read1->const_buffer.data_length),
            0);

  iree_io_file_contents_free(read2);
  iree_io_file_contents_free(read1);
}

}  // namespace
}  // namespace io
}  // namespace iree

#endif  // IREE_FILE_IO_ENABLE
