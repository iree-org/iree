// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for iree_async_primitive_t lifecycle operations (dup, close).

#include "iree/async/primitive.h"

#if !defined(IREE_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <unistd.h>
#else
// clang-format off
#include <windows.h>
// clang-format on
#endif  // !IREE_PLATFORM_WINDOWS

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Creates a platform-native primitive for testing. On POSIX, creates a pipe and
// returns the read end (caller must close the write end via close_write_end).
// On Windows, creates an auto-reset Event.
class PrimitiveTestFixture : public ::testing::Test {
 protected:
#if !defined(IREE_PLATFORM_WINDOWS)
  void SetUp() override {
    int fds[2];
    ASSERT_EQ(pipe(fds), 0) << "pipe: " << strerror(errno);
    read_fd_ = fds[0];
    write_fd_ = fds[1];
  }

  void TearDown() override {
    if (write_fd_ >= 0) close(write_fd_);
    if (read_fd_ >= 0) close(read_fd_);
  }

  iree_async_primitive_t MakePrimitive() {
    return iree_async_primitive_from_fd(read_fd_);
  }

  // Verifies that a primitive refers to a valid, readable fd by writing a byte
  // through the pipe and reading it back from the given fd.
  void AssertPrimitiveValid(iree_async_primitive_t primitive) {
    ASSERT_EQ(primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
    char byte = 'x';
    ASSERT_EQ(write(write_fd_, &byte, 1), 1);
    char result = 0;
    ASSERT_EQ(read(primitive.value.fd, &result, 1), 1);
    ASSERT_EQ(result, 'x');
  }

  int read_fd_ = -1;
  int write_fd_ = -1;
#else
  void SetUp() override {
    event_ = CreateEventW(NULL, FALSE, FALSE, NULL);
    ASSERT_NE(event_, (HANDLE)NULL) << "CreateEvent: " << GetLastError();
  }

  void TearDown() override {
    if (event_) CloseHandle(event_);
  }

  iree_async_primitive_t MakePrimitive() {
    return iree_async_primitive_from_win32_handle((uintptr_t)event_);
  }

  // Verifies that a primitive refers to a valid Event handle by signaling
  // and waiting on it.
  void AssertPrimitiveValid(iree_async_primitive_t primitive) {
    ASSERT_EQ(primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE);
    HANDLE handle = (HANDLE)primitive.value.win32_handle;
    ASSERT_TRUE(SetEvent(handle)) << "SetEvent: " << GetLastError();
    DWORD wait_result = WaitForSingleObject(handle, 0);
    ASSERT_EQ(wait_result, WAIT_OBJECT_0);
  }

  HANDLE event_ = NULL;
#endif  // !IREE_PLATFORM_WINDOWS
};

//===----------------------------------------------------------------------===//
// iree_async_primitive_dup
//===----------------------------------------------------------------------===//

TEST_F(PrimitiveTestFixture, DupNoneFails) {
  iree_async_primitive_t dup = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_primitive_dup(iree_async_primitive_none(),
                                                &dup));
  EXPECT_TRUE(iree_async_primitive_is_none(dup));
}

TEST_F(PrimitiveTestFixture, DupProducesValidHandle) {
  iree_async_primitive_t original = MakePrimitive();
  iree_async_primitive_t dup = {};
  IREE_ASSERT_OK(iree_async_primitive_dup(original, &dup));

  // Dup should have the same type but a different underlying handle value.
  EXPECT_EQ(dup.type, original.type);
#if !defined(IREE_PLATFORM_WINDOWS)
  EXPECT_NE(dup.value.fd, original.value.fd);
#else
  EXPECT_NE(dup.value.win32_handle, original.value.win32_handle);
#endif

  // The dup should be independently usable.
  AssertPrimitiveValid(dup);

  iree_async_primitive_close(&dup);
}

TEST_F(PrimitiveTestFixture, DupIsIndependentOfOriginal) {
  iree_async_primitive_t original = MakePrimitive();
  iree_async_primitive_t dup = {};
  IREE_ASSERT_OK(iree_async_primitive_dup(original, &dup));

  // Close the dup — the original should still be valid.
  iree_async_primitive_close(&dup);
  EXPECT_TRUE(iree_async_primitive_is_none(dup));
  AssertPrimitiveValid(original);
}

TEST_F(PrimitiveTestFixture, MultipleDupsAreIndependent) {
  iree_async_primitive_t original = MakePrimitive();
  iree_async_primitive_t dup1 = {};
  iree_async_primitive_t dup2 = {};
  IREE_ASSERT_OK(iree_async_primitive_dup(original, &dup1));
  IREE_ASSERT_OK(iree_async_primitive_dup(original, &dup2));

  // All three should be distinct handles.
#if !defined(IREE_PLATFORM_WINDOWS)
  EXPECT_NE(dup1.value.fd, dup2.value.fd);
  EXPECT_NE(dup1.value.fd, original.value.fd);
#else
  EXPECT_NE(dup1.value.win32_handle, dup2.value.win32_handle);
  EXPECT_NE(dup1.value.win32_handle, original.value.win32_handle);
#endif

  iree_async_primitive_close(&dup1);
  iree_async_primitive_close(&dup2);
  AssertPrimitiveValid(original);
}

//===----------------------------------------------------------------------===//
// iree_async_primitive_close
//===----------------------------------------------------------------------===//

TEST_F(PrimitiveTestFixture, CloseNoneIsNoop) {
  iree_async_primitive_t none = iree_async_primitive_none();
  iree_async_primitive_close(&none);  // Should not crash.
  EXPECT_TRUE(iree_async_primitive_is_none(none));
}

TEST_F(PrimitiveTestFixture, CloseNullIsNoop) {
  iree_async_primitive_close(NULL);  // Should not crash.
}

TEST_F(PrimitiveTestFixture, CloseSetsToNone) {
  // Dup so we have a handle we can close without affecting the fixture's
  // cleanup.
  iree_async_primitive_t original = MakePrimitive();
  iree_async_primitive_t dup = {};
  IREE_ASSERT_OK(iree_async_primitive_dup(original, &dup));
  EXPECT_FALSE(iree_async_primitive_is_none(dup));

  iree_async_primitive_close(&dup);
  EXPECT_TRUE(iree_async_primitive_is_none(dup));
}

//===----------------------------------------------------------------------===//
// iree_async_primitive_t construction helpers
//===----------------------------------------------------------------------===//

TEST(PrimitiveTest, NoneIsDefault) {
  iree_async_primitive_t none = iree_async_primitive_none();
  EXPECT_EQ(none.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_TRUE(iree_async_primitive_is_none(none));
}

TEST(PrimitiveTest, MakePreservesTypeAndValue) {
  iree_async_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  value.reserved = 0x42;
  iree_async_primitive_t p =
      iree_async_primitive_make(IREE_ASYNC_PRIMITIVE_TYPE_NONE, value);
  EXPECT_EQ(p.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(p.value.reserved, 0x42);
}

TEST(PrimitiveTest, IsNoneOnlyMatchesNoneType) {
  // Non-NONE type with zero value should not be considered NONE.
  iree_async_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  iree_async_primitive_t p =
      iree_async_primitive_make(IREE_ASYNC_PRIMITIVE_TYPE_FD, value);
  EXPECT_FALSE(iree_async_primitive_is_none(p));
}

#if defined(IREE_ASYNC_HAVE_FD)
TEST(PrimitiveTest, FromFd) {
  iree_async_primitive_t p = iree_async_primitive_from_fd(42);
  EXPECT_EQ(p.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  EXPECT_EQ(p.value.fd, 42);
  EXPECT_FALSE(iree_async_primitive_is_none(p));
}
#endif  // IREE_ASYNC_HAVE_FD

#if defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
TEST(PrimitiveTest, FromWin32Handle) {
  iree_async_primitive_t p =
      iree_async_primitive_from_win32_handle(0xDEADBEEF);
  EXPECT_EQ(p.type, IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE);
  EXPECT_EQ(p.value.win32_handle, (uintptr_t)0xDEADBEEF);
  EXPECT_FALSE(iree_async_primitive_is_none(p));
}
#endif  // IREE_ASYNC_HAVE_WIN32_HANDLE

#if defined(IREE_ASYNC_HAVE_MACH_PORT)
TEST(PrimitiveTest, FromMachPort) {
  iree_async_primitive_t p = iree_async_primitive_from_mach_port(12345);
  EXPECT_EQ(p.type, IREE_ASYNC_PRIMITIVE_TYPE_MACH_PORT);
  EXPECT_EQ(p.value.mach_port, 12345u);
  EXPECT_FALSE(iree_async_primitive_is_none(p));
}
#endif  // IREE_ASYNC_HAVE_MACH_PORT

}  // namespace
