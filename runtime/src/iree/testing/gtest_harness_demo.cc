// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demo of iree/testing/gtest_harness.h usage.
// Can be used with iree-bazel-try for one-shot testing.

#include "iree/testing/gtest_harness.h"

TEST(GTestHarnessDemo, StatusOk) { IREE_EXPECT_OK(iree_ok_status()); }

TEST(GTestHarnessDemo, StatusMatchers) {
  EXPECT_THAT(iree_ok_status(), IsOk());
  EXPECT_THAT(iree_make_status(IREE_STATUS_INVALID_ARGUMENT),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST(GTestHarnessDemo, AllocatorSystem) {
  iree_allocator_t allocator = iree_allocator_system();
  void* ptr = NULL;
  iree_status_t status = iree_allocator_malloc(allocator, 1024, &ptr);
  IREE_EXPECT_OK(status);
  EXPECT_NE(ptr, nullptr);
  iree_allocator_free(allocator, ptr);
}
