// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_executable.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(LocalExecutableTest, DecodeDispatchReturnStatus) {
  iree_hal_local_executable_t executable = {};
  // Only v0.7 libraries return statuses.
  executable.library_version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_7;
  iree_hal_executable_dispatch_attrs_v0_t attrs = {};
  executable.dispatch_attrs = &attrs;
  executable.export_count = 1;

  IREE_EXPECT_OK(iree_hal_local_executable_status_from_ret(&executable, 0, 0));
  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
  IREE_EXPECT_OK(iree_hal_local_executable_status_from_ret(&executable, 0, 0));

  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_FAILED_PRECONDITION));
  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_FAILED_PRECONDITION));
  attrs.flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_RETURNS_STATUS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_INCOMPATIBLE));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_DEFERRED));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_local_executable_status_from_ret(&executable, 0, -1));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      iree_hal_local_executable_status_from_ret(&executable, 0, 32));

  executable.dispatch_attrs = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 0, IREE_STATUS_FAILED_PRECONDITION));
  executable.dispatch_attrs = &attrs;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_local_executable_status_from_ret(
                            &executable, 1, IREE_STATUS_FAILED_PRECONDITION));
}

}  // namespace
