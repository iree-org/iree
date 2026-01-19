// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Minimal harness for one-shot gtest experiments via iree-bazel-try.
// Includes gtest, status matchers, and common usings for convenient tests.
//
// Basic test:
//   iree-bazel-try -e '
//   #include "iree/testing/gtest_harness.h"
//   TEST(Quick, StatusOk) {
//     IREE_EXPECT_OK(iree_ok_status());
//   }
//   '
//
// With status matchers:
//   iree-bazel-try -e '
//   #include "iree/testing/gtest_harness.h"
//   TEST(StatusTest, Matchers) {
//     EXPECT_THAT(iree_ok_status(), IsOk());
//     EXPECT_THAT(iree_make_status(IREE_STATUS_INVALID_ARGUMENT),
//                 StatusIs(StatusCode::kInvalidArgument));
//   }
//   '
//
// The harness handles:
// - gtest/gmock inclusion and main() via gtest_main
// - IREE status matchers (IsOk, StatusIs, IsOkAndHolds)
// - Common test assertion macros (IREE_EXPECT_OK, IREE_ASSERT_OK)

#ifndef IREE_TESTING_GTEST_HARNESS_H_
#define IREE_TESTING_GTEST_HARNESS_H_

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Bring common test utilities into scope for convenience.
using iree::Status;
using iree::StatusCode;
using iree::StatusOr;
using iree::testing::status::IsOk;
using iree::testing::status::IsOkAndHolds;
using iree::testing::status::StatusIs;
using ::testing::_;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Ne;
using ::testing::Not;

#endif  // IREE_TESTING_GTEST_HARNESS_H_
