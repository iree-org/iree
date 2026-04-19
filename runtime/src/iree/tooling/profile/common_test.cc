// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/common.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

TEST(ProfileCommonTest, NamesProfileStatusCodes) {
  EXPECT_STREQ("CANCELLED",
               iree_profile_status_code_name(IREE_STATUS_CANCELLED));
  EXPECT_STREQ("UNKNOWN_STATUS", iree_profile_status_code_name(UINT32_MAX));
}

}  // namespace
