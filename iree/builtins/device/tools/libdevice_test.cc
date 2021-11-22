// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "iree/base/api.h"
#include "iree/builtins/device/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

TEST(LibDeviceTest, iree_h2f_ieee) {
  // Just ensuring that the code links.
  EXPECT_EQ(0.25f, iree_h2f_ieee(0x3400));
}

TEST(LibDeviceTest, iree_f2h_ieee) {
  // Just ensuring that the code links.
  EXPECT_EQ(0x3400, iree_f2h_ieee(0.25f));
}
