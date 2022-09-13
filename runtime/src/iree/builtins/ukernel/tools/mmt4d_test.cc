// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// THIS IS STILL JUST A PLACEHOLDER - NOT AN ACTUAL TEST YET.

#include <stdint.h>

// Include in expected order with stdint and other system headers first.
// See the note in mmt4d.h about stdint.h. This won't be an issue in most uses
// but clang-format really likes to put the mmt4d.h above the system headers
// due to this _test.cc file naming.

#include "iree/base/api.h"
#include "iree/builtins/ukernel/mmt4d.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

TEST(MMT4DTest, iree_mmt4d_example_matmul_f32) {
  iree_ukernel_mmt4d_params_t params;
  memset(&params, 0, sizeof params);
  params.type = iree_ukernel_mmt4d_type_f32f32f32;
  EXPECT_EQ(0, iree_ukernel_mmt4d(&params));
}
