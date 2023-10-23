// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/testing/gtest.h"

extern "C" int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  // Pass through flags to gtest (allowing --help to fall through).
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();

  IREE_TRACE_APP_EXIT(ret);
  return ret;
}
