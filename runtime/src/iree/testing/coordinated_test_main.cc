// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Gtest main() replacement for coordinated multi-process tests.
//
// Links in place of gtest_main. Handles child dispatch before gtest
// initialization so that child processes never run RUN_ALL_TESTS().
// The launcher process saves argc/argv for use in TEST bodies via
// iree_coordinated_test_argc/argv().
//
// Test files register their config at static init:
//   IREE_COORDINATED_TEST_REGISTER(kConfig);
//
// TEST bodies call iree_coordinated_test_run() with the saved args:
//   TEST(MyTest, Basic) {
//     ASSERT_EQ(0, iree_coordinated_test_run(
//         iree_coordinated_test_argc(),
//         iree_coordinated_test_argv(), &kConfig));
//   }

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"

// Defined in coordinated_test.c — saves argc/argv for the accessors.
extern "C" void iree_coordinated_test_set_args(int argc, char** argv);

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  // Child dispatch: if --iree_test_role is present, run the role and exit.
  // This must happen before InitGoogleTest — children don't run the test
  // suite. Uses the globally registered config
  // (IREE_COORDINATED_TEST_REGISTER).
  int child_result = iree_coordinated_test_dispatch_if_child(argc, argv, NULL);
  if (child_result >= 0) {
    IREE_TRACE_APP_EXIT(child_result);
    return child_result;
  }

  // Launcher path: save argc/argv for TEST bodies, then run gtest.
  iree_coordinated_test_set_args(argc, argv);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();

  IREE_TRACE_APP_EXIT(ret);
  return ret;
}
