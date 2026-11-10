// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C API tests for the plugin activation flags that must be reachable through
// the embedding API. Loading a library is a global step that precedes
// initialization, so it stays out of the session flags.

#include <gtest/gtest.h>

#include <string>

#include "iree/compiler/embedding_api.h"

namespace {

class PluginFlagsCAPITest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ireeCompilerGlobalInitialize(); }
  static void TearDownTestSuite() { ireeCompilerGlobalShutdown(); }

  void SetUp() override { session = ireeCompilerSessionCreate(); }
  void TearDown() override { ireeCompilerSessionDestroy(session); }

  std::string getFlags() {
    std::string flags;
    ireeCompilerSessionGetFlags(
        session, /*nonDefaultOnly=*/false,
        [](const char *flag, size_t length, void *userData) {
          auto *flags = static_cast<std::string *>(userData);
          flags->append(flag, length);
          flags->push_back('\n');
        },
        &flags);
    return flags;
  }

  iree_compiler_session_t *session = nullptr;
};

TEST_F(PluginFlagsCAPITest, ActivationFlagsAccepted) {
  const char *flags[] = {"--iree-plugin=-example", "--iree-print-plugin-info"};
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 2, flags);
  ASSERT_EQ(err, nullptr) << ireeCompilerErrorGetMessage(err);
  std::string actual = getFlags();
  EXPECT_NE(actual.find("--iree-plugin=-example"), std::string::npos) << actual;
  EXPECT_NE(actual.find("--iree-print-plugin-info"), std::string::npos)
      << actual;
}

TEST_F(PluginFlagsCAPITest, LoadFlagRejected) {
  const char *flag = "--iree-load-plugin=/does/not/exist";
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, &flag);
  ASSERT_NE(err, nullptr) << "library loading is not a session option";
  ireeCompilerErrorDestroy(err);
}

} // namespace
