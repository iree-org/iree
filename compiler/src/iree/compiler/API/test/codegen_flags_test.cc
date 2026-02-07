// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C API tests for common codegen flags that should be accessible via the
// embedding API.

#include <gtest/gtest.h>

#include "iree/compiler/embedding_api.h"

namespace {

class CodegenFlagsCAPITest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ireeCompilerGlobalInitialize(); }
  static void TearDownTestSuite() { ireeCompilerGlobalShutdown(); }

  void SetUp() override { session = ireeCompilerSessionCreate(); }
  void TearDown() override { ireeCompilerSessionDestroy(session); }

  iree_compiler_session_t *session = nullptr;
};

TEST_F(CodegenFlagsCAPITest, TuningSpecPathFlagAccepted) {
  const char *flag = "--iree-codegen-tuning-spec-path=/tmp/spec.mlir";
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, &flag);
  ASSERT_EQ(err, nullptr)
      << "Tuning spec path flag should be accepted via C API";
}

TEST_F(CodegenFlagsCAPITest, DefaultTuningSpecsFlagNotRegistered) {
  const char *flag = "--iree-codegen-enable-default-tuning-specs";
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, &flag);
  ASSERT_NE(err, nullptr) << "CLI-only flag should not be accessible via C API";
  ireeCompilerErrorDestroy(err);
}

} // namespace
