// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C API tests for ROCM-specific flags. This test is in the ROCM plugin
// directory because these flags are registered by ROCMOptions and only
// available when ROCM is enabled.

#include <gtest/gtest.h>

#include "iree/compiler/embedding_api.h"

namespace {

class ROCMCAPITest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ireeCompilerGlobalInitialize(); }
  static void TearDownTestSuite() { ireeCompilerGlobalShutdown(); }

  void SetUp() override { session = ireeCompilerSessionCreate(); }
  void TearDown() override { ireeCompilerSessionDestroy(session); }

  iree_compiler_session_t *session = nullptr;
};

TEST_F(ROCMCAPITest, TuningSpecPathFlagAccepted) {
  const char *flag = "--iree-codegen-tuning-spec-path=/tmp/spec.mlir";
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, &flag);
  ASSERT_EQ(err, nullptr)
      << "Tuning spec path flag should be accepted via C API";
}

TEST_F(ROCMCAPITest, DefaultTuningSpecsFlagNotRegistered) {
  const char *flag = "--iree-codegen-enable-default-tuning-specs";
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, &flag);
  ASSERT_NE(err, nullptr) << "CLI-only flag should not be accessible via C API";
  ireeCompilerErrorDestroy(err);
}

} // namespace
