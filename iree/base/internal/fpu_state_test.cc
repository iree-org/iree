// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/fpu_state.h"

#include "iree/testing/gtest.h"

namespace {

// NOTE: depending on compiler options or architecture denormals may always be
// flushed to zero. Here we just test that they are flushed when we request them
// to be.
TEST(FPUStateTest, FlushDenormalsToZero) {
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);

  float f = 1.0f;
  volatile float* fp = &f;
  *fp = *fp * 1e-39f;
  EXPECT_EQ(0.0f, f);

  iree_fpu_state_pop(fpu_state);
}

}  // namespace
