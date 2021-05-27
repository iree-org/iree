// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
