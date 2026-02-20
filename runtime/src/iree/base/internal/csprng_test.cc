// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/csprng.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(CSPRNG, FillSmallBuffer) {
  uint8_t buffer[32];
  memset(buffer, 0, sizeof(buffer));

  IREE_ASSERT_OK(iree_csprng_fill(iree_make_byte_span(buffer, sizeof(buffer))));

  // Check that at least some bytes are non-zero (extremely high probability).
  // A correctly functioning CSPRNG should produce non-zero bytes.
  int non_zero_count = 0;
  for (size_t i = 0; i < sizeof(buffer); ++i) {
    if (buffer[i] != 0) ++non_zero_count;
  }
  EXPECT_GT(non_zero_count, 0)
      << "CSPRNG returned all zeros, which is extremely unlikely";
}

TEST(CSPRNG, FillLargeBuffer) {
  // Test with buffer > 65536 bytes to exercise chunking on Emscripten.
  constexpr size_t kSize = 128 * 1024;
  auto buffer = std::make_unique<uint8_t[]>(kSize);
  memset(buffer.get(), 0, kSize);

  IREE_ASSERT_OK(iree_csprng_fill(iree_make_byte_span(buffer.get(), kSize)));

  // Check that at least some bytes are non-zero.
  int non_zero_count = 0;
  for (size_t i = 0; i < kSize; ++i) {
    if (buffer[i] != 0) ++non_zero_count;
  }
  EXPECT_GT(non_zero_count, 0)
      << "CSPRNG returned all zeros, which is extremely unlikely";
}

TEST(CSPRNG, FillZeroLengthBuffer) {
  // Zero-length buffer should succeed without error.
  IREE_EXPECT_OK(iree_csprng_fill(iree_make_byte_span(nullptr, 0)));
}

TEST(CSPRNG, MultipleCallsProduceDifferentData) {
  uint8_t buffer1[64];
  uint8_t buffer2[64];

  IREE_ASSERT_OK(
      iree_csprng_fill(iree_make_byte_span(buffer1, sizeof(buffer1))));
  IREE_ASSERT_OK(
      iree_csprng_fill(iree_make_byte_span(buffer2, sizeof(buffer2))));

  // Two consecutive calls should produce different data with high probability.
  EXPECT_NE(0, memcmp(buffer1, buffer2, sizeof(buffer1)))
      << "Two consecutive CSPRNG fills produced identical data";
}

}  // namespace
