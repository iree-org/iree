// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/memory.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(SecureMemory, WipeZerosMemory) {
  uint8_t buffer[256];
  memset(buffer, 0xFF, sizeof(buffer));

  iree_memory_wipe(buffer, sizeof(buffer));

  // Verify all bytes are zero.
  for (size_t i = 0; i < sizeof(buffer); ++i) {
    EXPECT_EQ(0, buffer[i]) << "Byte at index " << i << " not wiped";
  }
}

TEST(SecureMemory, WipeEmptyBuffer) {
  // Zero-length wipe should not crash.
  iree_memory_wipe(nullptr, 0);
}

TEST(SecureMemory, LockAndUnlock) {
  constexpr size_t kSize = 4096;
  auto buffer = std::make_unique<uint8_t[]>(kSize);

  // Lock may fail if the process lacks privileges or exceeds RLIMIT_MEMLOCK.
  // This is not fatal - we test the API contract but allow failure.
  iree_status_t status = iree_memory_lock(buffer.get(), kSize);
  if (iree_status_is_ok(status)) {
    // If lock succeeded, unlock should work.
    iree_memory_unlock(buffer.get(), kSize);
  } else {
    // Lock failed (e.g., insufficient privileges). This is acceptable.
    iree_status_ignore(status);
  }
}

TEST(SecureMemory, LockEmptyBuffer) {
  // Zero-length lock should succeed as a no-op.
  IREE_EXPECT_OK(iree_memory_lock(nullptr, 0));
  iree_memory_unlock(nullptr, 0);
}

TEST(SecureMemory, ProtectSensitive) {
  constexpr size_t kSize = 4096;
  auto buffer = std::make_unique<uint8_t[]>(kSize);

  // This is best-effort and may be a no-op on some platforms.
  // Just verify it doesn't crash.
  iree_memory_protect_sensitive(buffer.get(), kSize);
}

TEST(SecureMemory, ProtectSensitiveEmptyBuffer) {
  // Zero-length protect should not crash.
  iree_memory_protect_sensitive(nullptr, 0);
}

}  // namespace
