// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/digest.h"

#include <string.h>

#define IREE_HAL_REPLAY_FNV1A64_OFFSET_BASIS 0xcbf29ce484222325ull
#define IREE_HAL_REPLAY_FNV1A64_PRIME 0x100000001b3ull

uint64_t iree_hal_replay_digest_fnv1a64_initialize(void) {
  return IREE_HAL_REPLAY_FNV1A64_OFFSET_BASIS;
}

uint64_t iree_hal_replay_digest_fnv1a64_update(uint64_t state,
                                               iree_const_byte_span_t bytes) {
  for (iree_host_size_t i = 0; i < bytes.data_length; ++i) {
    state ^= bytes.data[i];
    state *= IREE_HAL_REPLAY_FNV1A64_PRIME;
  }
  return state;
}

uint64_t iree_hal_replay_digest_fnv1a64_iovecs(
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  uint64_t state = iree_hal_replay_digest_fnv1a64_initialize();
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    state = iree_hal_replay_digest_fnv1a64_update(state, iovecs[i]);
  }
  return state;
}

void iree_hal_replay_digest_store_fnv1a64(uint64_t digest,
                                          uint8_t out_digest[32]) {
  iree_hal_replay_digest_clear(out_digest);
  memcpy(out_digest, &digest, sizeof(digest));
}

uint64_t iree_hal_replay_digest_load_fnv1a64(const uint8_t digest[32]) {
  uint64_t value = 0;
  memcpy(&value, digest, sizeof(value));
  return value;
}

void iree_hal_replay_digest_clear(uint8_t out_digest[32]) {
  memset(out_digest, 0, 32);
}

bool iree_hal_replay_digest_is_zero(const uint8_t digest[32]) {
  uint8_t any_digest_byte = 0;
  for (iree_host_size_t i = 0; i < 32; ++i) {
    any_digest_byte |= digest[i];
  }
  return any_digest_byte == 0;
}
