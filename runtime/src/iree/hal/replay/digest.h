// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_DIGEST_H_
#define IREE_HAL_REPLAY_DIGEST_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the initial FNV-1a 64-bit digest state.
uint64_t iree_hal_replay_digest_fnv1a64_initialize(void);

// Updates an FNV-1a 64-bit digest with |bytes|.
uint64_t iree_hal_replay_digest_fnv1a64_update(uint64_t state,
                                               iree_const_byte_span_t bytes);

// Computes an FNV-1a 64-bit digest over |iovecs| in order.
uint64_t iree_hal_replay_digest_fnv1a64_iovecs(
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs);

// Stores an FNV-1a 64-bit digest in the replay 32-byte digest field.
void iree_hal_replay_digest_store_fnv1a64(uint64_t digest,
                                          uint8_t out_digest[32]);

// Loads an FNV-1a 64-bit digest from the replay 32-byte digest field.
uint64_t iree_hal_replay_digest_load_fnv1a64(const uint8_t digest[32]);

// Clears the replay 32-byte digest field.
void iree_hal_replay_digest_clear(uint8_t out_digest[32]);

// Returns true if the replay 32-byte digest field is all zero.
bool iree_hal_replay_digest_is_zero(const uint8_t digest[32]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_DIGEST_H_
