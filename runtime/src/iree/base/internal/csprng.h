// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_CSPRNG_H_
#define IREE_BASE_INTERNAL_CSPRNG_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Cryptographically Secure Pseudo-Random Number Generator (CSPRNG)
//===----------------------------------------------------------------------===//
//
// Platform-abstracted access to the operating system's cryptographic random
// number generator. Use for security-sensitive purposes:
//   - Cryptographic key generation
//   - Nonces and IVs
//   - Session tokens
//   - Any randomness where predictability would be a security vulnerability
//
// For non-security purposes (shuffling, load balancing, jitter), prefer
// iree/base/internal/prng.h which is faster but NOT cryptographically secure.
//
// Platform implementations:
//   Windows:       BCryptGenRandom (system preferred RNG)
//   macOS/BSD:     arc4random_buf
//   Linux/Android: getrandom() syscall (requires kernel 3.17+)
//   Emscripten:    crypto.getRandomValues

// Fills |buffer| with cryptographically random data.
//
// This function blocks until all requested bytes are available. On modern
// systems this completes immediately; blocking only occurs during early boot
// before the entropy pool is initialized (extremely rare in practice).
//
// Returns IREE_STATUS_INTERNAL on platform RNG failure (should not happen
// on correctly configured systems).
IREE_API_EXPORT iree_status_t iree_csprng_fill(iree_byte_span_t buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_CSPRNG_H_
