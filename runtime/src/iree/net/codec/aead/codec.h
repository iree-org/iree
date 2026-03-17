// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CODEC_AEAD_CODEC_H_
#define IREE_NET_CODEC_AEAD_CODEC_H_

#include "iree/base/api.h"
#include "iree/net/codec.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AEAD codec
//===----------------------------------------------------------------------===//
//
// Authenticated encryption using ChaCha20-Poly1305 with pre-shared keys (PSK).
//
// == PSK Generation ==
//
// The PSK must be 32 bytes of cryptographically random data:
//
//   uint8_t psk[IREE_NET_AEAD_KEY_SIZE];
//   iree_csprng_fill(iree_make_byte_span(psk, sizeof(psk)));
//
// See iree/base/internal/csprng.h for the platform-abstracted CSPRNG.
//
// == PSK Protection ==
//
// For high-security deployments, protect the PSK in memory:
//
//   // Lock to prevent swapping to disk.
//   iree_memory_lock(psk, sizeof(psk));
//
//   // Exclude from core dumps (Linux).
//   iree_memory_protect_sensitive(psk, sizeof(psk));
//
//   // ... use psk to create codec ...
//
//   // Securely erase when done.
//   iree_memory_wipe(psk, sizeof(psk));
//   iree_memory_unlock(psk, sizeof(psk));
//
// See iree/base/internal/memory.h for memory protection functions.
//
// == PSK Distribution ==
//
// The PSK must be shared between client and server through a secure channel
// before establishing the encrypted connection. Options include:
//   - Out-of-band provisioning (secure configuration management)
//   - Key exchange protocol (caller's responsibility to implement)
//   - Hardware security module integration
//
// This codec does NOT perform key exchange.

// Key size for ChaCha20-Poly1305 (256 bits).
#define IREE_NET_AEAD_KEY_SIZE 32

// Role in connection (determines key derivation direction).
typedef enum iree_net_aead_role_e {
  // Client role: sends with client->server key, receives with server->client.
  IREE_NET_AEAD_ROLE_CLIENT = 0,
  // Server role: sends with server->client key, receives with client->server.
  IREE_NET_AEAD_ROLE_SERVER = 1,
} iree_net_aead_role_t;

// Creates an AEAD codec using ChaCha20-Poly1305.
//
// Wire format overhead: 32 bytes per packet (16-byte nonce + 16-byte auth tag).
//
// Security properties:
//   - Direction-specific keys derived from |psk| using BLAKE2b with domain
//     separation strings. This prevents nonce collision across directions.
//   - 64-bit random session prefix provides ~4 billion sessions before birthday
//     collision (sufficient for any realistic deployment with same PSK).
//   - 64-bit counter per session allows unlimited packets per session.
//   - Replay protection via monotonic counter tracking within each session.
//   - AEAD verification before replay check to prevent timing oracles.
//   - Keys are securely wiped on destroy.
//
// |psk| must be exactly IREE_NET_AEAD_KEY_SIZE (32) bytes. The key is copied
// internally; the caller may free their copy after this call returns.
//
// |role| determines which derived key is used for encode (send) vs decode
// (recv). Both endpoints must use the same PSK but opposite roles.
//
// Thread safety: A single codec instance supports concurrent encode and decode
// from different threads (full-duplex). Multiple concurrent encodes require
// external synchronization to ensure nonce ordering. Multiple concurrent
// decodes are safe but duplicate packets may both succeed if they race - for
// strict exactly-once semantics, use a single decode thread or external
// locking.
IREE_API_EXPORT iree_status_t iree_net_aead_codec_create(
    iree_const_byte_span_t psk, iree_net_aead_role_t role,
    iree_allocator_t host_allocator, iree_net_codec_t** out_codec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CODEC_AEAD_CODEC_H_
