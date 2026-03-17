// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/codec/aead/codec.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/csprng.h"
#include "iree/base/internal/memory.h"
#include "monocypher.h"

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Wire nonce is 16 bytes: [64-bit random prefix][64-bit counter].
// The 64-bit prefix provides session uniqueness with a birthday bound of ~4
// billion sessions before 50% collision probability. The 64-bit counter
// provides per-session nonce uniqueness for effectively unlimited packets.
// Combined with the 16-byte Poly1305 tag, total overhead is 32 bytes/packet.
#define IREE_NET_AEAD_NONCE_SIZE 16
#define IREE_NET_AEAD_TAG_SIZE 16

static const char* IREE_NET_AEAD_DOMAIN_CLIENT_TO_SERVER =
    "iree-net:client->server";
static const char* IREE_NET_AEAD_DOMAIN_SERVER_TO_CLIENT =
    "iree-net:server->client";

//===----------------------------------------------------------------------===//
// AEAD codec structure
//===----------------------------------------------------------------------===//

typedef struct iree_net_aead_codec_t {
  iree_net_codec_t base;

  // Direction-specific keys derived from PSK.
  uint8_t send_key[IREE_NET_AEAD_KEY_SIZE];
  uint8_t recv_key[IREE_NET_AEAD_KEY_SIZE];

  // Send state: random prefix (set at creation) + atomic counter.
  uint64_t send_prefix;
  iree_atomic_uint64_t send_counter;

  // Receive state: peer's prefix (learned from first message) + counter min.
  uint64_t recv_prefix;
  iree_atomic_int32_t recv_prefix_initialized;  // 0 = pending, 1 = set
  iree_atomic_uint64_t recv_counter_min;
} iree_net_aead_codec_t;

//===----------------------------------------------------------------------===//
// Key derivation
//===----------------------------------------------------------------------===//

static void iree_net_aead_derive_key(const uint8_t* psk, const char* domain,
                                     uint8_t* out_key) {
  crypto_blake2b_keyed(out_key, IREE_NET_AEAD_KEY_SIZE, psk,
                       IREE_NET_AEAD_KEY_SIZE, (const uint8_t*)domain,
                       strlen(domain));
}

//===----------------------------------------------------------------------===//
// iree_net_aead_codec_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_aead_codec_create(
    iree_const_byte_span_t psk, iree_net_aead_role_t role,
    iree_allocator_t host_allocator, iree_net_codec_t** out_codec) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_codec);
  *out_codec = NULL;

  if (psk.data_length != IREE_NET_AEAD_KEY_SIZE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PSK must be exactly %d bytes, got %" PRIhsz,
                            IREE_NET_AEAD_KEY_SIZE, psk.data_length);
  }

  // Generate random 64-bit prefix for session uniqueness.
  // With 64 bits of randomness, the birthday bound is ~4 billion sessions
  // before 50% collision probability - sufficient for any realistic deployment.
  uint64_t send_prefix = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_csprng_fill(iree_make_byte_span((uint8_t*)&send_prefix,
                                               sizeof(send_prefix))));

  iree_net_aead_codec_t* codec = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*codec), (void**)&codec));
  memset(codec, 0, sizeof(*codec));

  // Derive direction-specific keys from PSK.
  uint8_t client_to_server_key[IREE_NET_AEAD_KEY_SIZE] = {0};
  uint8_t server_to_client_key[IREE_NET_AEAD_KEY_SIZE] = {0};
  iree_net_aead_derive_key(psk.data, IREE_NET_AEAD_DOMAIN_CLIENT_TO_SERVER,
                           client_to_server_key);
  iree_net_aead_derive_key(psk.data, IREE_NET_AEAD_DOMAIN_SERVER_TO_CLIENT,
                           server_to_client_key);

  if (role == IREE_NET_AEAD_ROLE_CLIENT) {
    memcpy(codec->send_key, client_to_server_key, IREE_NET_AEAD_KEY_SIZE);
    memcpy(codec->recv_key, server_to_client_key, IREE_NET_AEAD_KEY_SIZE);
  } else {
    memcpy(codec->send_key, server_to_client_key, IREE_NET_AEAD_KEY_SIZE);
    memcpy(codec->recv_key, client_to_server_key, IREE_NET_AEAD_KEY_SIZE);
  }

  // Wipe intermediate keys.
  iree_memory_wipe(client_to_server_key, sizeof(client_to_server_key));
  iree_memory_wipe(server_to_client_key, sizeof(server_to_client_key));

  // Initialize base codec with vtable (declared below).
  extern const iree_net_codec_vtable_t iree_net_aead_codec_vtable;
  iree_net_codec_initialize(&iree_net_aead_codec_vtable,
                            IREE_NET_CODEC_FLAG_REQUIRES_CPU_ACCESS,
                            host_allocator, &codec->base);

  // Initialize send state.
  codec->send_prefix = send_prefix;
  iree_atomic_store(&codec->send_counter, 0, iree_memory_order_relaxed);

  // Initialize receive state. Prefix is learned from first received message.
  codec->recv_prefix = 0;
  iree_atomic_store(&codec->recv_prefix_initialized, 0,
                    iree_memory_order_relaxed);
  iree_atomic_store(&codec->recv_counter_min, 0, iree_memory_order_relaxed);

  *out_codec = &codec->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_net_aead_codec_destroy(iree_net_codec_t* base_codec) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(base_codec);
  iree_net_aead_codec_t* codec = (iree_net_aead_codec_t*)base_codec;

  // Securely wipe keys before freeing.
  iree_memory_wipe(codec->send_key, sizeof(codec->send_key));
  iree_memory_wipe(codec->recv_key, sizeof(codec->recv_key));

  iree_allocator_t allocator = codec->base.host_allocator;
  iree_allocator_free(allocator, codec);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Vtable implementation
//===----------------------------------------------------------------------===//

static iree_net_codec_overhead_t iree_net_aead_codec_query_overhead(
    iree_net_codec_t* base_codec) {
  (void)base_codec;
  iree_net_codec_overhead_t overhead = {0};
  overhead.prefix = IREE_NET_AEAD_NONCE_SIZE;
  overhead.suffix = IREE_NET_AEAD_TAG_SIZE;
  return overhead;
}

static iree_status_t iree_net_aead_codec_encode(
    iree_net_codec_t* base_codec, uint8_t* payload_ptr,
    iree_host_size_t payload_length) {
  iree_net_aead_codec_t* codec = (iree_net_aead_codec_t*)base_codec;

  // Generate counter from atomic and check for overflow.
  uint64_t counter =
      iree_atomic_fetch_add(&codec->send_counter, 1, iree_memory_order_relaxed);
  if (IREE_UNLIKELY(counter == UINT64_MAX)) {
    // Counter exhausted. This should never happen in practice (would require
    // sending 2^64 packets), but we fail safely rather than wrap.
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AEAD nonce counter exhausted");
  }

  // Pointers into pre-allocated buffer regions.
  uint8_t* nonce_ptr = payload_ptr - IREE_NET_AEAD_NONCE_SIZE;
  uint8_t* tag_ptr = payload_ptr + payload_length;

  // Write wire nonce: [64-bit prefix][64-bit counter] (little-endian).
  memcpy(nonce_ptr, &codec->send_prefix, sizeof(codec->send_prefix));
  memcpy(nonce_ptr + sizeof(codec->send_prefix), &counter, sizeof(counter));

  // Monocypher uses 24-byte nonce; pad our 16-byte nonce with zeros.
  uint8_t full_nonce[24] = {0};
  memcpy(full_nonce, nonce_ptr, IREE_NET_AEAD_NONCE_SIZE);

  // Encrypt in-place and write tag.
  crypto_aead_lock(payload_ptr,      // ciphertext output (in-place)
                   tag_ptr,          // tag output
                   codec->send_key,  // direction-specific key
                   full_nonce,       // 24-byte nonce
                   NULL, 0,          // no additional data
                   payload_ptr,      // plaintext input (same buffer)
                   payload_length);

  return iree_ok_status();
}

static iree_status_t iree_net_aead_codec_decode(iree_net_codec_t* base_codec,
                                                uint8_t* frame_ptr,
                                                iree_host_size_t frame_length,
                                                iree_byte_span_t* out_payload) {
  iree_net_aead_codec_t* codec = (iree_net_aead_codec_t*)base_codec;

  const iree_host_size_t overhead =
      IREE_NET_AEAD_NONCE_SIZE + IREE_NET_AEAD_TAG_SIZE;

  if (frame_length < overhead) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AEAD frame too small: %" PRIhsz " < %" PRIhsz,
                            frame_length, overhead);
  }

  iree_host_size_t ciphertext_length = frame_length - overhead;

  // Parse frame layout: [prefix:8][counter:8][ciphertext:N][tag:16].
  uint8_t* nonce_ptr = frame_ptr;
  uint8_t* ciphertext_ptr = frame_ptr + IREE_NET_AEAD_NONCE_SIZE;
  uint8_t* tag_ptr = ciphertext_ptr + ciphertext_length;

  // Extract prefix and counter from wire nonce.
  uint64_t prefix = 0;
  uint64_t counter = 0;
  memcpy(&prefix, nonce_ptr, sizeof(prefix));
  memcpy(&counter, nonce_ptr + sizeof(prefix), sizeof(counter));

  // Pad nonce for Monocypher (uses 24-byte nonce internally).
  uint8_t full_nonce[24] = {0};
  memcpy(full_nonce, nonce_ptr, IREE_NET_AEAD_NONCE_SIZE);

  // SECURITY: Verify AEAD before replay check to prevent timing oracle.
  int result = crypto_aead_unlock(ciphertext_ptr,   // plaintext output
                                  tag_ptr,          // tag to verify
                                  codec->recv_key,  // direction-specific key
                                  full_nonce,       // 24-byte nonce
                                  NULL, 0,          // no additional data
                                  ciphertext_ptr,   // ciphertext input
                                  ciphertext_length);

  if (result != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "AEAD authentication failed");
  }

  // AEAD verified - now check prefix consistency and replay protection.
  //
  // On first message, learn the peer's prefix. On subsequent messages, verify
  // it matches. This ensures replay protection tracks counters within a single
  // session.
  if (!iree_atomic_load(&codec->recv_prefix_initialized,
                        iree_memory_order_acquire)) {
    // First message: learn peer's prefix.
    codec->recv_prefix = prefix;
    iree_atomic_store(&codec->recv_prefix_initialized, 1,
                      iree_memory_order_release);
  } else if (prefix != codec->recv_prefix) {
    // Prefix mismatch - either a bug or an attack. Re-encrypt to avoid leaking
    // plaintext.
    crypto_aead_lock(ciphertext_ptr, tag_ptr, codec->recv_key, full_nonce, NULL,
                     0, ciphertext_ptr, ciphertext_length);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "AEAD prefix mismatch: expected %016" PRIx64
                            ", got %016" PRIx64,
                            codec->recv_prefix, prefix);
  }

  // Check replay protection on counter.
  uint64_t min_counter =
      iree_atomic_load(&codec->recv_counter_min, iree_memory_order_acquire);

  if (counter < min_counter) {
    // Re-encrypt before returning to avoid leaking plaintext.
    crypto_aead_lock(ciphertext_ptr, tag_ptr, codec->recv_key, full_nonce, NULL,
                     0, ciphertext_ptr, ciphertext_length);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay detected: counter %" PRIu64
                            " < min %" PRIu64,
                            counter, min_counter);
  }

  // Advance minimum counter, handling potential overflow.
  uint64_t new_min = counter + 1;
  if (IREE_UNLIKELY(new_min == 0)) {
    // Counter overflow - session must be rekeyed.
    iree_memory_wipe(codec->recv_key, sizeof(codec->recv_key));
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AEAD counter space exhausted");
  }

  uint64_t current_min = 0;
  do {
    current_min =
        iree_atomic_load(&codec->recv_counter_min, iree_memory_order_relaxed);
    if (new_min <= current_min) break;
  } while (!iree_atomic_compare_exchange_strong(
      &codec->recv_counter_min, &current_min, new_min,
      iree_memory_order_release, iree_memory_order_relaxed));

  *out_payload = iree_make_byte_span(ciphertext_ptr, ciphertext_length);
  return iree_ok_status();
}

const iree_net_codec_vtable_t iree_net_aead_codec_vtable = {
    .destroy = iree_net_aead_codec_destroy,
    .query_overhead = iree_net_aead_codec_query_overhead,
    .encode = iree_net_aead_codec_encode,
    .decode = iree_net_aead_codec_decode,
};
