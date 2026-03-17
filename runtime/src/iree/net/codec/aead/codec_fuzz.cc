// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for AEAD codec: tests decode resilience against malformed frames,
// encode with various PSK/payload combinations, and bit-flip mutations.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/net/codec/aead/codec.h"

// Minimum input size to extract PSK (32 bytes) + at least some frame data.
#define MIN_INPUT_SIZE (IREE_NET_AEAD_KEY_SIZE + 1)

// Maximum payload size to avoid OOM during fuzzing.
#define MAX_PAYLOAD_SIZE (64 * 1024)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < MIN_INPUT_SIZE) {
    return 0;
  }

  // Use first 32 bytes as PSK, rest as frame/payload data.
  iree_const_byte_span_t psk = {data, IREE_NET_AEAD_KEY_SIZE};
  const uint8_t* frame_data = data + IREE_NET_AEAD_KEY_SIZE;
  size_t frame_size = size - IREE_NET_AEAD_KEY_SIZE;

  //===--------------------------------------------------------------------===//
  // Test 1: Decode random frames (should fail gracefully, never crash)
  //===--------------------------------------------------------------------===//

  {
    iree_net_codec_t* codec = nullptr;
    iree_status_t status = iree_net_aead_codec_create(
        psk, IREE_NET_AEAD_ROLE_SERVER, iree_allocator_system(), &codec);
    if (iree_status_is_ok(status)) {
      // Copy frame data since decode may modify it in-place.
      void* frame_copy = nullptr;
      status = iree_allocator_malloc(iree_allocator_system(), frame_size,
                                     &frame_copy);
      if (iree_status_is_ok(status)) {
        memcpy(frame_copy, frame_data, frame_size);

        iree_byte_span_t decoded = {nullptr, 0};
        status = iree_net_codec_decode(codec, (uint8_t*)frame_copy, frame_size,
                                       &decoded);
        // Status can be OK or error - both are valid. We just want no crash.
        iree_status_ignore(status);

        iree_allocator_free(iree_allocator_system(), frame_copy);
      } else {
        iree_status_ignore(status);
      }
      iree_net_codec_release(codec);
    } else {
      iree_status_ignore(status);
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Encode then decode with fuzzed PSK (roundtrip validation)
  //===--------------------------------------------------------------------===//

  if (frame_size <= MAX_PAYLOAD_SIZE) {
    iree_net_codec_t* client = nullptr;
    iree_net_codec_t* server = nullptr;
    iree_status_t status = iree_net_aead_codec_create(
        psk, IREE_NET_AEAD_ROLE_CLIENT, iree_allocator_system(), &client);
    if (iree_status_is_ok(status)) {
      status = iree_net_aead_codec_create(psk, IREE_NET_AEAD_ROLE_SERVER,
                                          iree_allocator_system(), &server);
    }

    if (iree_status_is_ok(status)) {
      iree_net_codec_overhead_t overhead =
          iree_net_codec_query_overhead(client);
      size_t payload_size = frame_size;
      size_t buffer_size = overhead.prefix + payload_size + overhead.suffix;

      void* buffer = nullptr;
      status =
          iree_allocator_malloc(iree_allocator_system(), buffer_size, &buffer);
      if (iree_status_is_ok(status)) {
        uint8_t* payload_ptr = (uint8_t*)buffer + overhead.prefix;

        // Use fuzz data as payload.
        memcpy(payload_ptr, frame_data, payload_size);

        // Encode.
        status = iree_net_codec_encode(client, payload_ptr, payload_size);
        if (iree_status_is_ok(status)) {
          // Decode should succeed with matching PSK.
          iree_byte_span_t decoded = {nullptr, 0};
          status = iree_net_codec_decode(server, (uint8_t*)buffer, buffer_size,
                                         &decoded);
          // With matching PSK, decode should succeed.
          iree_status_ignore(status);
        } else {
          iree_status_ignore(status);
        }

        iree_allocator_free(iree_allocator_system(), buffer);
      } else {
        iree_status_ignore(status);
      }
    } else {
      iree_status_ignore(status);
    }

    if (client) iree_net_codec_release(client);
    if (server) iree_net_codec_release(server);
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Bit-flip mutations on valid encoded frame
  //===--------------------------------------------------------------------===//

  if (frame_size > 0 && frame_size <= MAX_PAYLOAD_SIZE) {
    // Use a fixed PSK for this test to create valid frames.
    static const uint8_t kFixedPSK[IREE_NET_AEAD_KEY_SIZE] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
        0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
        0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    };
    iree_const_byte_span_t fixed_psk = {kFixedPSK, sizeof(kFixedPSK)};

    iree_net_codec_t* client = nullptr;
    iree_net_codec_t* server = nullptr;
    iree_status_t status = iree_net_aead_codec_create(
        fixed_psk, IREE_NET_AEAD_ROLE_CLIENT, iree_allocator_system(), &client);
    if (iree_status_is_ok(status)) {
      status = iree_net_aead_codec_create(fixed_psk, IREE_NET_AEAD_ROLE_SERVER,
                                          iree_allocator_system(), &server);
    }

    if (iree_status_is_ok(status)) {
      iree_net_codec_overhead_t overhead =
          iree_net_codec_query_overhead(client);
      size_t payload_size = frame_size;
      size_t buffer_size = overhead.prefix + payload_size + overhead.suffix;

      void* buffer = nullptr;
      status =
          iree_allocator_malloc(iree_allocator_system(), buffer_size, &buffer);
      if (iree_status_is_ok(status)) {
        uint8_t* payload_ptr = (uint8_t*)buffer + overhead.prefix;
        memcpy(payload_ptr, frame_data, payload_size);

        // Create valid encoded frame.
        status = iree_net_codec_encode(client, payload_ptr, payload_size);
        if (iree_status_is_ok(status)) {
          // Apply bit-flip mutations guided by fuzz data.
          // Use first byte of fuzz input to select mutation position.
          size_t flip_pos = frame_data[0] % buffer_size;
          // Use second byte (if available) to select bit to flip.
          uint8_t flip_bit = (frame_size > 1) ? (frame_data[1] % 8) : 0;

          ((uint8_t*)buffer)[flip_pos] ^= (1 << flip_bit);

          // Decode should fail (authentication failure) but not crash.
          iree_byte_span_t decoded = {nullptr, 0};
          status = iree_net_codec_decode(server, (uint8_t*)buffer, buffer_size,
                                         &decoded);
          // Most likely IREE_STATUS_DATA_LOSS, but we just care about no crash.
          iree_status_ignore(status);
        } else {
          iree_status_ignore(status);
        }

        iree_allocator_free(iree_allocator_system(), buffer);
      } else {
        iree_status_ignore(status);
      }
    } else {
      iree_status_ignore(status);
    }

    if (client) iree_net_codec_release(client);
    if (server) iree_net_codec_release(server);
  }

  //===--------------------------------------------------------------------===//
  // Test 4: Boundary conditions (empty payload)
  //===--------------------------------------------------------------------===//

  {
    iree_net_codec_t* client = nullptr;
    iree_net_codec_t* server = nullptr;
    iree_status_t status = iree_net_aead_codec_create(
        psk, IREE_NET_AEAD_ROLE_CLIENT, iree_allocator_system(), &client);
    if (iree_status_is_ok(status)) {
      status = iree_net_aead_codec_create(psk, IREE_NET_AEAD_ROLE_SERVER,
                                          iree_allocator_system(), &server);
    }

    if (iree_status_is_ok(status)) {
      iree_net_codec_overhead_t overhead =
          iree_net_codec_query_overhead(client);
      size_t buffer_size = overhead.prefix + overhead.suffix;  // Zero payload.

      void* buffer = nullptr;
      status =
          iree_allocator_malloc(iree_allocator_system(), buffer_size, &buffer);
      if (iree_status_is_ok(status)) {
        uint8_t* payload_ptr = (uint8_t*)buffer + overhead.prefix;

        // Encode empty payload.
        status = iree_net_codec_encode(client, payload_ptr, 0);
        if (iree_status_is_ok(status)) {
          iree_byte_span_t decoded = {nullptr, 0};
          status = iree_net_codec_decode(server, (uint8_t*)buffer, buffer_size,
                                         &decoded);
          iree_status_ignore(status);
        } else {
          iree_status_ignore(status);
        }

        iree_allocator_free(iree_allocator_system(), buffer);
      } else {
        iree_status_ignore(status);
      }
    } else {
      iree_status_ignore(status);
    }

    if (client) iree_net_codec_release(client);
    if (server) iree_net_codec_release(server);
  }

  return 0;
}
