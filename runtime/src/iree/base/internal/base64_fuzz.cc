// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "iree/base/internal/base64.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Phase 1: Fuzz decode — arbitrary bytes as base64 input.
  {
    iree_string_view_t encoded = {(const char*)data, size};
    iree_host_size_t max_size = iree_base64_decoded_size(encoded);
    if (max_size > 1024 * 1024) return 0;

    uint8_t* buffer = nullptr;
    if (max_size > 0) {
      buffer = (uint8_t*)malloc(max_size);
      if (!buffer) return 0;
    }

    iree_host_size_t actual_length = 0;
    iree_status_t status = iree_base64_decode(
        encoded, iree_make_byte_span(buffer, max_size), &actual_length);
    iree_status_ignore(status);

    free(buffer);
  }

  // Phase 2: Fuzz roundtrip — treat input as raw binary, encode then decode.
  {
    if (size > 256 * 1024) return 0;

    iree_const_byte_span_t input = iree_make_const_byte_span(data, size);
    iree_host_size_t encoded_size = iree_base64_encoded_size(size);

    char* encode_buffer = (char*)malloc(encoded_size);
    if (!encode_buffer && encoded_size > 0) return 0;

    iree_host_size_t encode_length = 0;
    iree_status_t encode_status = iree_base64_encode(
        input, iree_make_mutable_string_view(encode_buffer, encoded_size),
        &encode_length);
    if (!iree_status_is_ok(encode_status)) {
      iree_status_ignore(encode_status);
      free(encode_buffer);
      return 0;
    }

    // Decode the encoded result — must succeed and match original.
    iree_string_view_t encoded =
        iree_make_string_view(encode_buffer, encode_length);
    iree_host_size_t decoded_size = iree_base64_decoded_size(encoded);

    uint8_t* decode_buffer = (uint8_t*)malloc(decoded_size);
    if (!decode_buffer && decoded_size > 0) {
      free(encode_buffer);
      return 0;
    }

    iree_host_size_t decode_length = 0;
    iree_status_t decode_status = iree_base64_decode(
        encoded, iree_make_byte_span(decode_buffer, decoded_size),
        &decode_length);
    // Roundtrip must always succeed.
    if (!iree_status_is_ok(decode_status)) {
      __builtin_trap();
    }
    // Decoded data must match original input.
    if (decode_length != size) {
      __builtin_trap();
    }
    for (size_t i = 0; i < size; ++i) {
      if (decode_buffer[i] != data[i]) {
        __builtin_trap();
      }
    }

    free(decode_buffer);
    free(encode_buffer);
  }

  return 0;
}
