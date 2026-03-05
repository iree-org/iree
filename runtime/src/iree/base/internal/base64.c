// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/base64.h"

// Decode table: maps ASCII byte value to 6-bit decoded value.
// 0xFF indicates an invalid character. 0xFE indicates padding ('=').
// clang-format off
static const uint8_t iree_base64_decode_table[256] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F,  // +, /
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,  // 0-7
    0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF,  // 8-9, =
    0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,  // A-G
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,  // H-O
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,  // P-W
    0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  // X-Z
    0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,  // a-g
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,  // h-o
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,  // p-w
    0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  // x-z
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
};
// clang-format on

iree_status_t iree_base64_decode(iree_string_view_t encoded,
                                 iree_host_size_t out_buffer_capacity,
                                 uint8_t* out_buffer,
                                 iree_host_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(out_length);
  *out_length = 0;

  if (encoded.size == 0) {
    return iree_ok_status();
  }

  IREE_ASSERT_ARGUMENT(out_buffer);

  // Strip trailing padding to simplify processing. We handle 0-2 padding
  // characters at the end.
  iree_host_size_t input_length = encoded.size;
  iree_host_size_t padding_count = 0;
  while (input_length > 0 && encoded.data[input_length - 1] == '=') {
    ++padding_count;
    --input_length;
  }
  if (padding_count > 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid base64 padding: %zu '=' characters "
                            "(maximum 2)",
                            padding_count);
  }

  // After stripping padding, the remaining length determines the decoded size.
  // Valid unpadded lengths mod 4: 0 (empty or full groups), 2 (1 extra byte),
  // 3 (2 extra bytes). Length mod 4 == 1 is invalid (6 bits cannot encode a
  // whole byte).
  iree_host_size_t remainder = input_length % 4;
  if (remainder == 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid base64 length: %zu characters (after removing padding) "
        "leaves 1 trailing character which cannot encode a complete byte",
        input_length);
  }

  iree_host_size_t full_groups = input_length / 4;
  iree_host_size_t decoded_length = full_groups * 3;
  if (remainder == 2) {
    decoded_length += 1;
  } else if (remainder == 3) {
    decoded_length += 2;
  }

  if (out_buffer_capacity < decoded_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "base64 decode buffer too small: need %zu bytes "
                            "but only %zu available",
                            decoded_length, out_buffer_capacity);
  }

  // Decode full 4-character groups (each produces 3 bytes).
  iree_host_size_t output_position = 0;
  iree_host_size_t input_position = 0;
  for (iree_host_size_t i = 0; i < full_groups; ++i) {
    uint8_t a = iree_base64_decode_table[(uint8_t)encoded.data[input_position]];
    uint8_t b =
        iree_base64_decode_table[(uint8_t)encoded.data[input_position + 1]];
    uint8_t c =
        iree_base64_decode_table[(uint8_t)encoded.data[input_position + 2]];
    uint8_t d =
        iree_base64_decode_table[(uint8_t)encoded.data[input_position + 3]];
    if ((a | b | c | d) & 0x80) {
      // At least one invalid character. Find and report the first one.
      for (int j = 0; j < 4; ++j) {
        uint8_t v =
            iree_base64_decode_table[(uint8_t)encoded.data[input_position + j]];
        if (v & 0x80) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "invalid base64 character '%c' (0x%02X) at position %zu",
              encoded.data[input_position + j],
              (unsigned)encoded.data[input_position + j], input_position + j);
        }
      }
    }
    out_buffer[output_position] = (uint8_t)((a << 2) | (b >> 4));
    out_buffer[output_position + 1] = (uint8_t)((b << 4) | (c >> 2));
    out_buffer[output_position + 2] = (uint8_t)((c << 6) | d);
    input_position += 4;
    output_position += 3;
  }

  // Decode trailing 2 or 3 characters (produces 1 or 2 bytes).
  if (remainder == 2) {
    uint8_t a = iree_base64_decode_table[(uint8_t)encoded.data[input_position]];
    uint8_t b =
        iree_base64_decode_table[(uint8_t)encoded.data[input_position + 1]];
    if ((a | b) & 0x80) {
      for (int j = 0; j < 2; ++j) {
        uint8_t v =
            iree_base64_decode_table[(uint8_t)encoded.data[input_position + j]];
        if (v & 0x80) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "invalid base64 character '%c' (0x%02X) at position %zu",
              encoded.data[input_position + j],
              (unsigned)encoded.data[input_position + j], input_position + j);
        }
      }
    }
    out_buffer[output_position] = (uint8_t)((a << 2) | (b >> 4));
    output_position += 1;
  } else if (remainder == 3) {
    uint8_t a = iree_base64_decode_table[(uint8_t)encoded.data[input_position]];
    uint8_t b =
        iree_base64_decode_table[(uint8_t)encoded.data[input_position + 1]];
    uint8_t c =
        iree_base64_decode_table[(uint8_t)encoded.data[input_position + 2]];
    if ((a | b | c) & 0x80) {
      for (int j = 0; j < 3; ++j) {
        uint8_t v =
            iree_base64_decode_table[(uint8_t)encoded.data[input_position + j]];
        if (v & 0x80) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "invalid base64 character '%c' (0x%02X) at position %zu",
              encoded.data[input_position + j],
              (unsigned)encoded.data[input_position + j], input_position + j);
        }
      }
    }
    out_buffer[output_position] = (uint8_t)((a << 2) | (b >> 4));
    out_buffer[output_position + 1] = (uint8_t)((b << 4) | (c >> 2));
    output_position += 2;
  }

  *out_length = output_position;
  return iree_ok_status();
}
