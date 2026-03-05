// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base64 encoding and decoding (RFC 4648).
//
// Standard alphabet: A-Z, a-z, 0-9, +, / with = padding.
// Decode handles missing trailing padding gracefully (1-2 missing =
// characters). Encode always produces padded output.

#ifndef IREE_BASE_INTERNAL_BASE64_H_
#define IREE_BASE_INTERNAL_BASE64_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Encoding (binary -> base64 text)
//===----------------------------------------------------------------------===//

// Returns the exact encoded size for |data_length| bytes of binary data.
// The result is always a multiple of 4 (padded output).
// Returns IREE_HOST_SIZE_MAX on overflow (pathological inputs only).
// IREE_HOST_SIZE_MAX is never a valid encoded size since the result is always
// a multiple of 4 and IREE_HOST_SIZE_MAX is not.
static inline iree_host_size_t iree_base64_encoded_size(
    iree_host_size_t data_length) {
  // Each 3-byte group produces 4 base64 characters, rounded up.
  iree_host_size_t padded = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(data_length, 2, &padded))) {
    return IREE_HOST_SIZE_MAX;
  }
  iree_host_size_t result = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(padded / 3, 4, &result))) {
    return IREE_HOST_SIZE_MAX;
  }
  return result;
}

// Encodes binary data to base64 text with = padding.
//
// |out_string.size| must be at least iree_base64_encoded_size() bytes.
// On success, |out_string_length| receives the number of characters written.
//
// The output is NOT null-terminated. If you need a null terminator, allocate
// one extra byte and set it yourself.
//
// Returns IREE_STATUS_OUT_OF_RANGE if |out_string.size| is too small.
iree_status_t iree_base64_encode(iree_const_byte_span_t data,
                                 iree_mutable_string_view_t out_string,
                                 iree_host_size_t* out_string_length);

//===----------------------------------------------------------------------===//
// Decoding (base64 text -> binary)
//===----------------------------------------------------------------------===//

// Returns the exact decoded byte count for base64-encoded data.
// Accounts for trailing = padding if present.
// Returns 0 for empty input.
// Returns IREE_HOST_SIZE_MAX on overflow (provably unreachable since
// (L/4)*3+2 < SIZE_MAX for any L, but checked for defense-in-depth).
static inline iree_host_size_t iree_base64_decoded_size(
    iree_string_view_t encoded) {
  if (encoded.size == 0) return 0;
  // Strip trailing padding to get the data-carrying length.
  iree_host_size_t length = encoded.size;
  while (length > 0 && encoded.data[length - 1] == '=') --length;
  // Every 4 base64 characters encode 3 bytes. Trailing groups of 2 characters
  // encode 1 byte, 3 characters encode 2 bytes.
  iree_host_size_t groups_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(length / 4, 3, &groups_size))) {
    return IREE_HOST_SIZE_MAX;
  }
  iree_host_size_t remainder_size = (length % 4) ? (length % 4) - 1 : 0;
  iree_host_size_t result = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(groups_size, remainder_size, &result))) {
    return IREE_HOST_SIZE_MAX;
  }
  return result;
}

// Decodes base64-encoded data into |out_buffer|.
//
// |out_buffer.data_length| must be at least iree_base64_decoded_size(encoded)
// bytes. On success, |out_length| receives the actual decoded byte count.
//
// Empty input is valid and produces 0 decoded bytes.
//
// Returns:
//   IREE_STATUS_OK on success.
//   IREE_STATUS_INVALID_ARGUMENT for invalid base64 characters, invalid
//     padding, or input length that cannot represent valid base64 (length
//     mod 4 == 1 without padding).
//   IREE_STATUS_OUT_OF_RANGE if |out_buffer.data_length| is too small.
iree_status_t iree_base64_decode(iree_string_view_t encoded,
                                 iree_byte_span_t out_buffer,
                                 iree_host_size_t* out_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_BASE64_H_
