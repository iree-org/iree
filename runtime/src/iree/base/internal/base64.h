// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base64 decoding (RFC 4648).
//
// Standard alphabet: A-Z, a-z, 0-9, +, / with = padding.
// Handles missing trailing padding gracefully (1-2 missing = characters).

#ifndef IREE_BASE_INTERNAL_BASE64_H_
#define IREE_BASE_INTERNAL_BASE64_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the maximum decoded byte count for base64-encoded data.
// The actual decoded length may be 1-2 bytes less when padding is present.
// Returns 0 for empty input.
static inline iree_host_size_t iree_base64_decoded_size(
    iree_string_view_t encoded) {
  if (encoded.size == 0) return 0;
  // Every 4 base64 characters encode 3 bytes. Unpadded trailing groups of
  // 2 characters encode 1 byte, 3 characters encode 2 bytes.
  return (encoded.size / 4) * 3 +
         ((encoded.size % 4) ? (encoded.size % 4) - 1 : 0);
}

// Decodes base64-encoded data into |out_buffer|.
//
// |out_buffer_capacity| must be at least iree_base64_decoded_size(encoded)
// bytes. On success, |out_length| receives the actual decoded byte count.
//
// Empty input is valid and produces 0 decoded bytes.
//
// Returns:
//   IREE_STATUS_OK on success.
//   IREE_STATUS_INVALID_ARGUMENT for invalid base64 characters, invalid
//     padding, or input length that cannot represent valid base64 (length
//     mod 4 == 1 without padding).
//   IREE_STATUS_OUT_OF_RANGE if |out_buffer_capacity| is too small.
iree_status_t iree_base64_decode(iree_string_view_t encoded,
                                 iree_host_size_t out_buffer_capacity,
                                 uint8_t* out_buffer,
                                 iree_host_size_t* out_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_BASE64_H_
