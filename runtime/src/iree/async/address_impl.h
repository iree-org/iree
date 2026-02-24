// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private implementation helpers for address parsing and formatting.
//
// These functions operate on raw bytes without any platform socket headers.
// They are implemented in address.c and called by the platform-specific
// address_posix.c / address_win32.c / address_generic.c files.

#ifndef IREE_ASYNC_ADDRESS_IMPL_H_
#define IREE_ASYNC_ADDRESS_IMPL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Byte-order helpers
//===----------------------------------------------------------------------===//

// Converts a 16-bit value from host byte order to network byte order
// (big-endian).
static inline uint16_t iree_async_htons(uint16_t value) {
#if defined(IREE_ENDIANNESS_LITTLE)
  return (uint16_t)((value >> 8) | (value << 8));
#else
  return value;
#endif
}

// Converts a 16-bit value from network byte order (big-endian) to host byte
// order.
static inline uint16_t iree_async_ntohs(uint16_t value) {
#if defined(IREE_ENDIANNESS_LITTLE)
  return (uint16_t)((value >> 8) | (value << 8));
#else
  return value;
#endif
}

//===----------------------------------------------------------------------===//
// IPv4 parsing and formatting
//===----------------------------------------------------------------------===//

// Parses a dotted-quad IPv4 address string into 4 bytes in network order.
// Validates:
//   - Exactly 4 decimal octets separated by 3 dots
//   - Each octet in range [0, 255]
//   - No leading zeros (rejects "01.02.03.04")
//   - No trailing characters after the fourth octet
//   - No hex digits, whitespace, or empty octets
// Returns IREE_STATUS_INVALID_ARGUMENT on any validation failure.
iree_status_t iree_async_address_parse_ipv4(iree_string_view_t host,
                                            uint8_t out_octets[4]);

// Formats an IPv4 address and port as "o.o.o.o:p" into |buffer|.
// |buffer| must have at least IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH bytes.
// |octets| is the 4-byte address in network order.
// |port| is in host byte order.
// Returns the number of characters written (no NUL is appended).
iree_host_size_t iree_async_address_format_ipv4(const uint8_t octets[4],
                                                uint16_t port, char* buffer);

//===----------------------------------------------------------------------===//
// IPv6 parsing and formatting
//===----------------------------------------------------------------------===//

// Parses a colon-hex IPv6 address string into 16 bytes in network order.
// Supports:
//   - Full 8-group form (2001:0db8:0000:0000:0000:0000:0000:0001)
//   - Zero compression with :: (at most one occurrence)
//   - IPv4-mapped form (::ffff:192.168.1.1)
// Validates:
//   - 1-8 groups of 1-4 hex digits
//   - At most one :: for zero compression
//   - No invalid hex characters
// Returns IREE_STATUS_INVALID_ARGUMENT on any validation failure.
iree_status_t iree_async_address_parse_ipv6(iree_string_view_t host,
                                            uint8_t out_bytes[16]);

// Formats an IPv6 address and port as "[addr]:p" using RFC 5952 shortest form.
// |buffer| must have at least IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH bytes.
// Rules applied:
//   - Longest run of consecutive all-zero 16-bit groups replaced with ::
//   - Ties broken by leftmost run
//   - Single zero group is NOT compressed (remains "0")
//   - Hex digits in lowercase, no leading zeros in groups
//   - IPv4-mapped addresses (::ffff:0:0/96 prefix) shown as [::ffff:a.b.c.d]
//   - Non-zero scope_id appended as %<id> inside brackets
// |bytes| is the 16-byte address in network order.
// |port| is in host byte order.
// |scope_id| is the interface scope (0 means none, omitted from output).
// Returns the number of characters written (no NUL is appended).
iree_host_size_t iree_async_address_format_ipv6(const uint8_t bytes[16],
                                                uint16_t port,
                                                uint32_t scope_id,
                                                char* buffer);

//===----------------------------------------------------------------------===//
// Unix path formatting
//===----------------------------------------------------------------------===//

// Formats a Unix socket path as "unix:/path" or "unix:@name" (abstract).
// |buffer| must have at least IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH bytes.
// |path| is the socket path (for abstract, this is the name without the
// leading NUL byte). |is_abstract| indicates abstract namespace.
// Returns the number of characters written (no NUL is appended).
iree_host_size_t iree_async_address_format_unix(iree_string_view_t path,
                                                bool is_abstract, char* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_ADDRESS_IMPL_H_
