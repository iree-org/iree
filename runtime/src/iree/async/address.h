// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Network address abstraction.
//
// iree_async_address_t is a value type that wraps platform sockaddr structures
// without exposing platform socket headers. It supports IPv4, IPv6, and Unix
// domain addresses.
//
// Addresses can be constructed from typed components (from_ipv4, from_ipv6,
// from_unix), parsed from strings (from_string), or formatted back to strings
// (format). The string representation is round-trippable:
//   format(from_string(format(address))) == format(address)
//
// String formats:
//   IPv4:  "host:port"    (e.g., "127.0.0.1:8080", "0.0.0.0:0")
//   IPv6:  "[host]:port"  (e.g., "[::1]:80", "[fe80::1%3]:443")
//   Unix:  "unix:path"    (e.g., "unix:/tmp/sock", "unix:@abstract")

#ifndef IREE_ASYNC_ADDRESS_H_
#define IREE_ASYNC_ADDRESS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_async_address_t
//===----------------------------------------------------------------------===//

// Network address (value type). Encapsulates sockaddr_storage without exposing
// platform socket headers. Can represent IPv4, IPv6, and Unix domain addresses.
typedef struct iree_async_address_t {
  // Sufficient for sockaddr_storage on all platforms.
  uint8_t storage[128];
  // Actual address byte length within storage.
  iree_host_size_t length;
} iree_async_address_t;

// Constructs an IPv4 address from a dotted-quad string and port.
// |host| may be empty for INADDR_ANY.
IREE_API_EXPORT iree_status_t iree_async_address_from_ipv4(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address);

// Constructs an IPv6 address from a colon-hex string and port.
// |host| may be empty for in6addr_any.
IREE_API_EXPORT iree_status_t iree_async_address_from_ipv6(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address);

// Constructs a Unix domain socket address from a filesystem path.
// Paths prefixed with '@' are treated as abstract namespace sockets
// (Linux/Android only; returns IREE_STATUS_UNAVAILABLE on other platforms).
IREE_API_EXPORT iree_status_t iree_async_address_from_unix(
    iree_string_view_t path, iree_async_address_t* out_address);

// Parses an address string into an iree_async_address_t.
// Recognized formats:
//   IPv4:  "host:port"    (e.g., "127.0.0.1:8080", ":0" for INADDR_ANY)
//   IPv6:  "[host]:port"  (e.g., "[::1]:80", "[fe80::1%3]:443")
//   Unix:  "unix:path"    (e.g., "unix:/tmp/sock", "unix:@abstract")
//
// Port must be a decimal integer in [0, 65535] with no leading zeros.
// The formats match those produced by iree_async_address_format().
IREE_API_EXPORT iree_status_t iree_async_address_from_string(
    iree_string_view_t address_str, iree_async_address_t* out_address);

// Maximum formatted length for any address type.
// Sufficient for IPv6 "[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:65535" (47)
// and Unix "unix:" + max sun_path (113). A buffer of this size always succeeds.
#define IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH 128

// Formats an address as a human-readable string (e.g. "127.0.0.1:8080").
// Writes up to |buffer_capacity| bytes to |buffer| (no NUL terminator).
// On success, |out_value| is set to a view into |buffer| with the formatted
// string. Returns IREE_STATUS_OUT_OF_RANGE if |buffer_capacity| is
// insufficient.
IREE_API_EXPORT iree_status_t iree_async_address_format(
    const iree_async_address_t* address, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_ADDRESS_H_
