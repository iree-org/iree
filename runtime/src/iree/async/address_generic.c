// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/address_impl.h"
#include "iree/async/socket.h"
#include "iree/base/api.h"

#if defined(IREE_PLATFORM_GENERIC) || defined(IREE_PLATFORM_EMSCRIPTEN)

#include <stddef.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Self-contained sockaddr definitions (no platform socket headers needed).
// Layout matches Linux for consistency. These are only used as the internal
// storage format within iree_async_address_t on generic platforms.
//===----------------------------------------------------------------------===//

#define IREE_AF_INET 2
#define IREE_AF_INET6 10

typedef struct {
  uint16_t sin_family;
  uint16_t sin_port;
  uint8_t sin_addr[4];
  uint8_t sin_zero[8];
} iree_sockaddr_in_t;

typedef struct {
  uint16_t sin6_family;
  uint16_t sin6_port;
  uint32_t sin6_flowinfo;
  uint8_t sin6_addr[16];
  uint32_t sin6_scope_id;
} iree_sockaddr_in6_t;

//===----------------------------------------------------------------------===//
// iree_async_address_from_ipv4
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_ipv4(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address) {
  memset(out_address, 0, sizeof(*out_address));
  iree_sockaddr_in_t* addr = (iree_sockaddr_in_t*)out_address->storage;
  addr->sin_family = IREE_AF_INET;
  addr->sin_port = iree_async_htons(port);
  if (iree_string_view_is_empty(host)) {
    // INADDR_ANY (all zeros, already set by memset).
  } else {
    IREE_RETURN_IF_ERROR(iree_async_address_parse_ipv4(host, addr->sin_addr));
  }
  out_address->length = sizeof(iree_sockaddr_in_t);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_async_address_from_ipv6
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_ipv6(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address) {
  memset(out_address, 0, sizeof(*out_address));
  iree_sockaddr_in6_t* addr = (iree_sockaddr_in6_t*)out_address->storage;
  addr->sin6_family = IREE_AF_INET6;
  addr->sin6_port = iree_async_htons(port);
  if (iree_string_view_is_empty(host)) {
    // in6addr_any (all zeros, already set by memset).
  } else {
    // Split zone suffix (%<zone>) before parsing the address.
    iree_string_view_t zone = iree_string_view_empty();
    iree_host_size_t percent_index = iree_string_view_find_char(host, '%', 0);
    if (percent_index != IREE_STRING_VIEW_NPOS) {
      zone =
          iree_string_view_substr(host, percent_index + 1, IREE_HOST_SIZE_MAX);
      host = iree_string_view_substr(host, 0, percent_index);
    }

    IREE_RETURN_IF_ERROR(iree_async_address_parse_ipv6(host, addr->sin6_addr));

    // Resolve zone to scope_id (numeric only on generic platforms).
    if (!iree_string_view_is_empty(zone)) {
      uint32_t scope_id = 0;
      if (!iree_string_view_atoi_uint32(zone, &scope_id)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "non-numeric IPv6 zone IDs not supported on "
                                "this platform (use numeric interface index)");
      }
      addr->sin6_scope_id = scope_id;
    } else if (percent_index != IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty IPv6 zone ID after '%%'");
    }
  }
  out_address->length = sizeof(iree_sockaddr_in6_t);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_async_address_from_unix
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_unix(
    iree_string_view_t path, iree_async_address_t* out_address) {
  (void)path;
  (void)out_address;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Unix domain sockets not supported on this platform");
}

//===----------------------------------------------------------------------===//
// iree_async_address_format
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_format(
    const iree_async_address_t* address, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_value) {
  if (address->length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty address");
  }

  // Read family from first 2 bytes (uint16_t at offset 0 in both
  // iree_sockaddr_in_t and iree_sockaddr_in6_t).
  uint16_t family;
  memcpy(&family, address->storage, sizeof(family));

  char temp[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_host_size_t length = 0;

  switch (family) {
    case IREE_AF_INET: {
      const iree_sockaddr_in_t* addr =
          (const iree_sockaddr_in_t*)address->storage;
      uint16_t port = iree_async_ntohs(addr->sin_port);
      length = iree_async_address_format_ipv4(addr->sin_addr, port, temp);
      break;
    }
    case IREE_AF_INET6: {
      const iree_sockaddr_in6_t* addr =
          (const iree_sockaddr_in6_t*)address->storage;
      uint16_t port = iree_async_ntohs(addr->sin6_port);
      length = iree_async_address_format_ipv6(addr->sin6_addr, port,
                                              addr->sin6_scope_id, temp);
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown address family %d", (int)family);
  }

  if (length > buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "address format requires %" PRIhsz
                            " bytes but buffer capacity is %" PRIhsz,
                            length, buffer_capacity);
  }

  memcpy(buffer, temp, length);
  if (out_value) {
    out_value->data = buffer;
    out_value->size = length;
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_GENERIC || IREE_PLATFORM_EMSCRIPTEN
