// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/address_impl.h"
#include "iree/async/socket.h"
#include "iree/base/api.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <stddef.h>
#include <string.h>
#include <winsock2.h>
#include <ws2tcpip.h>

// AF_UNIX support requires Windows 10 1803+ and the afunix.h header.
#if defined(AF_UNIX)
#include <afunix.h>
#define IREE_ASYNC_HAVE_UNIX_SOCKETS 1
#endif  // AF_UNIX

//===----------------------------------------------------------------------===//
// iree_async_address_from_ipv4
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_ipv4(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address) {
  memset(out_address, 0, sizeof(*out_address));
  struct sockaddr_in* addr = (struct sockaddr_in*)out_address->storage;
  addr->sin_family = AF_INET;
  addr->sin_port = iree_async_htons(port);
  if (iree_string_view_is_empty(host)) {
    // INADDR_ANY (all zeros, already set by memset).
  } else {
    uint8_t octets[4];
    IREE_RETURN_IF_ERROR(iree_async_address_parse_ipv4(host, octets));
    memcpy(&addr->sin_addr, octets, 4);
  }
  out_address->length = sizeof(struct sockaddr_in);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_async_address_from_ipv6
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_ipv6(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address) {
  memset(out_address, 0, sizeof(*out_address));
  struct sockaddr_in6* addr = (struct sockaddr_in6*)out_address->storage;
  addr->sin6_family = AF_INET6;
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

    IREE_RETURN_IF_ERROR(
        iree_async_address_parse_ipv6(host, addr->sin6_addr.s6_addr));

    // Resolve zone to scope_id (numeric only on Windows).
    if (!iree_string_view_is_empty(zone)) {
      uint32_t scope_id = 0;
      if (!iree_string_view_atoi_uint32(zone, &scope_id)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "non-numeric IPv6 zone IDs not supported on "
                                "Windows (use numeric interface index)");
      }
      addr->sin6_scope_id = scope_id;
    } else if (percent_index != IREE_STRING_VIEW_NPOS) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "empty IPv6 zone ID after '%%'");
    }
  }
  out_address->length = sizeof(struct sockaddr_in6);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_async_address_from_unix
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_unix(
    iree_string_view_t path, iree_async_address_t* out_address) {
#if defined(IREE_ASYNC_HAVE_UNIX_SOCKETS)
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Unix socket path must not be empty");
  }

  // Abstract namespace is not supported on Windows.
  if (path.size > 0 && path.data[0] == '@') {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "abstract namespace Unix sockets are only supported on Linux");
  }

  memset(out_address, 0, sizeof(*out_address));
  struct sockaddr_un* addr = (struct sockaddr_un*)out_address->storage;
  addr->sun_family = AF_UNIX;

  if (path.size >= sizeof(addr->sun_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Unix socket path too long (%" PRIhsz
                            " bytes, max %zu bytes)",
                            path.size, sizeof(addr->sun_path) - 1);
  }
  memcpy(addr->sun_path, path.data, path.size);
  addr->sun_path[path.size] = '\0';
  out_address->length = offsetof(struct sockaddr_un, sun_path) + path.size + 1;
  return iree_ok_status();
#else
  (void)path;
  (void)out_address;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "Unix domain sockets not supported on this platform");
#endif  // IREE_ASYNC_HAVE_UNIX_SOCKETS
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

  const struct sockaddr* sa = (const struct sockaddr*)address->storage;
  char temp[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_host_size_t length = 0;

  switch (sa->sa_family) {
    case AF_INET: {
      const struct sockaddr_in* addr = (const struct sockaddr_in*)sa;
      const uint8_t* octets = (const uint8_t*)&addr->sin_addr;
      uint16_t port = iree_async_ntohs(addr->sin_port);
      length = iree_async_address_format_ipv4(octets, port, temp);
      break;
    }
    case AF_INET6: {
      const struct sockaddr_in6* addr = (const struct sockaddr_in6*)sa;
      uint16_t port = iree_async_ntohs(addr->sin6_port);
      length = iree_async_address_format_ipv6(addr->sin6_addr.s6_addr, port,
                                              addr->sin6_scope_id, temp);
      break;
    }
#if defined(IREE_ASYNC_HAVE_UNIX_SOCKETS)
    case AF_UNIX: {
      const struct sockaddr_un* addr = (const struct sockaddr_un*)sa;
      iree_host_size_t path_offset = offsetof(struct sockaddr_un, sun_path);
      iree_host_size_t max_path = sizeof(addr->sun_path);
      iree_host_size_t stored =
          address->length > path_offset ? address->length - path_offset : 0;
      if (stored > max_path) stored = max_path;
      if (stored > 0 && addr->sun_path[stored - 1] == '\0') {
        stored -= 1;
      }
      iree_string_view_t path = {addr->sun_path, stored};
      length =
          iree_async_address_format_unix(path, /*is_abstract=*/false, temp);
      break;
    }
#endif  // IREE_ASYNC_HAVE_UNIX_SOCKETS
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown address family %d", (int)sa->sa_family);
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

#endif  // IREE_PLATFORM_WINDOWS
