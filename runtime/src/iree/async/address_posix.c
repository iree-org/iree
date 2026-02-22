// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/address_impl.h"
#include "iree/async/socket.h"
#include "iree/base/api.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)

#include <net/if.h>
#include <netinet/in.h>
#include <stddef.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

// BSD-family platforms have a length field in their sockaddr structs.
#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
#define IREE_ASYNC_HAVE_SOCKADDR_LEN 1
#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD

//===----------------------------------------------------------------------===//
// iree_async_address_from_ipv4
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_address_from_ipv4(
    iree_string_view_t host, uint16_t port, iree_async_address_t* out_address) {
  memset(out_address, 0, sizeof(*out_address));
  struct sockaddr_in* addr = (struct sockaddr_in*)out_address->storage;
#if defined(IREE_ASYNC_HAVE_SOCKADDR_LEN)
  addr->sin_len = sizeof(struct sockaddr_in);
#endif  // IREE_ASYNC_HAVE_SOCKADDR_LEN
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
#if defined(IREE_ASYNC_HAVE_SOCKADDR_LEN)
  addr->sin6_len = sizeof(struct sockaddr_in6);
#endif  // IREE_ASYNC_HAVE_SOCKADDR_LEN
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

    // Resolve zone to scope_id.
    if (!iree_string_view_is_empty(zone)) {
      uint32_t scope_id = 0;
      if (iree_string_view_atoi_uint32(zone, &scope_id)) {
        addr->sin6_scope_id = scope_id;
      } else {
        // Interface name: requires NUL-terminated copy for if_nametoindex.
        char name_buffer[IF_NAMESIZE];
        if (zone.size >= sizeof(name_buffer)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "IPv6 zone ID too long");
        }
        memcpy(name_buffer, zone.data, zone.size);
        name_buffer[zone.size] = '\0';
        unsigned int ifindex = if_nametoindex(name_buffer);
        if (ifindex == 0) {
          return iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "unknown network interface for zone ID");
        }
        addr->sin6_scope_id = ifindex;
      }
    } else if (percent_index != IREE_STRING_VIEW_NPOS) {
      // Trailing '%' with no zone ID.
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
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Unix socket path must not be empty");
  }

  memset(out_address, 0, sizeof(*out_address));
  struct sockaddr_un* addr = (struct sockaddr_un*)out_address->storage;
#if defined(IREE_ASYNC_HAVE_SOCKADDR_LEN)
  addr->sun_len = sizeof(struct sockaddr_un);
#endif  // IREE_ASYNC_HAVE_SOCKADDR_LEN
  addr->sun_family = AF_UNIX;

  // Abstract namespace: path starts with '@' (Linux/Android only).
  if (path.size > 0 && path.data[0] == '@') {
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
    iree_host_size_t name_length = path.size - 1;
    if (name_length >= sizeof(addr->sun_path)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "abstract Unix socket name too long (%" PRIhsz
                              " bytes, max %zu bytes)",
                              name_length, sizeof(addr->sun_path) - 1);
    }
    addr->sun_path[0] = '\0';
    memcpy(addr->sun_path + 1, path.data + 1, name_length);
    // Abstract namespace: length includes the NUL byte but not a trailing NUL.
    out_address->length =
        offsetof(struct sockaddr_un, sun_path) + 1 + name_length;
#else
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "abstract namespace Unix sockets are only supported on Linux");
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
  } else {
    // Filesystem path.
    if (path.size >= sizeof(addr->sun_path)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Unix socket path too long (%" PRIhsz
                              " bytes, max %zu bytes)",
                              path.size, sizeof(addr->sun_path) - 1);
    }
    memcpy(addr->sun_path, path.data, path.size);
    addr->sun_path[path.size] = '\0';
    out_address->length =
        offsetof(struct sockaddr_un, sun_path) + path.size + 1;
  }

  return iree_ok_status();
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
    case AF_UNIX: {
      const struct sockaddr_un* addr = (const struct sockaddr_un*)sa;
      iree_host_size_t path_offset = offsetof(struct sockaddr_un, sun_path);
      iree_host_size_t max_path = sizeof(addr->sun_path);
      if (address->length > path_offset && addr->sun_path[0] == '\0') {
        // Abstract namespace.
        iree_host_size_t name_length = address->length - path_offset - 1;
        if (name_length > max_path - 1) name_length = max_path - 1;
        iree_string_view_t name = {addr->sun_path + 1, name_length};
        length =
            iree_async_address_format_unix(name, /*is_abstract=*/true, temp);
      } else {
        // Filesystem path: the stored length may or may not include a
        // trailing NUL. Our constructor always includes it, but
        // kernel-sourced addresses from getsockname/accept may not.
        iree_host_size_t stored =
            address->length > path_offset ? address->length - path_offset : 0;
        if (stored > max_path) stored = max_path;
        if (stored > 0 && addr->sun_path[stored - 1] == '\0') {
          stored -= 1;
        }
        iree_string_view_t path = {addr->sun_path, stored};
        length =
            iree_async_address_format_unix(path, /*is_abstract=*/false, temp);
      }
      break;
    }
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

#endif  // IREE_PLATFORM_LINUX || ANDROID || APPLE || BSD
