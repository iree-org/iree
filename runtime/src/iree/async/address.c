// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/address.h"

#include <string.h>

#include "iree/async/address_impl.h"

//===----------------------------------------------------------------------===//
// Formatting helpers (write into guaranteed-large-enough buffer)
//===----------------------------------------------------------------------===//

// Writes a single decimal unsigned integer. Returns chars written.
static iree_host_size_t iree_address_write_uint(char* buffer, uint32_t value) {
  if (value == 0) {
    buffer[0] = '0';
    return 1;
  }
  char digits[10];
  int count = 0;
  while (value > 0) {
    digits[count++] = '0' + (char)(value % 10);
    value /= 10;
  }
  for (int i = count - 1; i >= 0; --i) {
    buffer[count - 1 - i] = digits[i];
  }
  return (iree_host_size_t)count;
}

// Writes a 1-4 digit lowercase hex value (no leading zeros). Returns chars
// written.
static iree_host_size_t iree_address_write_hex16(char* buffer, uint16_t value) {
  static const char hex_chars[] = "0123456789abcdef";
  if (value == 0) {
    buffer[0] = '0';
    return 1;
  }
  char digits[4];
  int count = 0;
  while (value > 0) {
    digits[count++] = hex_chars[value & 0xf];
    value >>= 4;
  }
  for (int i = count - 1; i >= 0; --i) {
    buffer[count - 1 - i] = digits[i];
  }
  return (iree_host_size_t)count;
}

//===----------------------------------------------------------------------===//
// IPv4 parsing
//===----------------------------------------------------------------------===//

iree_status_t iree_async_address_parse_ipv4(iree_string_view_t host,
                                            uint8_t out_octets[4]) {
  const char* data = host.data;
  iree_host_size_t length = host.size;
  iree_host_size_t position = 0;

  for (int octet_index = 0; octet_index < 4; ++octet_index) {
    // Each octet must start with a digit.
    if (position >= length || data[position] < '0' || data[position] > '9') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid IPv4 address: expected digit at "
                              "position %" PRIhsz,
                              position);
    }

    // Reject leading zeros (ambiguous octal interpretation).
    if (data[position] == '0' && position + 1 < length &&
        data[position + 1] >= '0' && data[position + 1] <= '9') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid IPv4 address: leading zero in octet %d",
                              octet_index);
    }

    // Accumulate decimal digits.
    uint32_t value = 0;
    while (position < length && data[position] >= '0' &&
           data[position] <= '9') {
      value = value * 10 + (uint32_t)(data[position] - '0');
      if (value > 255) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid IPv4 address: octet %d value exceeds 255", octet_index);
      }
      ++position;
    }

    out_octets[octet_index] = (uint8_t)value;

    // After octets 0-2, expect a dot separator.
    if (octet_index < 3) {
      if (position >= length || data[position] != '.') {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid IPv4 address: expected '.' after "
                                "octet %d",
                                octet_index);
      }
      ++position;
    }
  }

  // Reject trailing garbage.
  if (position != length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid IPv4 address: unexpected characters after address");
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// IPv6 parsing
//===----------------------------------------------------------------------===//

static inline int iree_hex_digit_value(char character) {
  if (character >= '0' && character <= '9') return character - '0';
  if (character >= 'a' && character <= 'f') return character - 'a' + 10;
  if (character >= 'A' && character <= 'F') return character - 'A' + 10;
  return -1;
}

iree_status_t iree_async_address_parse_ipv6(iree_string_view_t host,
                                            uint8_t out_bytes[16]) {
  const char* data = host.data;
  iree_host_size_t length = host.size;
  iree_host_size_t position = 0;

  memset(out_bytes, 0, 16);

  uint16_t groups[8];
  int group_count = 0;
  int compression_index = -1;

  while (position < length && group_count < 8) {
    // Check for :: (zero compression).
    if (position + 1 < length && data[position] == ':' &&
        data[position + 1] == ':') {
      if (compression_index >= 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid IPv6 address: multiple :: not allowed");
      }
      compression_index = group_count;
      position += 2;
      if (position >= length) break;
      continue;
    }

    // Colon separator between groups (not before first, not right after ::).
    if (group_count > 0 && compression_index != group_count) {
      if (position >= length || data[position] != ':') {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "invalid IPv6 address: expected ':' between groups");
      }
      ++position;
    }

    // Scan ahead for '.' to detect IPv4-mapped suffix.
    {
      iree_host_size_t scan = position;
      while (scan < length && data[scan] != ':' && data[scan] != '.') {
        ++scan;
      }
      if (scan < length && data[scan] == '.') {
        if (group_count > 6) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "invalid IPv6 address: too many groups "
                                  "before IPv4-mapped suffix");
        }
        iree_string_view_t ipv4_part = {data + position, length - position};
        uint8_t ipv4_octets[4];
        IREE_RETURN_IF_ERROR(
            iree_async_address_parse_ipv4(ipv4_part, ipv4_octets));
        groups[group_count++] =
            (uint16_t)((uint16_t)ipv4_octets[0] << 8 | ipv4_octets[1]);
        groups[group_count++] =
            (uint16_t)((uint16_t)ipv4_octets[2] << 8 | ipv4_octets[3]);
        position = length;
        break;
      }
    }

    // Parse hex group (1-4 hex digits).
    uint32_t value = 0;
    int digit_count = 0;
    while (position < length && digit_count < 4) {
      int digit = iree_hex_digit_value(data[position]);
      if (digit < 0) break;
      value = (value << 4) | (uint32_t)digit;
      ++digit_count;
      ++position;
    }

    if (digit_count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid IPv6 address: expected hex digit at position %" PRIhsz,
          position);
    }

    groups[group_count++] = (uint16_t)value;
  }

  // Reject trailing garbage.
  if (position != length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid IPv6 address: unexpected characters after address");
  }

  // Expand groups into 16 bytes.
  if (compression_index >= 0) {
    int groups_before = compression_index;
    int groups_after = group_count - compression_index;
    if (groups_before + groups_after >= 8) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid IPv6 address: too many groups with :: compression");
    }
    for (int i = 0; i < groups_before; ++i) {
      out_bytes[i * 2] = (uint8_t)(groups[i] >> 8);
      out_bytes[i * 2 + 1] = (uint8_t)(groups[i] & 0xff);
    }
    int end_offset = 8 - groups_after;
    for (int i = 0; i < groups_after; ++i) {
      int source_index = compression_index + i;
      out_bytes[(end_offset + i) * 2] = (uint8_t)(groups[source_index] >> 8);
      out_bytes[(end_offset + i) * 2 + 1] =
          (uint8_t)(groups[source_index] & 0xff);
    }
  } else {
    if (group_count != 8) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid IPv6 address: expected 8 groups, got %d (use :: for "
          "zero compression)",
          group_count);
    }
    for (int i = 0; i < 8; ++i) {
      out_bytes[i * 2] = (uint8_t)(groups[i] >> 8);
      out_bytes[i * 2 + 1] = (uint8_t)(groups[i] & 0xff);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// IPv4 formatting
//===----------------------------------------------------------------------===//

iree_host_size_t iree_async_address_format_ipv4(const uint8_t octets[4],
                                                uint16_t port, char* buffer) {
  iree_host_size_t position = 0;
  position += iree_address_write_uint(buffer + position, octets[0]);
  buffer[position++] = '.';
  position += iree_address_write_uint(buffer + position, octets[1]);
  buffer[position++] = '.';
  position += iree_address_write_uint(buffer + position, octets[2]);
  buffer[position++] = '.';
  position += iree_address_write_uint(buffer + position, octets[3]);
  buffer[position++] = ':';
  position += iree_address_write_uint(buffer + position, port);
  return position;
}

//===----------------------------------------------------------------------===//
// IPv6 formatting (RFC 5952)
//===----------------------------------------------------------------------===//

iree_host_size_t iree_async_address_format_ipv6(const uint8_t bytes[16],
                                                uint16_t port,
                                                uint32_t scope_id,
                                                char* buffer) {
  // Convert to 8 groups.
  uint16_t groups[8];
  for (int i = 0; i < 8; ++i) {
    groups[i] = (uint16_t)((uint16_t)bytes[i * 2] << 8 | bytes[i * 2 + 1]);
  }

  // RFC 5952: find longest run of consecutive zero groups (>1, leftmost wins).
  int best_start = -1;
  int best_length = 1;
  int current_start = -1;
  int current_length = 0;
  for (int i = 0; i < 8; ++i) {
    if (groups[i] == 0) {
      if (current_start < 0) {
        current_start = i;
        current_length = 1;
      } else {
        ++current_length;
      }
    } else {
      if (current_length > best_length) {
        best_start = current_start;
        best_length = current_length;
      }
      current_start = -1;
      current_length = 0;
    }
  }
  if (current_length > best_length) {
    best_start = current_start;
    best_length = current_length;
  }

  // Check for IPv4-mapped (::ffff:0:0/96 prefix).
  bool is_ipv4_mapped =
      (groups[0] == 0 && groups[1] == 0 && groups[2] == 0 && groups[3] == 0 &&
       groups[4] == 0 && groups[5] == 0xffff);

  iree_host_size_t position = 0;
  buffer[position++] = '[';

  if (is_ipv4_mapped) {
    memcpy(buffer + position, "::ffff:", 7);
    position += 7;
    position += iree_address_write_uint(buffer + position, bytes[12]);
    buffer[position++] = '.';
    position += iree_address_write_uint(buffer + position, bytes[13]);
    buffer[position++] = '.';
    position += iree_address_write_uint(buffer + position, bytes[14]);
    buffer[position++] = '.';
    position += iree_address_write_uint(buffer + position, bytes[15]);
  } else {
    for (int i = 0; i < 8; ++i) {
      if (best_start >= 0 && i == best_start) {
        memcpy(buffer + position, "::", 2);
        position += 2;
        i += best_length - 1;
        continue;
      }
      if (i > 0 && !(best_start >= 0 && i == best_start + best_length)) {
        buffer[position++] = ':';
      }
      position += iree_address_write_hex16(buffer + position, groups[i]);
    }
  }

  if (scope_id != 0) {
    buffer[position++] = '%';
    position += iree_address_write_uint(buffer + position, scope_id);
  }

  buffer[position++] = ']';
  buffer[position++] = ':';
  position += iree_address_write_uint(buffer + position, port);
  return position;
}

//===----------------------------------------------------------------------===//
// Unix path formatting
//===----------------------------------------------------------------------===//

iree_host_size_t iree_async_address_format_unix(iree_string_view_t path,
                                                bool is_abstract,
                                                char* buffer) {
  iree_host_size_t position = 0;
  memcpy(buffer + position, "unix:", 5);
  position += 5;
  if (is_abstract) {
    buffer[position++] = '@';
  }
  memcpy(buffer + position, path.data, path.size);
  position += path.size;
  return position;
}

//===----------------------------------------------------------------------===//
// String-to-address parsing
//===----------------------------------------------------------------------===//

// Parses a decimal port string into a uint16_t.
// Rejects empty strings, non-digit characters, leading zeros (except "0"
// itself), and values exceeding 65535. |full_address| is included in error
// messages for context.
static iree_status_t iree_async_address_parse_port(
    iree_string_view_t port_str, iree_string_view_t full_address,
    uint16_t* out_port) {
  if (port_str.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty port in address '%.*s'",
                            (int)full_address.size, full_address.data);
  }

  // Reject leading zeros (ambiguous; "0" alone is fine).
  if (port_str.size > 1 && port_str.data[0] == '0') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "leading zeros in port in address '%.*s'",
                            (int)full_address.size, full_address.data);
  }

  uint32_t value = 0;
  for (iree_host_size_t i = 0; i < port_str.size; ++i) {
    char character = port_str.data[i];
    if (character < '0' || character > '9') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "non-numeric port in address '%.*s'",
                              (int)full_address.size, full_address.data);
    }
    value = value * 10 + (uint32_t)(character - '0');
    if (value > 65535) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "port out of range in address '%.*s'",
                              (int)full_address.size, full_address.data);
    }
  }

  *out_port = (uint16_t)value;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_address_from_string(
    iree_string_view_t address_str, iree_async_address_t* out_address) {
  if (address_str.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty address");
  }

  // Unix: "unix:<path>"
  if (iree_string_view_starts_with(address_str,
                                   iree_make_cstring_view("unix:"))) {
    iree_string_view_t path =
        iree_string_view_substr(address_str, 5, IREE_HOST_SIZE_MAX);
    return iree_async_address_from_unix(path, out_address);
  }

  // IPv6: "[<host>]:<port>"
  if (address_str.data[0] == '[') {
    // Find closing bracket.
    iree_host_size_t close_bracket = IREE_HOST_SIZE_MAX;
    for (iree_host_size_t i = 1; i < address_str.size; ++i) {
      if (address_str.data[i] == ']') {
        close_bracket = i;
        break;
      }
    }
    if (close_bracket == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing closing ']' in address '%.*s'",
                              (int)address_str.size, address_str.data);
    }

    // Extract host (content between brackets, may include %zone).
    iree_string_view_t host =
        iree_string_view_substr(address_str, 1, close_bracket - 1);

    // Expect "]:port" after the closing bracket.
    iree_host_size_t after_bracket = close_bracket + 1;
    if (after_bracket >= address_str.size ||
        address_str.data[after_bracket] != ':') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected ':port' after ']' in address '%.*s'",
                              (int)address_str.size, address_str.data);
    }
    iree_string_view_t port_str = iree_string_view_substr(
        address_str, after_bracket + 1, IREE_HOST_SIZE_MAX);

    uint16_t port = 0;
    IREE_RETURN_IF_ERROR(
        iree_async_address_parse_port(port_str, address_str, &port));
    return iree_async_address_from_ipv6(host, port, out_address);
  }

  // IPv4: "<host>:<port>" (split at last colon).
  iree_host_size_t last_colon = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = address_str.size; i > 0; --i) {
    if (address_str.data[i - 1] == ':') {
      last_colon = i - 1;
      break;
    }
  }
  if (last_colon == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing ':port' in address '%.*s'",
                            (int)address_str.size, address_str.data);
  }

  iree_string_view_t host = iree_string_view_substr(address_str, 0, last_colon);
  iree_string_view_t port_str =
      iree_string_view_substr(address_str, last_colon + 1, IREE_HOST_SIZE_MAX);

  uint16_t port = 0;
  IREE_RETURN_IF_ERROR(
      iree_async_address_parse_port(port_str, address_str, &port));
  return iree_async_address_from_ipv4(host, port, out_address);
}
