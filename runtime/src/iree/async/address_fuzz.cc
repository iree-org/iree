// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for address parsing and formatting.
//
// Exercises all three parsers (IPv4, IPv6, Unix) with arbitrary input and
// verifies round-trip invariants: any successfully parsed address must format
// without error. Also fuzzes the formatter directly with random struct contents
// to ensure no crashes on malformed address data.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/async/address.h"
#include "iree/base/api.h"

#define FUZZ_ASSERT(condition) \
  do {                         \
    if (!(condition)) {        \
      __builtin_trap();        \
    }                          \
  } while (0)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return 0;

  // Extract port from first 2 bytes of fuzz input.
  uint16_t port;
  memcpy(&port, data, sizeof(port));
  data += 2;
  size -= 2;

  iree_string_view_t input = {(const char*)data, size};
  iree_async_address_t address;

  // Fuzz IPv4 parser + round-trip invariant.
  {
    iree_status_t status = iree_async_address_from_ipv4(input, port, &address);
    if (iree_status_is_ok(status)) {
      // Successful parse must format without error.
      char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
      iree_string_view_t formatted = iree_string_view_empty();
      iree_status_t format_status = iree_async_address_format(
          &address, sizeof(buffer), buffer, &formatted);
      FUZZ_ASSERT(iree_status_is_ok(format_status));
      FUZZ_ASSERT(formatted.size > 0);
    } else {
      iree_status_ignore(status);
    }
  }

  // Fuzz IPv6 parser + round-trip invariant.
  {
    iree_status_t status = iree_async_address_from_ipv6(input, port, &address);
    if (iree_status_is_ok(status)) {
      char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
      iree_string_view_t formatted = iree_string_view_empty();
      iree_status_t format_status = iree_async_address_format(
          &address, sizeof(buffer), buffer, &formatted);
      FUZZ_ASSERT(iree_status_is_ok(format_status));
      FUZZ_ASSERT(formatted.size > 0);
    } else {
      iree_status_ignore(status);
    }
  }

  // Fuzz Unix socket parser + round-trip invariant.
  {
    iree_status_t status = iree_async_address_from_unix(input, &address);
    if (iree_status_is_ok(status)) {
      char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
      iree_string_view_t formatted = iree_string_view_empty();
      iree_status_t format_status = iree_async_address_format(
          &address, sizeof(buffer), buffer, &formatted);
      FUZZ_ASSERT(iree_status_is_ok(format_status));
      FUZZ_ASSERT(formatted.size > 0);
    } else {
      iree_status_ignore(status);
    }
  }

  // Fuzz iree_async_address_from_string + round-trip invariant.
  // If from_string succeeds, format must succeed, and from_string on the
  // formatted output must succeed and produce the same formatted result.
  {
    iree_status_t status = iree_async_address_from_string(input, &address);
    if (iree_status_is_ok(status)) {
      char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
      iree_string_view_t formatted = iree_string_view_empty();
      iree_status_t format_status = iree_async_address_format(
          &address, sizeof(buffer), buffer, &formatted);
      FUZZ_ASSERT(iree_status_is_ok(format_status));
      FUZZ_ASSERT(formatted.size > 0);

      // Second round-trip: from_string(formatted) must also succeed and
      // produce the same formatted output (canonical form is stable).
      iree_async_address_t address2;
      iree_status_t status2 =
          iree_async_address_from_string(formatted, &address2);
      FUZZ_ASSERT(iree_status_is_ok(status2));

      char buffer2[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
      iree_string_view_t formatted2 = iree_string_view_empty();
      iree_status_t format_status2 = iree_async_address_format(
          &address2, sizeof(buffer2), buffer2, &formatted2);
      FUZZ_ASSERT(iree_status_is_ok(format_status2));
      FUZZ_ASSERT(iree_string_view_equal(formatted, formatted2));
    } else {
      iree_status_ignore(status);
    }
  }

  // Fuzz the formatter with arbitrary struct contents.
  // This exercises the family switch, length validation, and partial-struct
  // code paths that can't be reached via the normal parsing API.
  if (size >= sizeof(iree_async_address_t)) {
    memcpy(&address, data, sizeof(address));
    // Clamp length to storage bounds to avoid OOB reads in format.
    if (address.length > sizeof(address.storage)) {
      address.length = sizeof(address.storage);
    }
    char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
    iree_string_view_t formatted = iree_string_view_empty();
    iree_status_t status =
        iree_async_address_format(&address, sizeof(buffer), buffer, &formatted);
    // Random family/contents may legitimately fail — just don't crash.
    iree_status_ignore(status);
  }

  return 0;
}
