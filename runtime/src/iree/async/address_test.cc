// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/address.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Helper: format an address to a std::string for easy comparison.
static std::string FormatAddress(const iree_async_address_t& address) {
  char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status =
      iree_async_address_format(&address, sizeof(buffer), buffer, &value);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return "<format failed>";
  }
  return std::string(value.data, value.size);
}

// Helper: parse IPv4 and return the address (or fail the test).
static iree_async_address_t ParseIPv4(const char* host, uint16_t port) {
  iree_async_address_t address;
  iree_string_view_t host_view =
      host ? iree_make_cstring_view(host) : iree_string_view_empty();
  iree_status_t status =
      iree_async_address_from_ipv4(host_view, port, &address);
  IREE_EXPECT_OK(status);
  return address;
}

// Helper: parse IPv6 and return the address (or fail the test).
static iree_async_address_t ParseIPv6(const char* host, uint16_t port) {
  iree_async_address_t address;
  iree_string_view_t host_view =
      host ? iree_make_cstring_view(host) : iree_string_view_empty();
  iree_status_t status =
      iree_async_address_from_ipv6(host_view, port, &address);
  IREE_EXPECT_OK(status);
  return address;
}

//===----------------------------------------------------------------------===//
// IPv4 Parsing — Valid Cases
//===----------------------------------------------------------------------===//

TEST(AddressFromIpv4, EmptyHostIsAny) {
  iree_async_address_t address;
  IREE_EXPECT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 8080, &address));
  EXPECT_EQ(FormatAddress(address), "0.0.0.0:8080");
}

TEST(AddressFromIpv4, Loopback) {
  auto address = ParseIPv4("127.0.0.1", 80);
  EXPECT_EQ(FormatAddress(address), "127.0.0.1:80");
}

TEST(AddressFromIpv4, AllZeros) {
  auto address = ParseIPv4("0.0.0.0", 0);
  EXPECT_EQ(FormatAddress(address), "0.0.0.0:0");
}

TEST(AddressFromIpv4, MaxOctets) {
  auto address = ParseIPv4("255.255.255.255", 65535);
  EXPECT_EQ(FormatAddress(address), "255.255.255.255:65535");
}

TEST(AddressFromIpv4, EphemeralPort) {
  auto address = ParseIPv4("192.168.1.1", 0);
  EXPECT_EQ(FormatAddress(address), "192.168.1.1:0");
}

TEST(AddressFromIpv4, TypicalServerPort) {
  auto address = ParseIPv4("10.0.0.1", 443);
  EXPECT_EQ(FormatAddress(address), "10.0.0.1:443");
}

TEST(AddressFromIpv4, SingleDigitOctets) {
  auto address = ParseIPv4("1.2.3.4", 5);
  EXPECT_EQ(FormatAddress(address), "1.2.3.4:5");
}

//===----------------------------------------------------------------------===//
// IPv4 Parsing — Error Cases
//===----------------------------------------------------------------------===//

TEST(AddressFromIpv4, OctetOverflow) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("256.0.0.1"), 80, &address));
}

TEST(AddressFromIpv4, OctetOverflowLarge) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("999.0.0.1"), 80, &address));
}

TEST(AddressFromIpv4, TooFewDots) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("1.2.3"), 80, &address));
}

TEST(AddressFromIpv4, TooManyDots) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("1.2.3.4.5"), 80, &address));
}

TEST(AddressFromIpv4, LeadingZeros) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv4(iree_make_cstring_view("01.02.03.04"), 80,
                                   &address));
}

TEST(AddressFromIpv4, LeadingZeroSingle) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("1.02.3.4"), 80, &address));
}

TEST(AddressFromIpv4, TrailingGarbage) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv4(iree_make_cstring_view("1.2.3.4xyz"), 80,
                                   &address));
}

TEST(AddressFromIpv4, WhitespaceLeading) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view(" 1.2.3.4"), 80, &address));
}

TEST(AddressFromIpv4, WhitespaceTrailing) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("1.2.3.4 "), 80, &address));
}

TEST(AddressFromIpv4, EmptyOctet) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("1..3.4"), 80, &address));
}

TEST(AddressFromIpv4, HexRejected) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv4(iree_make_cstring_view("0x7f.0.0.1"), 80,
                                   &address));
}

TEST(AddressFromIpv4, JustDots) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("..."), 80, &address));
}

TEST(AddressFromIpv4, NegativeRejected) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(
                            iree_make_cstring_view("-1.0.0.0"), 80, &address));
}

TEST(AddressFromIpv4, VeryLongDigitString) {
  // Digits that overflow a uint32 accumulator (exceeds 255 check).
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv4(iree_make_cstring_view("9999999999.0.0.1"),
                                   80, &address));
}

TEST(AddressFromIpv4, EmbeddedNul) {
  // string_view with embedded NUL byte — not a valid digit.
  iree_string_view_t host = {("1.2\0.3.4"), 8};
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv4(host, 80, &address));
}

//===----------------------------------------------------------------------===//
// IPv6 Parsing — Valid Cases
//===----------------------------------------------------------------------===//

TEST(AddressFromIpv6, EmptyHostIsAny) {
  iree_async_address_t address;
  IREE_EXPECT_OK(
      iree_async_address_from_ipv6(iree_string_view_empty(), 80, &address));
  EXPECT_EQ(FormatAddress(address), "[::]:80");
}

TEST(AddressFromIpv6, Loopback) {
  auto address = ParseIPv6("::1", 80);
  EXPECT_EQ(FormatAddress(address), "[::1]:80");
}

TEST(AddressFromIpv6, FullForm) {
  auto address = ParseIPv6("2001:0db8:0000:0000:0000:0000:0000:0001", 443);
  EXPECT_EQ(FormatAddress(address), "[2001:db8::1]:443");
}

TEST(AddressFromIpv6, CompressionMiddle) {
  auto address = ParseIPv6("2001:db8::1", 80);
  EXPECT_EQ(FormatAddress(address), "[2001:db8::1]:80");
}

TEST(AddressFromIpv6, CompressionStart) {
  auto address = ParseIPv6("::1", 80);
  EXPECT_EQ(FormatAddress(address), "[::1]:80");
}

TEST(AddressFromIpv6, CompressionEnd) {
  auto address = ParseIPv6("2001:db8::", 80);
  EXPECT_EQ(FormatAddress(address), "[2001:db8::]:80");
}

TEST(AddressFromIpv6, AllZeros) {
  auto address = ParseIPv6("::", 0);
  EXPECT_EQ(FormatAddress(address), "[::]:0");
}

TEST(AddressFromIpv6, AllOnes) {
  auto address = ParseIPv6("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", 1);
  EXPECT_EQ(FormatAddress(address),
            "[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:1");
}

TEST(AddressFromIpv6, LinkLocal) {
  auto address = ParseIPv6("fe80::1", 0);
  EXPECT_EQ(FormatAddress(address), "[fe80::1]:0");
}

TEST(AddressFromIpv6, MixedCase) {
  auto address = ParseIPv6("2001:DB8::ABCD", 80);
  // Output is always lowercase per RFC 5952.
  EXPECT_EQ(FormatAddress(address), "[2001:db8::abcd]:80");
}

TEST(AddressFromIpv6, SingleZeroGroupNotCompressed) {
  // RFC 5952: single zero group is NOT compressed.
  auto address = ParseIPv6("2001:db8:0:1:0:0:0:1", 80);
  // The run of 3 zeros at positions 4-6 gets compressed, not the single at 2.
  EXPECT_EQ(FormatAddress(address), "[2001:db8:0:1::1]:80");
}

TEST(AddressFromIpv6, LongestRunWins) {
  // Two runs of zeros: positions 2-3 (length 2) and 5-7 (length 3).
  auto address = ParseIPv6("1:1:0:0:1:0:0:0", 80);
  EXPECT_EQ(FormatAddress(address), "[1:1:0:0:1::]:80");
}

TEST(AddressFromIpv6, LeftmostTiebreak) {
  // Two runs of equal length: positions 1-2 and 5-6.
  auto address = ParseIPv6("1:0:0:1:1:0:0:1", 80);
  EXPECT_EQ(FormatAddress(address), "[1::1:1:0:0:1]:80");
}

TEST(AddressFromIpv6, Ipv4Mapped) {
  auto address = ParseIPv6("::ffff:192.168.1.1", 8080);
  EXPECT_EQ(FormatAddress(address), "[::ffff:192.168.1.1]:8080");
}

TEST(AddressFromIpv6, Ipv4MappedLoopback) {
  auto address = ParseIPv6("::ffff:127.0.0.1", 80);
  EXPECT_EQ(FormatAddress(address), "[::ffff:127.0.0.1]:80");
}

TEST(AddressFromIpv6, LeadingZerosInGroup) {
  // Leading zeros in groups are valid input.
  auto address = ParseIPv6("2001:0db8:0000:0000:0000:0000:0000:0001", 80);
  EXPECT_EQ(FormatAddress(address), "[2001:db8::1]:80");
}

TEST(AddressFromIpv6, MaxPort) {
  auto address = ParseIPv6("::1", 65535);
  EXPECT_EQ(FormatAddress(address), "[::1]:65535");
}

//===----------------------------------------------------------------------===//
// IPv6 Parsing — Error Cases
//===----------------------------------------------------------------------===//

TEST(AddressFromIpv6, DoubleCompressionRejected) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("2001::db8::1"), 80,
                                   &address));
}

TEST(AddressFromIpv6, TooManyGroups) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("1:2:3:4:5:6:7:8:9"),
                                   80, &address));
}

TEST(AddressFromIpv6, TooFewGroupsNoCompression) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("1:2:3:4:5:6:7"), 80,
                                   &address));
}

TEST(AddressFromIpv6, GroupTooLong) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("2001:dbbbb::1"), 80,
                                   &address));
}

TEST(AddressFromIpv6, InvalidHexChar) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("2001:xyzw::1"), 80,
                                   &address));
}

TEST(AddressFromIpv6, TrailingColon) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("2001:db8:1:"), 80,
                                   &address));
}

TEST(AddressFromIpv6, LeadingSingleColon) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view(":1"), 80, &address));
}

TEST(AddressFromIpv6, TooManyGroupsWithCompression) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("1:2:3:4:5::6:7:8"),
                                   80, &address));
}

TEST(AddressFromIpv6, Ipv4MappedOctetOverflow) {
  // IPv4-mapped form with an octet exceeding 255.
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("::ffff:256.1.1.1"),
                                   80, &address));
}

TEST(AddressFromIpv6, Ipv4MappedTooFewOctets) {
  // IPv4-mapped form with fewer than 4 octets.
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(iree_make_cstring_view("::ffff:1.2.3"), 80,
                                   &address));
}

TEST(AddressFromIpv6, SixGroupsPlusIpv4Suffix) {
  // 6 hex groups + IPv4 suffix = 8 total 16-bit groups (valid).
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_ipv6(
      iree_make_cstring_view("1:2:3:4:5:6:1.2.3.4"), 80, &address));
}

TEST(AddressFromIpv6, CompressionPlusIpv4TooManyGroups) {
  // Compression + hex groups + IPv4 suffix would exceed 8 total groups.
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(
          iree_make_cstring_view("::1:2:3:4:5:6:1.2.3.4"), 80, &address));
}

TEST(AddressFromIpv6, ZoneIdNumeric) {
  // Numeric zone ID maps directly to sin6_scope_id.
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_ipv6(
      iree_make_cstring_view("fe80::1%3"), 80, &address));
  EXPECT_EQ(FormatAddress(address), "[fe80::1%3]:80");
}

TEST(AddressFromIpv6, ZoneIdZeroNotFormatted) {
  // No zone suffix → scope_id is 0 → no % in output.
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_ipv6(iree_make_cstring_view("fe80::1"),
                                              80, &address));
  EXPECT_EQ(FormatAddress(address), "[fe80::1]:80");
}

TEST(AddressFromIpv6, ZoneIdEmpty) {
  // Trailing % with no zone ID is invalid.
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_ipv6(
                            iree_make_cstring_view("fe80::1%"), 80, &address));
}

TEST(AddressFromIpv6, ZoneIdLargeNumeric) {
  // Large numeric scope_id.
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_ipv6(
      iree_make_cstring_view("fe80::1%42"), 80, &address));
  EXPECT_EQ(FormatAddress(address), "[fe80::1%42]:80");
}

#if defined(IREE_PLATFORM_LINUX)

TEST(AddressFromIpv6, ZoneIdInterfaceName) {
  // "lo" exists on all Linux systems — resolves to a non-zero interface index.
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_ipv6(
      iree_make_cstring_view("fe80::1%lo"), 80, &address));
  // The formatted output uses the numeric index, not the name.
  std::string formatted = FormatAddress(address);
  // Should start with [fe80::1% and end with ]:80.
  EXPECT_TRUE(formatted.find("[fe80::1%") == 0);
  EXPECT_TRUE(formatted.find("]:80") != std::string::npos);
}

TEST(AddressFromIpv6, ZoneIdUnknownInterface) {
  // Non-existent interface name (must be < IF_NAMESIZE = 16 chars).
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_async_address_from_ipv6(iree_make_cstring_view("fe80::1%noif0"), 80,
                                   &address));
}

TEST(AddressFromIpv6, ZoneIdInterfaceNameTooLong) {
  // Interface name exceeding IF_NAMESIZE (16 bytes on Linux).
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_ipv6(
          iree_make_cstring_view("fe80::1%this_name_is_way_too_long"), 80,
          &address));
}

#endif  // IREE_PLATFORM_LINUX

//===----------------------------------------------------------------------===//
// Unix Socket Parsing
//===----------------------------------------------------------------------===//

#if !defined(IREE_PLATFORM_GENERIC) && !defined(IREE_PLATFORM_EMSCRIPTEN)

TEST(AddressFromUnix, SimplePath) {
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_unix(
      iree_make_cstring_view("/tmp/socket.sock"), &address));
  EXPECT_EQ(FormatAddress(address), "unix:/tmp/socket.sock");
}

TEST(AddressFromUnix, RelativePath) {
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_unix(
      iree_make_cstring_view("./relative.sock"), &address));
  EXPECT_EQ(FormatAddress(address), "unix:./relative.sock");
}

TEST(AddressFromUnix, EmptyPathRejected) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_unix(iree_string_view_empty(), &address));
}

TEST(AddressFromUnix, PathTooLong) {
  // sun_path is typically 104 (macOS) or 108 (Linux) bytes.
  // A 200-byte path should exceed the limit on all platforms.
  std::string long_path(200, 'x');
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_unix(
          iree_make_string_view(long_path.data(), long_path.size()), &address));
}

TEST(AddressFromUnix, MaxLengthPath) {
  // Create a path that is exactly the maximum length (sizeof(sun_path) - 1).
  // We test with 103 bytes which fits on both Linux (107 max) and macOS (103
  // max).
  std::string max_path(103, 'a');
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_unix(
      iree_make_string_view(max_path.data(), max_path.size()), &address));
  EXPECT_EQ(FormatAddress(address), "unix:" + max_path);
}

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)

TEST(AddressFromUnix, AbstractNamespace) {
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_unix(
      iree_make_cstring_view("@my-abstract-socket"), &address));
  EXPECT_EQ(FormatAddress(address), "unix:@my-abstract-socket");
}

TEST(AddressFromUnix, AbstractNamespaceEmpty) {
  // "@" alone means abstract namespace with an empty name (just the NUL byte).
  iree_async_address_t address;
  IREE_EXPECT_OK(
      iree_async_address_from_unix(iree_make_cstring_view("@"), &address));
  EXPECT_EQ(FormatAddress(address), "unix:@");
}

#else  // !IREE_PLATFORM_LINUX && !IREE_PLATFORM_ANDROID

TEST(AddressFromUnix, AbstractNamespaceUnavailable) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_async_address_from_unix(
                            iree_make_cstring_view("@abstract"), &address));
}

#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

TEST(AddressFormat, UnixPathWithoutTrailingNulInLength) {
  // Construct a valid Unix address via the normal API, then simulate what
  // getsockname() returns: the socklen_t may not include the trailing NUL byte.
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_unix(
      iree_make_cstring_view("/tmp/test.sock"), &address));
  // Subtract 1 from the stored length (drop the NUL from the count).
  address.length -= 1;
  // Format should still produce the correct path.
  EXPECT_EQ(FormatAddress(address), "unix:/tmp/test.sock");
}

#endif  // !IREE_PLATFORM_GENERIC && !IREE_PLATFORM_EMSCRIPTEN

//===----------------------------------------------------------------------===//
// Format — Buffer Capacity
//===----------------------------------------------------------------------===//

TEST(AddressFormat, BufferExactSize) {
  auto address = ParseIPv4("1.2.3.4", 80);
  // "1.2.3.4:80" = 10 chars.
  char buffer[10];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
  EXPECT_EQ(std::string(value.data, value.size), "1.2.3.4:80");
}

TEST(AddressFormat, BufferTooSmall) {
  auto address = ParseIPv4("127.0.0.1", 8080);
  // "127.0.0.1:8080" = 14 chars. Provide only 5.
  char buffer[5];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
}

TEST(AddressFormat, BufferCapacityZero) {
  auto address = ParseIPv4("1.2.3.4", 80);
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_async_address_format(&address, 0, NULL, &value));
}

TEST(AddressFormat, OversizedBuffer) {
  auto address = ParseIPv4("10.0.0.1", 443);
  char buffer[256];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
  EXPECT_EQ(std::string(value.data, value.size), "10.0.0.1:443");
  // Value points into the buffer.
  EXPECT_EQ(value.data, buffer);
}

TEST(AddressFormat, OutValueNull) {
  auto address = ParseIPv4("1.2.3.4", 80);
  char buffer[64];
  // out_value is NULL — should still succeed without writing the view.
  IREE_EXPECT_OK(iree_async_address_format(&address, sizeof(buffer), buffer,
                                           /*out_value=*/NULL));
}

TEST(AddressFormat, EmptyAddress) {
  iree_async_address_t address;
  memset(&address, 0, sizeof(address));
  char buffer[64];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
}

//===----------------------------------------------------------------------===//
// Format — IPv6 Specific
//===----------------------------------------------------------------------===//

TEST(AddressFormat, Ipv6Loopback) {
  auto address = ParseIPv6("::1", 80);
  char buffer[64];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
  EXPECT_EQ(std::string(value.data, value.size), "[::1]:80");
}

TEST(AddressFormat, Ipv6Any) {
  auto address = ParseIPv6("::", 0);
  char buffer[64];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
  EXPECT_EQ(std::string(value.data, value.size), "[::]:0");
}

TEST(AddressFormat, Ipv6Mapped) {
  auto address = ParseIPv6("::ffff:10.0.0.1", 9090);
  char buffer[64];
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      iree_async_address_format(&address, sizeof(buffer), buffer, &value));
  EXPECT_EQ(std::string(value.data, value.size), "[::ffff:10.0.0.1]:9090");
}

TEST(AddressFormat, TruncatedLengthTooShort) {
  // Address with length=1 is too short for any valid sockaddr family header.
  iree_async_address_t address;
  memset(&address, 0, sizeof(address));
  address.length = 1;
  char buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t formatted = iree_string_view_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_format(&address, sizeof(buffer), buffer, &formatted));
}

//===----------------------------------------------------------------------===//
// Round-Trip Tests
//===----------------------------------------------------------------------===//

TEST(AddressRoundTrip, Ipv4Loopback) {
  auto address = ParseIPv4("127.0.0.1", 8080);
  EXPECT_EQ(FormatAddress(address), "127.0.0.1:8080");
}

TEST(AddressRoundTrip, Ipv4Any) {
  iree_async_address_t address;
  IREE_EXPECT_OK(
      iree_async_address_from_ipv4(iree_string_view_empty(), 0, &address));
  EXPECT_EQ(FormatAddress(address), "0.0.0.0:0");
}

TEST(AddressRoundTrip, Ipv6Compressed) {
  auto address = ParseIPv6("2001:db8::1", 443);
  EXPECT_EQ(FormatAddress(address), "[2001:db8::1]:443");
}

TEST(AddressRoundTrip, Ipv6Full) {
  auto address = ParseIPv6("2001:db8:85a3:0:0:8a2e:370:7334", 80);
  EXPECT_EQ(FormatAddress(address), "[2001:db8:85a3::8a2e:370:7334]:80");
}

TEST(AddressRoundTrip, Ipv6AllOnes) {
  auto address = ParseIPv6("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", 1);
  EXPECT_EQ(FormatAddress(address),
            "[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:1");
}

#if !defined(IREE_PLATFORM_GENERIC) && !defined(IREE_PLATFORM_EMSCRIPTEN)
TEST(AddressRoundTrip, UnixFilesystem) {
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_unix(
      iree_make_cstring_view("/var/run/daemon.sock"), &address));
  EXPECT_EQ(FormatAddress(address), "unix:/var/run/daemon.sock");
}
#endif  // !IREE_PLATFORM_GENERIC && !IREE_PLATFORM_EMSCRIPTEN

//===----------------------------------------------------------------------===//
// AddressFromString — Valid IPv4
//===----------------------------------------------------------------------===//

// Helper: parse from string and return the address (or fail the test).
static iree_async_address_t ParseFromString(const char* address_str) {
  iree_async_address_t address;
  IREE_EXPECT_OK(iree_async_address_from_string(
      iree_make_cstring_view(address_str), &address));
  return address;
}

TEST(AddressFromString, Ipv4Loopback) {
  auto address = ParseFromString("127.0.0.1:80");
  EXPECT_EQ(FormatAddress(address), "127.0.0.1:80");
}

TEST(AddressFromString, Ipv4AllZeros) {
  auto address = ParseFromString("0.0.0.0:0");
  EXPECT_EQ(FormatAddress(address), "0.0.0.0:0");
}

TEST(AddressFromString, Ipv4MaxValues) {
  auto address = ParseFromString("255.255.255.255:65535");
  EXPECT_EQ(FormatAddress(address), "255.255.255.255:65535");
}

TEST(AddressFromString, Ipv4EmptyHostIsAny) {
  auto address = ParseFromString(":8080");
  EXPECT_EQ(FormatAddress(address), "0.0.0.0:8080");
}

TEST(AddressFromString, Ipv4TypicalServer) {
  auto address = ParseFromString("10.0.0.1:443");
  EXPECT_EQ(FormatAddress(address), "10.0.0.1:443");
}

TEST(AddressFromString, Ipv4EphemeralPort) {
  auto address = ParseFromString("192.168.1.1:0");
  EXPECT_EQ(FormatAddress(address), "192.168.1.1:0");
}

//===----------------------------------------------------------------------===//
// AddressFromString — Valid IPv6
//===----------------------------------------------------------------------===//

TEST(AddressFromString, Ipv6Loopback) {
  auto address = ParseFromString("[::1]:80");
  EXPECT_EQ(FormatAddress(address), "[::1]:80");
}

TEST(AddressFromString, Ipv6Any) {
  auto address = ParseFromString("[::]:0");
  EXPECT_EQ(FormatAddress(address), "[::]:0");
}

TEST(AddressFromString, Ipv6Compressed) {
  auto address = ParseFromString("[2001:db8::1]:443");
  EXPECT_EQ(FormatAddress(address), "[2001:db8::1]:443");
}

TEST(AddressFromString, Ipv6Ipv4Mapped) {
  auto address = ParseFromString("[::ffff:192.168.1.1]:8080");
  EXPECT_EQ(FormatAddress(address), "[::ffff:192.168.1.1]:8080");
}

TEST(AddressFromString, Ipv6ZoneId) {
  auto address = ParseFromString("[fe80::1%3]:80");
  EXPECT_EQ(FormatAddress(address), "[fe80::1%3]:80");
}

TEST(AddressFromString, Ipv6NonCanonicalInput) {
  // Leading zeros in groups are accepted but not preserved in output.
  auto address = ParseFromString("[2001:0db8::1]:80");
  EXPECT_EQ(FormatAddress(address), "[2001:db8::1]:80");
}

TEST(AddressFromString, Ipv6Full) {
  auto address = ParseFromString("[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:1");
  EXPECT_EQ(FormatAddress(address),
            "[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:1");
}

TEST(AddressFromString, Ipv6MaxPort) {
  auto address = ParseFromString("[::1]:65535");
  EXPECT_EQ(FormatAddress(address), "[::1]:65535");
}

TEST(AddressFromString, Ipv6EmptyBrackets) {
  // Empty brackets with port — from_ipv6 treats empty host as in6addr_any.
  auto address = ParseFromString("[]:0");
  EXPECT_EQ(FormatAddress(address), "[::]:0");
}

//===----------------------------------------------------------------------===//
// AddressFromString — Valid Unix
//===----------------------------------------------------------------------===//

#if !defined(IREE_PLATFORM_GENERIC) && !defined(IREE_PLATFORM_EMSCRIPTEN)

TEST(AddressFromString, UnixFilesystem) {
  auto address = ParseFromString("unix:/tmp/socket.sock");
  EXPECT_EQ(FormatAddress(address), "unix:/tmp/socket.sock");
}

TEST(AddressFromString, UnixRelativePath) {
  auto address = ParseFromString("unix:./relative.sock");
  EXPECT_EQ(FormatAddress(address), "unix:./relative.sock");
}

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
TEST(AddressFromString, UnixAbstract) {
  auto address = ParseFromString("unix:@my-socket");
  EXPECT_EQ(FormatAddress(address), "unix:@my-socket");
}
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID

#endif  // !IREE_PLATFORM_GENERIC && !IREE_PLATFORM_EMSCRIPTEN

//===----------------------------------------------------------------------===//
// AddressFromString — Error Cases
//===----------------------------------------------------------------------===//

TEST(AddressFromString, Empty) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_string(iree_string_view_empty(), &address));
}

TEST(AddressFromString, Ipv4MissingPort) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("127.0.0.1"), &address));
}

TEST(AddressFromString, Ipv4EmptyPort) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("127.0.0.1:"), &address));
}

TEST(AddressFromString, Ipv4NonNumericPort) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("127.0.0.1:abc"), &address));
}

TEST(AddressFromString, Ipv4PortOverflow) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_string(iree_make_cstring_view("127.0.0.1:99999"),
                                     &address));
}

TEST(AddressFromString, Ipv4PortLeadingZeros) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("127.0.0.1:080"), &address));
}

TEST(AddressFromString, Ipv4PortLeadingZeroDouble) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("127.0.0.1:00"), &address));
}

TEST(AddressFromString, NoColonAtAll) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("just_a_word"), &address));
}

TEST(AddressFromString, Ipv6MissingCloseBracket) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_string(iree_make_cstring_view("[::1"), &address));
}

TEST(AddressFromString, Ipv6NoPortAfterBracket) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("[::1]"), &address));
}

TEST(AddressFromString, Ipv6MissingColonAfterBracket) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("[::1]80"), &address));
}

TEST(AddressFromString, Ipv6EmptyPort) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("[::1]:"), &address));
}

TEST(AddressFromString, Ipv6NonNumericPort) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("[::1]:abc"), &address));
}

TEST(AddressFromString, Ipv6PortOverflow) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("[::1]:99999"), &address));
}

TEST(AddressFromString, Ipv6PortLeadingZeros) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("[::1]:080"), &address));
}

#if !defined(IREE_PLATFORM_GENERIC) && !defined(IREE_PLATFORM_EMSCRIPTEN)
TEST(AddressFromString, UnixEmptyPath) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("unix:"), &address));
}
#endif  // !IREE_PLATFORM_GENERIC && !IREE_PLATFORM_EMSCRIPTEN

TEST(AddressFromString, Ipv4BadHost) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_async_address_from_string(
                            iree_make_cstring_view("1.2.3.4.5:80"), &address));
}

TEST(AddressFromString, Ipv6BadHost) {
  iree_async_address_t address;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_address_from_string(
          iree_make_cstring_view("[2001::db8::1]:80"), &address));
}

//===----------------------------------------------------------------------===//
// AddressFromString — Round-Trip Tests
//===----------------------------------------------------------------------===//

// Helper: parse → format → parse → format, verify the two formatted strings
// match. This validates that the canonical form is stable.
static void VerifyRoundTrip(const char* input) {
  iree_async_address_t address1;
  IREE_ASSERT_OK(
      iree_async_address_from_string(iree_make_cstring_view(input), &address1));
  std::string formatted1 = FormatAddress(address1);
  ASSERT_NE(formatted1, "<format failed>") << "First format failed";

  iree_async_address_t address2;
  IREE_ASSERT_OK(iree_async_address_from_string(
      iree_make_string_view(formatted1.data(), formatted1.size()), &address2));
  std::string formatted2 = FormatAddress(address2);
  EXPECT_EQ(formatted1, formatted2)
      << "Round-trip failed for input '" << input << "'";
}

TEST(AddressFromStringRoundTrip, Ipv4Loopback) {
  VerifyRoundTrip("127.0.0.1:8080");
}

TEST(AddressFromStringRoundTrip, Ipv4Any) { VerifyRoundTrip(":0"); }

TEST(AddressFromStringRoundTrip, Ipv4Max) {
  VerifyRoundTrip("255.255.255.255:65535");
}

TEST(AddressFromStringRoundTrip, Ipv6Loopback) { VerifyRoundTrip("[::1]:80"); }

TEST(AddressFromStringRoundTrip, Ipv6Compressed) {
  VerifyRoundTrip("[2001:db8::1]:443");
}

TEST(AddressFromStringRoundTrip, Ipv6Full) {
  VerifyRoundTrip("[ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff]:1");
}

TEST(AddressFromStringRoundTrip, Ipv6Mapped) {
  VerifyRoundTrip("[::ffff:10.0.0.1]:9090");
}

TEST(AddressFromStringRoundTrip, Ipv6ZoneId) {
  VerifyRoundTrip("[fe80::1%42]:80");
}

#if !defined(IREE_PLATFORM_GENERIC) && !defined(IREE_PLATFORM_EMSCRIPTEN)
TEST(AddressFromStringRoundTrip, UnixPath) {
  VerifyRoundTrip("unix:/var/run/daemon.sock");
}

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
TEST(AddressFromStringRoundTrip, UnixAbstract) {
  VerifyRoundTrip("unix:@my-abstract-socket");
}
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_ANDROID
#endif  // !IREE_PLATFORM_GENERIC && !IREE_PLATFORM_EMSCRIPTEN

}  // namespace
