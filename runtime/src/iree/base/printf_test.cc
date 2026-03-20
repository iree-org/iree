// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

// Helper: format with iree_vsnprintf and return as std::string.
// Uses va_list to avoid -Wformat-security on non-literal format strings.
static std::string FormatV(const char* format, ...) {
  va_list args1, args2;
  va_start(args1, format);
  va_copy(args2, args1);
  // First call to measure size.
  int length = iree_vsnprintf(nullptr, 0, format, args1);
  va_end(args1);
  EXPECT_GE(length, 0) << "iree_vsnprintf returned error for: " << format;
  if (length < 0) {
    va_end(args2);
    return "<error>";
  }
  std::string result(length, '\0');
  iree_vsnprintf(result.data(), length + 1, format, args2);
  va_end(args2);
  return result;
}

// Convenience wrapper for test readability.
#define Format(...) FormatV(__VA_ARGS__)

// Helper: format with iree_vsnprintf into a small buffer and verify truncation.
static std::string FormatTruncatedV(size_t buffer_size, const char* format,
                                    ...) {
  std::string buffer(buffer_size, 'X');
  va_list args;
  va_start(args, format);
  iree_vsnprintf(buffer.data(), buffer_size, format, args);
  va_end(args);
  // Return only what was written (up to NUL).
  return std::string(buffer.c_str());
}

#define FormatTruncated(buffer_size, ...) \
  FormatTruncatedV(buffer_size, __VA_ARGS__)

// Helper: format with iree_vfctprintf via callback and return as std::string.
static void format_callback_append(char c, void* user_data) {
  static_cast<std::string*>(user_data)->push_back(c);
}
static std::string FormatCallbackV(const char* format, ...) {
  std::string result;
  va_list args;
  va_start(args, format);
  int length = iree_vfctprintf(format_callback_append, &result, format, args);
  va_end(args);
  EXPECT_EQ(length, static_cast<int>(result.size()));
  return result;
}

#define FormatCallback(...) FormatCallbackV(__VA_ARGS__)

// Helper: compare our output against libc snprintf for differential testing.
static void ExpectMatchesLibcV(const char* format, ...) {
  char libc_buffer[1024] = {};
  char iree_buffer[1024] = {};
  va_list args1, args2;
  va_start(args1, format);
  va_copy(args2, args1);
  int libc_length = vsnprintf(libc_buffer, sizeof(libc_buffer), format, args1);
  va_end(args1);
  iree_vsnprintf(iree_buffer, sizeof(iree_buffer), format, args2);
  va_end(args2);
  EXPECT_STREQ(iree_buffer, libc_buffer)
      << "Format: \"" << format << "\" (libc_length=" << libc_length << ")";
}

#define ExpectMatchesLibc(...) ExpectMatchesLibcV(__VA_ARGS__)

//===----------------------------------------------------------------------===//
// Literal and escape tests
//===----------------------------------------------------------------------===//

TEST(Printf, LiteralText) {
  EXPECT_EQ(Format("hello"), "hello");
  EXPECT_EQ(Format(""), "");
  EXPECT_EQ(Format("a"), "a");
}

TEST(Printf, PercentEscape) {
  EXPECT_EQ(Format("%%"), "%");
  EXPECT_EQ(Format("100%%"), "100%");
  EXPECT_EQ(Format("%%d"), "%d");
}

//===----------------------------------------------------------------------===//
// Integer formatting: %d, %i
//===----------------------------------------------------------------------===//

TEST(Printf, SignedDecimalBasic) {
  EXPECT_EQ(Format("%d", 0), "0");
  EXPECT_EQ(Format("%d", 1), "1");
  EXPECT_EQ(Format("%d", -1), "-1");
  EXPECT_EQ(Format("%d", 42), "42");
  EXPECT_EQ(Format("%d", -42), "-42");
  EXPECT_EQ(Format("%i", 42), "42");
}

TEST(Printf, SignedDecimalBoundaries) {
  EXPECT_EQ(Format("%d", INT_MAX), std::to_string(INT_MAX));
  EXPECT_EQ(Format("%d", INT_MIN), std::to_string(INT_MIN));
}

TEST(Printf, SignedDecimal64Bit) {
  EXPECT_EQ(Format("%lld", (long long)INT64_MAX), std::to_string(INT64_MAX));
  EXPECT_EQ(Format("%lld", (long long)INT64_MIN), std::to_string(INT64_MIN));
}

TEST(Printf, SignedDecimalFlags) {
  // '+' flag: show sign for positive.
  EXPECT_EQ(Format("%+d", 42), "+42");
  EXPECT_EQ(Format("%+d", -42), "-42");
  EXPECT_EQ(Format("%+d", 0), "+0");

  // ' ' flag: space before positive.
  EXPECT_EQ(Format("% d", 42), " 42");
  EXPECT_EQ(Format("% d", -42), "-42");

  // '+' overrides ' '.
  EXPECT_EQ(Format("%+ d", 42), "+42");
}

TEST(Printf, SignedDecimalWidth) {
  EXPECT_EQ(Format("%5d", 42), "   42");
  EXPECT_EQ(Format("%5d", -42), "  -42");
  EXPECT_EQ(Format("%-5d", 42), "42   ");
  EXPECT_EQ(Format("%05d", 42), "00042");
  EXPECT_EQ(Format("%05d", -42), "-0042");
}

TEST(Printf, SignedDecimalPrecision) {
  EXPECT_EQ(Format("%.5d", 42), "00042");
  EXPECT_EQ(Format("%.5d", -42), "-00042");
  EXPECT_EQ(Format("%.0d", 0), "");
  EXPECT_EQ(Format("%.0d", 1), "1");
  // Precision overrides zero-pad flag.
  EXPECT_EQ(Format("%08.5d", 42), "   00042");
}

TEST(Printf, SignedDecimalDynamicWidth) {
  EXPECT_EQ(Format("%*d", 5, 42), "   42");
  EXPECT_EQ(Format("%*d", -5, 42), "42   ");  // Negative = left-justify.
}

TEST(Printf, SignedDecimalDynamicPrecision) {
  EXPECT_EQ(Format("%.*d", 5, 42), "00042");
  EXPECT_EQ(Format("%.*d", -1, 42), "42");  // Negative = omit precision.
}

//===----------------------------------------------------------------------===//
// Integer formatting: %u, %x, %X, %o
//===----------------------------------------------------------------------===//

TEST(Printf, UnsignedDecimal) {
  EXPECT_EQ(Format("%u", 0u), "0");
  EXPECT_EQ(Format("%u", 42u), "42");
  EXPECT_EQ(Format("%u", UINT_MAX), std::to_string(UINT_MAX));
  EXPECT_EQ(Format("%llu", (unsigned long long)UINT64_MAX),
            std::to_string(UINT64_MAX));
}

TEST(Printf, Hex) {
  EXPECT_EQ(Format("%x", 0u), "0");
  EXPECT_EQ(Format("%x", 255u), "ff");
  EXPECT_EQ(Format("%X", 255u), "FF");
  EXPECT_EQ(Format("%08x", 255u), "000000ff");
  EXPECT_EQ(Format("%08X", 255u), "000000FF");
  EXPECT_EQ(Format("%#x", 255u), "0xff");
  EXPECT_EQ(Format("%#X", 255u), "0XFF");
  EXPECT_EQ(Format("%#x", 0u), "0");  // No prefix for zero.
}

TEST(Printf, Octal) {
  EXPECT_EQ(Format("%o", 0u), "0");
  EXPECT_EQ(Format("%o", 8u), "10");
  EXPECT_EQ(Format("%#o", 8u), "010");
  EXPECT_EQ(Format("%#o", 0u), "0");
  // C99 §7.19.6.1p6: "#" with precision 0 and value 0 must still print "0".
  EXPECT_EQ(Format("%#.0o", 0u), "0");
  EXPECT_EQ(Format("%.0o", 0u),
            "");  // Without #: empty for precision 0, value 0.
  EXPECT_EQ(Format("%#.5o", 8u), "00010");
  EXPECT_EQ(Format("%#10.0o", 0u), "         0");
}

//===----------------------------------------------------------------------===//
// Length modifiers
//===----------------------------------------------------------------------===//

TEST(Printf, LengthHH) {
  EXPECT_EQ(Format("%hhd", (int)(signed char)-1), "-1");
  EXPECT_EQ(Format("%hhu", (unsigned int)(unsigned char)255), "255");
  // Truncation: 256 wraps to 0 for unsigned char.
  EXPECT_EQ(Format("%hhu", (unsigned int)256), "0");
}

TEST(Printf, LengthH) {
  EXPECT_EQ(Format("%hd", (int)(short)-1), "-1");
  EXPECT_EQ(Format("%hu", (unsigned int)(unsigned short)65535), "65535");
}

TEST(Printf, LengthL) {
  EXPECT_EQ(Format("%ld", (long)42), "42");
  EXPECT_EQ(Format("%lu", (unsigned long)42), "42");
}

TEST(Printf, LengthLL) {
  EXPECT_EQ(Format("%lld", (long long)42), "42");
  EXPECT_EQ(Format("%llu", (unsigned long long)42), "42");
  EXPECT_EQ(Format("%llx", (unsigned long long)0xDEADBEEFull), "deadbeef");
}

TEST(Printf, LengthZ) {
  EXPECT_EQ(Format("%zu", (size_t)42), "42");
  EXPECT_EQ(Format("%zu", (size_t)0), "0");
  EXPECT_EQ(Format("%zd", (size_t)42), "42");
}

TEST(Printf, LengthT) {
  EXPECT_EQ(Format("%td", (ptrdiff_t)42), "42");
  EXPECT_EQ(Format("%td", (ptrdiff_t)-42), "-42");
}

TEST(Printf, LengthJ) {
  EXPECT_EQ(Format("%jd", (intmax_t)42), "42");
  EXPECT_EQ(Format("%ju", (uintmax_t)42), "42");
}

//===----------------------------------------------------------------------===//
// String formatting: %s
//===----------------------------------------------------------------------===//

TEST(Printf, StringBasic) {
  EXPECT_EQ(Format("%s", "hello"), "hello");
  EXPECT_EQ(Format("%s", ""), "");
}

TEST(Printf, StringNull) {
  EXPECT_EQ(Format("%s", (const char*)nullptr), "(null)");
}

TEST(Printf, StringPrecision) {
  EXPECT_EQ(Format("%.3s", "hello"), "hel");
  EXPECT_EQ(Format("%.10s", "hello"), "hello");
  EXPECT_EQ(Format("%.0s", "hello"), "");
}

TEST(Printf, StringWidth) {
  EXPECT_EQ(Format("%10s", "hello"), "     hello");
  EXPECT_EQ(Format("%-10s", "hello"), "hello     ");
}

TEST(Printf, StringWidthAndPrecision) {
  EXPECT_EQ(Format("%10.3s", "hello"), "       hel");
  EXPECT_EQ(Format("%-10.3s", "hello"), "hel       ");
}

TEST(Printf, StringDynamicPrecision) {
  EXPECT_EQ(Format("%.*s", 3, "hello"), "hel");
  EXPECT_EQ(Format("%.*s", 0, "hello"), "");
  EXPECT_EQ(Format("%.*s", -1, "hello"), "hello");  // Negative = no limit.
}

TEST(Printf, StringDynamicWidthAndPrecision) {
  EXPECT_EQ(Format("%*.*s", 10, 3, "hello"), "       hel");
}

// The security-critical test: precision must bound reads, not just output.
TEST(Printf, StringPrecisionBoundsReads) {
  // Create a string of exactly 5 characters with no NUL terminator.
  // ASAN will catch any over-read.
  char buffer[5] = {'h', 'e', 'l', 'l', 'o'};
  EXPECT_EQ(Format("%.*s", 5, buffer), "hello");
  EXPECT_EQ(Format("%.*s", 3, buffer), "hel");
  EXPECT_EQ(Format("%.*s", 0, buffer), "");
}

//===----------------------------------------------------------------------===//
// Character formatting: %c
//===----------------------------------------------------------------------===//

TEST(Printf, CharBasic) {
  EXPECT_EQ(Format("%c", 'A'), "A");
  EXPECT_EQ(Format("%c", ' '), " ");
  EXPECT_EQ(Format("%c", 0), std::string(1, '\0'));
}

TEST(Printf, CharWidth) {
  EXPECT_EQ(Format("%5c", 'A'), "    A");
  EXPECT_EQ(Format("%-5c", 'A'), "A    ");
}

//===----------------------------------------------------------------------===//
// Pointer formatting: %p
//===----------------------------------------------------------------------===//

TEST(Printf, PointerNull) { EXPECT_EQ(Format("%p", (void*)nullptr), "0x0"); }

TEST(Printf, PointerNonNull) {
  // Use a known address for predictable output.
  std::string result = Format("%p", (void*)0xDEAD);
  EXPECT_EQ(result, "0xdead");
}

TEST(Printf, PointerWidth) {
  EXPECT_EQ(Format("%20p", (void*)0x1), "                 0x1");
  EXPECT_EQ(Format("%-20p", (void*)0x1), "0x1                 ");
}

//===----------------------------------------------------------------------===//
// Floating-point formatting: %f
//===----------------------------------------------------------------------===//

TEST(Printf, FloatFixedBasic) {
  EXPECT_EQ(Format("%f", 0.0), "0.000000");
  EXPECT_EQ(Format("%f", 1.0), "1.000000");
  EXPECT_EQ(Format("%f", -1.0), "-1.000000");
  EXPECT_EQ(Format("%f", 3.14159), "3.141590");
}

TEST(Printf, FloatFixedPrecision) {
  EXPECT_EQ(Format("%.0f", 3.14), "3");
  EXPECT_EQ(Format("%.1f", 3.14), "3.1");
  EXPECT_EQ(Format("%.2f", 3.14), "3.14");
  EXPECT_EQ(Format("%.3f", 3.14), "3.140");
}

TEST(Printf, FloatFixedRounding) {
  EXPECT_EQ(Format("%.0f", 0.5), "0");     // Banker's rounding: 0.5 → 0.
  EXPECT_EQ(Format("%.0f", 1.5), "2");     // Banker's rounding: 1.5 → 2.
  EXPECT_EQ(Format("%.0f", 2.5), "2");     // Banker's rounding: 2.5 → 2.
  EXPECT_EQ(Format("%.1f", 1.25), "1.2");  // 1.25 → 1.2 (round to even).
}

TEST(Printf, FloatFixedWidth) {
  EXPECT_EQ(Format("%10.2f", 3.14), "      3.14");
  EXPECT_EQ(Format("%-10.2f", 3.14), "3.14      ");
  EXPECT_EQ(Format("%010.2f", 3.14), "0000003.14");
}

TEST(Printf, FloatFixedSign) {
  EXPECT_EQ(Format("%+f", 3.14), "+3.140000");
  EXPECT_EQ(Format("%+f", -3.14), "-3.140000");
  EXPECT_EQ(Format("% f", 3.14), " 3.140000");
}

TEST(Printf, FloatFixedHash) {
  // '#' forces decimal point even with precision 0.
  EXPECT_EQ(Format("%#.0f", 3.0), "3.");
}

//===----------------------------------------------------------------------===//
// Floating-point formatting: %e, %E
//===----------------------------------------------------------------------===//

TEST(Printf, FloatExponentialBasic) {
  EXPECT_EQ(Format("%e", 0.0), "0.000000e+00");
  EXPECT_EQ(Format("%e", 1.0), "1.000000e+00");
  EXPECT_EQ(Format("%e", 100.0), "1.000000e+02");
  EXPECT_EQ(Format("%e", 0.001), "1.000000e-03");
}

TEST(Printf, FloatExponentialUppercase) {
  EXPECT_EQ(Format("%E", 1.0), "1.000000E+00");
}

TEST(Printf, FloatExponentialPrecision) {
  EXPECT_EQ(Format("%.0e", 1.0), "1e+00");
  EXPECT_EQ(Format("%.2e", 1.0), "1.00e+00");
}

//===----------------------------------------------------------------------===//
// Floating-point formatting: %g, %G
//===----------------------------------------------------------------------===//

TEST(Printf, FloatGeneralBasic) {
  EXPECT_EQ(Format("%g", 0.0), "0");
  EXPECT_EQ(Format("%g", 1.0), "1");
  EXPECT_EQ(Format("%g", 100.0), "100");
  EXPECT_EQ(Format("%g", 100000.0), "100000");
  EXPECT_EQ(Format("%g", 1000000.0), "1e+06");
}

TEST(Printf, FloatGeneralSmallValues) {
  EXPECT_EQ(Format("%g", 0.0001), "0.0001");
  EXPECT_EQ(Format("%g", 0.00001), "1e-05");
}

TEST(Printf, FloatGeneralTrailingZeros) {
  // %g strips trailing zeros.
  EXPECT_EQ(Format("%g", 1.5), "1.5");
  EXPECT_EQ(Format("%g", 1.50), "1.5");
}

TEST(Printf, FloatGeneralHash) {
  // '#' flag preserves trailing zeros.
  EXPECT_EQ(Format("%#g", 1.0), "1.00000");
}

TEST(Printf, FloatGeneralUppercase) {
  EXPECT_EQ(Format("%G", 1000000.0), "1E+06");
}

//===----------------------------------------------------------------------===//
// Floating-point special values
//===----------------------------------------------------------------------===//

TEST(Printf, FloatNaN) {
  EXPECT_EQ(Format("%f", NAN), "nan");
  EXPECT_EQ(Format("%F", NAN), "NAN");
  EXPECT_EQ(Format("%e", NAN), "nan");
  EXPECT_EQ(Format("%g", NAN), "nan");
  EXPECT_EQ(Format("%10f", NAN), "       nan");
}

TEST(Printf, FloatInf) {
  EXPECT_EQ(Format("%f", INFINITY), "inf");
  EXPECT_EQ(Format("%f", -INFINITY), "-inf");
  EXPECT_EQ(Format("%F", INFINITY), "INF");
  EXPECT_EQ(Format("%+f", INFINITY), "+inf");
}

TEST(Printf, FloatNegativeZero) {
  EXPECT_EQ(Format("%f", -0.0), "-0.000000");
  EXPECT_EQ(Format("%e", -0.0), "-0.000000e+00");
  EXPECT_EQ(Format("%g", -0.0), "-0");
}

TEST(Printf, FloatSubnormals) {
  // Subnormals have biased exponent 0 and require special handling in the
  // log10 approximation and scaling paths (scale factors that would overflow
  // double range). Test a range of subnormal values.
  ExpectMatchesLibc("%e", 5e-324);    // Smallest subnormal (DBL_TRUE_MIN).
  ExpectMatchesLibc("%e", 1e-320);    // Small subnormal.
  ExpectMatchesLibc("%e", 1e-310);    // Near the normal/subnormal boundary.
  ExpectMatchesLibc("%e", 2.2e-308);  // Just below DBL_MIN.
  ExpectMatchesLibc("%g", 5e-324);
  ExpectMatchesLibc("%f", 5e-324);
}

TEST(Printf, FloatExtremeValues) {
  // Values near the limits of double range.
  ExpectMatchesLibc("%e", 1e300);
  ExpectMatchesLibc("%e", 1e308);
  ExpectMatchesLibc("%e", DBL_MAX);  // ~1.798e+308.
  ExpectMatchesLibc("%g", DBL_MAX);
  ExpectMatchesLibc("%e", DBL_MIN);  // Smallest normal: ~2.225e-308.
  ExpectMatchesLibc("%g", DBL_MIN);
}

TEST(Printf, FloatRoundingBoundary) {
  // Values that previously triggered rounding bugs at the boundary between
  // round-up and round-down, discovered by fuzzing.
  ExpectMatchesLibc("%e", 587826050.0);  // Division-based normalization.
  ExpectMatchesLibc("%e", -587826050.0);
  ExpectMatchesLibc("%e", 575537150.0);  // Same class of bug.
  ExpectMatchesLibc("%e", -575537150.0);
  ExpectMatchesLibc("%e", 127922.5625);  // Log10 off by 1 + truncation.

  // %g routing must use the post-rounding exponent. 9.5 rounds to 10 with
  // %.1g (1 sig digit), bumping the exponent from 0 to 1, which triggers %e.
  // 0.95 (actually 0.94999...) rounds to 0.9, NOT 1.0 — exponent stays at -1.
  ExpectMatchesLibc("%.1g", 9.5);
  ExpectMatchesLibc("%.1g", 0.95);
  ExpectMatchesLibc("%.2g", 9.95);
  ExpectMatchesLibc("%.2g", 99.5);
  ExpectMatchesLibc("%.3g", 999.5);
  ExpectMatchesLibc("%.1g", 0.0095);
}

TEST(Printf, FloatLargePrecision) {
  // Precisions beyond 17 significant digits produce trailing zeros. These
  // previously caused a stack buffer overflow when the zeros were written
  // into the fixed-size formatting buffer. Verify they are now streamed
  // directly to the output without overflow.

  // %e with large precision: 1 digit + '.' + 400 zeros + 'e+00' = 406 chars.
  std::string result = Format("%.400e", 1.0);
  EXPECT_EQ(result.substr(0, 4), "1.00");
  EXPECT_EQ(result.substr(result.size() - 4), "e+00");
  EXPECT_EQ((int)result.size(), 406);

  // %f with large precision on a large value. DBL_MAX has ~309 integral
  // digits + '.' + 100 fractional digits = ~410 chars.
  result = Format("%.100f", 1e200);
  EXPECT_GT((int)result.size(), 200);
  // Should end with many zeros (digits beyond 17 significant).
  EXPECT_EQ(result.substr(result.size() - 5), "00000");

  // %f with large precision on a small value.
  result = Format("%.50f", 0.1);
  EXPECT_EQ(result[0], '0');
  EXPECT_EQ(result[1], '.');
  EXPECT_EQ((int)result.size(), 52);  // "0." + 50 digits.

  // %f with extended precision on very small fractions. The fractional digit
  // path extends effective_precision beyond 17 for small values where leading
  // zeros don't consume significant digits. Exercises the full 19-digit path.
  result = Format("%.19f", 1e-18);
  EXPECT_EQ(result.substr(0, 2), "0.");
  EXPECT_EQ((int)result.size(), 21);  // "0." + 19 digits.

  result = Format("%.18f", 1e-18);
  EXPECT_EQ(result.substr(0, 2), "0.");
  EXPECT_EQ((int)result.size(), 20);  // "0." + 18 digits.

  // Dry-run (NULL buffer) should return correct length even for huge output.
  int dry_run_length = iree_snprintf(nullptr, 0, "%.500e", 1.0);
  EXPECT_EQ(dry_run_length, 506);  // 1 + '.' + 500 + 'e+00' = 506.
}

//===----------------------------------------------------------------------===//
// Multiple specifiers in one format string
//===----------------------------------------------------------------------===//

TEST(Printf, MultipleSpecifiers) {
  EXPECT_EQ(Format("%d + %d = %d", 1, 2, 3), "1 + 2 = 3");
  EXPECT_EQ(Format("%s: %d", "count", 42), "count: 42");
  EXPECT_EQ(Format("[%08X]", 255u), "[000000FF]");
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// Helper for testing intentionally-invalid format strings without triggering
// -Wformat warnings. Uses iree_vsnprintf to bypass format checking.
static int format_invalid(char* buffer, size_t count, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int result = iree_vsnprintf(buffer, count, format, args);
  va_end(args);
  return result;
}

TEST(Printf, UnknownSpecifier) {
  char buffer[32];
  EXPECT_EQ(format_invalid(buffer, sizeof(buffer), "%q", 42), -1);
}

TEST(Printf, TruncatedFormat) {
  char buffer[32];
  // Format string ending with '%' and no specifier.
  EXPECT_EQ(format_invalid(buffer, sizeof(buffer), "hello%"), -1);
}

//===----------------------------------------------------------------------===//
// Buffer truncation
//===----------------------------------------------------------------------===//

TEST(Printf, Truncation) {
  // Full output is "hello" (5 chars).
  EXPECT_EQ(FormatTruncated(6, "%s", "hello"), "hello");  // Exact fit.
  EXPECT_EQ(FormatTruncated(4, "%s", "hello"), "hel");    // Truncated.
  EXPECT_EQ(FormatTruncated(1, "%s", "hello"), "");       // Only NUL.

  // Return value is still the full length.
  char buffer[4];
  int result = iree_snprintf(buffer, sizeof(buffer), "%s", "hello");
  EXPECT_EQ(result, 5);                   // Would need 5 chars.
  EXPECT_EQ(std::string(buffer), "hel");  // But only wrote 3 + NUL.
}

TEST(Printf, DryRun) {
  // NULL buffer: measure only.
  int result = iree_snprintf(nullptr, 0, "hello %d", 42);
  EXPECT_EQ(result, 8);  // "hello 42" = 8 chars.
}

TEST(Printf, ZeroSizeBuffer) {
  char buffer = 'X';
  int result = iree_snprintf(&buffer, 0, "hello");
  EXPECT_EQ(result, 5);
  EXPECT_EQ(buffer, 'X');  // Should not write anything.
}

//===----------------------------------------------------------------------===//
// Callback mode (iree_fctprintf)
//===----------------------------------------------------------------------===//

TEST(Printf, CallbackBasic) {
  EXPECT_EQ(FormatCallback("%d + %s", 42, "hello"), "42 + hello");
}

TEST(Printf, CallbackMatchesBuffer) {
  // Verify callback and buffer modes produce identical output.
  auto check = [](const char* format, auto... args) {
    std::string buffer_result = Format(format, args...);
    std::string callback_result = FormatCallback(format, args...);
    EXPECT_EQ(buffer_result, callback_result) << "Format: " << format;
  };
  check("%d", 42);
  check("%s", "hello");
  check("%.3f", 3.14159);
  check("%08X", 0xDEADBEEFu);
  check("%+10.2f", -3.14);
}

//===----------------------------------------------------------------------===//
// Differential testing against libc
//===----------------------------------------------------------------------===//

TEST(Printf, DifferentialIntegers) {
  ExpectMatchesLibc("%d", 0);
  ExpectMatchesLibc("%d", 42);
  ExpectMatchesLibc("%d", -42);
  ExpectMatchesLibc("%d", INT_MAX);
  ExpectMatchesLibc("%d", INT_MIN);
  ExpectMatchesLibc("%u", 0u);
  ExpectMatchesLibc("%u", UINT_MAX);
  ExpectMatchesLibc("%x", 255u);
  ExpectMatchesLibc("%X", 255u);
  ExpectMatchesLibc("%o", 8u);
  ExpectMatchesLibc("%08x", 255u);
  ExpectMatchesLibc("%#x", 255u);
  ExpectMatchesLibc("%#X", 255u);
  ExpectMatchesLibc("%#o", 8u);
  ExpectMatchesLibc("%+d", 42);
  ExpectMatchesLibc("% d", 42);
  ExpectMatchesLibc("%10d", 42);
  ExpectMatchesLibc("%-10d", 42);
  ExpectMatchesLibc("%010d", 42);
  ExpectMatchesLibc("%.5d", 42);
  ExpectMatchesLibc("%.0d", 0);
  ExpectMatchesLibc("%lld", (long long)INT64_MAX);
  ExpectMatchesLibc("%lld", (long long)INT64_MIN);
  ExpectMatchesLibc("%llu", (unsigned long long)UINT64_MAX);
  ExpectMatchesLibc("%zu", (size_t)42);
}

TEST(Printf, DifferentialStrings) {
  ExpectMatchesLibc("%s", "hello");
  ExpectMatchesLibc("%.3s", "hello");
  ExpectMatchesLibc("%10s", "hello");
  ExpectMatchesLibc("%-10s", "hello");
  ExpectMatchesLibc("%10.3s", "hello");
  ExpectMatchesLibc("%-10.3s", "hello");
}

TEST(Printf, DifferentialFloats) {
  ExpectMatchesLibc("%f", 0.0);
  ExpectMatchesLibc("%f", 1.0);
  ExpectMatchesLibc("%f", -1.0);
  ExpectMatchesLibc("%f", 3.14159);
  ExpectMatchesLibc("%.0f", 3.0);
  ExpectMatchesLibc("%.2f", 3.14);
  ExpectMatchesLibc("%e", 0.0);
  ExpectMatchesLibc("%e", 1.0);
  ExpectMatchesLibc("%e", 100.0);
  ExpectMatchesLibc("%.2e", 1.0);
  ExpectMatchesLibc("%g", 0.0);
  ExpectMatchesLibc("%g", 1.0);
  ExpectMatchesLibc("%g", 100.0);
  ExpectMatchesLibc("%g", 0.0001);
}

//===----------------------------------------------------------------------===//
// PRI macro compatibility (verifies our format specifiers work with standard
// PRI macros that libc format-checking attributes expect)
//===----------------------------------------------------------------------===//

TEST(Printf, PRIMacros) {
  EXPECT_EQ(Format("%" PRId64, (int64_t)42), "42");
  EXPECT_EQ(Format("%" PRIu64, (uint64_t)42), "42");
  EXPECT_EQ(Format("%" PRIx64, (uint64_t)255), "ff");
  EXPECT_EQ(Format("%" PRIX64, (uint64_t)255), "FF");
  EXPECT_EQ(Format("%" PRId32, (int32_t)-42), "-42");
  EXPECT_EQ(Format("%" PRIu32, (uint32_t)42), "42");
}

//===----------------------------------------------------------------------===//
// Coverage: edge cases for paths not exercised by differential tests
//===----------------------------------------------------------------------===//

TEST(Printf, VariadicCallbackWrapper) {
  // Exercise iree_fctprintf (variadic) vs iree_vfctprintf (va_list).
  std::string result;
  int length =
      iree_fctprintf(format_callback_append, &result, "%d-%s", 42, "ok");
  EXPECT_EQ(length, 5);
  EXPECT_EQ(result, "42-ok");
}

TEST(Printf, LargeWidthPrecisionClamping) {
  // Width/precision > IREE_PRINTF_MAX_WIDTH_PRECISION (10000) gets clamped.
  // This exercises the clamping path in iree_printf_parse_uint and the dynamic
  // width/precision parser.
  std::string result = Format("%99999d", 42);
  EXPECT_GE((int)result.size(), 10000);
  EXPECT_LE((int)result.size(), 10001);
  // Dynamic width clamping via *.
  result = Format("%*d", 99999, 42);
  EXPECT_GE((int)result.size(), 10000);
  EXPECT_LE((int)result.size(), 10001);
  // Dynamic precision clamping via *.
  result = Format("%.*d", 99999, 42);
  EXPECT_GE((int)result.size(), 10000);
  EXPECT_LE((int)result.size(), 10001);
}

TEST(Printf, ExponentialLargePrecisionZero) {
  // %.400e of 0.0 should produce "0." followed by 400 zeros then "e+00".
  // This exercises the trailing_zeros overflow path in format_exponential.
  std::string result = Format("%.400e", 0.0);
  EXPECT_EQ((int)result.size(), 406);  // "0." + 400 zeros + "e+00"
  EXPECT_EQ(result.substr(0, 2), "0.");
  EXPECT_EQ(result.substr(result.size() - 4), "e+00");
}

TEST(Printf, SubnormalExponential) {
  // Subnormals exercise the multi-step normalization path where the
  // correction loops (normalized < 1.0 / normalized >= 10.0) may fire.
  std::string result = Format("%e", 5e-324);        // Smallest subnormal.
  EXPECT_NE(result.find("e-"), std::string::npos);  // Must have exponent.
  result = Format("%e", 2.2250738585072014e-308);   // Smallest normal.
  EXPECT_NE(result.find("e-"), std::string::npos);
}

}  // namespace
