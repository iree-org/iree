// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_COMMON_H_
#define IREE_BUILTINS_UKERNEL_COMMON_H_

//===----------------------------------------------------------------------===//
// Generic microkernel library
//===----------------------------------------------------------------------===//
//
// Rules summary:
// 1. Microkernels are bare-metal, excluding even the standard C library.
//    a. Can't #include any system header.
//    b. Can't #include any standard library header.
//    c. Can't interface with the OS in any way.
// 2. Microkernels code may be specialized for a target CPU architecture, but
//    not for a target platform/OS/triple. In particular:
//    a. It's OK to have a `#ifdef __aarch64__` but not a `#ifdef __ANDROID__`.
// 3. Microkernels are pure/reentrant/stateless.
//    a. Pure: the only effect of calling a ukernel is to write to destination
//       buffers specified by pointers passed as ukernel arguments.
//    b. Reentrant: ukernels may be called concurrently with
//       themselves, other ukernels, or any other code, on any thread.
//    c. Stateless: ukernels can't access any nonconstant global variable.

#ifdef __cplusplus
#error This file should only be included in ukernel/ code, which should be C, not C++.
#endif  // __cplusplus

// Include common flag values, shared with the compiler.
#include "iree/builtins/ukernel/exported_bits.h"

// Clang on Windows has __builtin_clzll; otherwise we need to use the
// windows intrinsic functions.
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#endif  // defined(_MSC_VER)

//===----------------------------------------------------------------------===//
// Compiler/version checks
//===----------------------------------------------------------------------===//

#if defined(__GNUC__)
#define IREE_UK_COMPILER_CLANG_OR_GCC
#endif

#define IREE_UK_VERSION_GREATER_EQUAL(a_major, a_minor, b_major, b_minor) \
  ((a_major > b_major) || ((a_major == b_major) && (a_minor >= b_minor)))

#if defined(__clang__)
#define IREE_UK_COMPILER_CLANG
#define IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(major, minor) \
  IREE_UK_VERSION_GREATER_EQUAL(__clang_major__, __clang_minor__, major, minor)
#else  // defined(__clang__)
#define IREE_UK_COMPILER_CLANG_VERSION_AT_LEAST(major, minor) 0
#endif  // defined(__clang__)

#if defined(IREE_UK_COMPILER_CLANG_OR_GCC) && !defined(IREE_UK_COMPILER_CLANG)
#define IREE_UK_COMPILER_GCC
#define IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(major, minor) \
  IREE_UK_VERSION_GREATER_EQUAL(__GNUC__, __GNUC_MINOR__, major, minor)
#else  // defined(__GNUC__) && !defined(IREE_UK_COMPILER_CLANG)
#define IREE_UK_COMPILER_GCC_VERSION_AT_LEAST(major, minor) 0
#endif  // defined(__GNUC__) && !defined(IREE_UK_COMPILER_CLANG)

#if defined(_MSC_VER)
#define IREE_UK_COMPILER_MSVC
#define IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(msc_ver) (_MSC_VER >= msc_ver)
#else  // defined(_MSC_VER)
#define IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(msc_ver) 0
#endif  // defined(_MSC_VER)

//===----------------------------------------------------------------------===//
// Attributes and metadata
//===----------------------------------------------------------------------===//

// Queries for [[attribute]] identifiers in modern compilers.
#ifdef __has_attribute
#define IREE_UK_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define IREE_UK_HAVE_ATTRIBUTE(x) 0
#endif  // __has_attribute

#ifdef __has_builtin
#define IREE_UK_HAVE_BUILTIN(x) __has_builtin(x)
#else
#define IREE_UK_HAVE_BUILTIN(x) 0
#endif  // __has_builtin

// Tagged on functions that are part of the public API.
// TODO(benvanik): use this to change symbol visibility? We don't want a library
// that embeds this one to transitively export functions but may want to have
// the functions exported based on how we link them. For now this is used as
// documentation.
#define IREE_UK_EXPORT

// Local fork of IREE_RESTRICT. We can't #include iree/base/attributes.h because
// it drags in platform headers, via target_platform.h. TODO, consider sharing
// this and other attributes that can be defined without any #include.
#if IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(1900)
#define IREE_UK_RESTRICT __restrict
#elif defined(IREE_UK_COMPILER_MSVC)
#define IREE_UK_RESTRICT
#else
#define IREE_UK_RESTRICT restrict
#endif  // IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(1900)

#if IREE_UK_HAVE_ATTRIBUTE(noinline) || defined(IREE_UK_COMPILER_GCC)
#define IREE_UK_ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
#define IREE_UK_ATTRIBUTE_NOINLINE
#endif  // IREE_UK_HAVE_ATTRIBUTE(noinline)

#if defined(IREE_UK_COMPILER_CLANG_OR_GCC)
#define IREE_UK_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_UK_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define IREE_UK_LIKELY(x) (x)
#define IREE_UK_UNLIKELY(x) (x)
#endif  // IREE_HAVE_ATTRIBUTE(likely)

#if IREE_UK_HAVE_ATTRIBUTE(aligned) || defined(IREE_UK_COMPILER_GCC)
#define IREE_UK_ATTRIBUTE_ALIGNED(N) __attribute__((aligned(N)))
#else
#define IREE_UK_ATTRIBUTE_ALIGNED(N)
#endif  // IREE_UK_HAVE_ATTRIBUTE(noinline)

#if IREE_UK_HAVE_ATTRIBUTE(maybe_unused) && defined(IREE_UK_COMPILER_CLANG)
#define IREE_UK_ATTRIBUTE_UNUSED __attribute__((maybe_unused))
#elif IREE_UK_HAVE_ATTRIBUTE(unused) || defined(IREE_UK_COMPILER_CLANG_OR_GCC)
#define IREE_UK_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define IREE_UK_ATTRIBUTE_UNUSED
#endif  // IREE_UK_HAVE_ATTRIBUTE(maybe_unused / unused)

//===----------------------------------------------------------------------===//
// Local replacement for stdbool.h
//===----------------------------------------------------------------------===//

// Exactly as in stdbool.h.
// As stdbool.h is only macros, not typedefs, and it is standardized how these
// macros expand, we can simply do them here. We still avoid #including it
// in case in some toolchain it might include unexpected other headers.
#define bool _Bool
#define true 1
#define false 0

//===----------------------------------------------------------------------===//
// IREE_UK_STATIC_ASSERT, a wrapper around the static-assertion keyword.
//===----------------------------------------------------------------------===//

// Currently unconditionally _Static_assert since C11, but will have to evaluate
// to static_assert on C23.
#define IREE_UK_STATIC_ASSERT(COND) _Static_assert(COND, #COND)

//===----------------------------------------------------------------------===//
// Local replacements for stdint.h types and constants
// Refer to the comment at the top of this file for why we can't include
// stdint.h.
//===----------------------------------------------------------------------===//

// These typedefs are making assumptions about the widths of standard C types.
// These assumptions are guarded by the IREE_UK_STATIC_ASSERT's below.
typedef signed char iree_uk_int8_t;
typedef short iree_uk_int16_t;
typedef int iree_uk_int32_t;
typedef long long iree_uk_int64_t;
typedef unsigned char iree_uk_uint8_t;
typedef unsigned short iree_uk_uint16_t;
typedef unsigned int iree_uk_uint32_t;
typedef unsigned long long iree_uk_uint64_t;

IREE_UK_STATIC_ASSERT(sizeof(iree_uk_int8_t) == 1);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_int16_t) == 2);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_int32_t) == 4);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_int64_t) == 8);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_uint8_t) == 1);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_uint16_t) == 2);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_uint32_t) == 4);
IREE_UK_STATIC_ASSERT(sizeof(iree_uk_uint64_t) == 8);

#define IREE_UK_INT8_MIN -0x80
#define IREE_UK_INT16_MIN -0x8000
#define IREE_UK_INT32_MIN -0x80000000
#define IREE_UK_INT64_MIN -0x8000000000000000LL
#define IREE_UK_INT8_MAX 0x7f
#define IREE_UK_INT16_MAX 0x7fff
#define IREE_UK_INT32_MAX 0x7fffffff
#define IREE_UK_INT64_MAX 0x7fffffffffffffffLL
#define IREE_UK_UINT8_MAX 0xffU
#define IREE_UK_UINT16_MAX 0xffffU
#define IREE_UK_UINT32_MAX 0xffffffffU
#define IREE_UK_UINT64_MAX 0xffffffffffffffffULL

// Helper for microkernel input validation. Macro to be type-generic.
#define IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(VALUE, BIT_COUNT) \
  (((VALUE) >= 0) && !((VALUE) >> (BIT_COUNT)))

// Helper for checking CPU feature requirements.
static inline bool iree_uk_all_bits_set(const iree_uk_uint64_t val,
                                        const iree_uk_uint64_t required_bits) {
  return (val & required_bits) == required_bits;
}

//===----------------------------------------------------------------------===//
// Architecture detection (copied from target_platorm.h)
//===----------------------------------------------------------------------===//

#if defined(__arm64) || defined(__aarch64__) || defined(_M_ARM64) || \
    defined(_M_ARM64EC)
#define IREE_UK_ARCH_ARM_64 1
#elif defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || \
    defined(__TARGET_ARCH_THUMB) || defined(_M_ARM)
#define IREE_UK_ARCH_ARM_32 1
#endif  // ARM

#if defined(__riscv) && (__riscv_xlen == 32)
#define IREE_UK_ARCH_RISCV_32 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#define IREE_UK_ARCH_RISCV_64 1
#endif  // RISCV

#if defined(__wasm32__)
#define IREE_UK_ARCH_WASM_32 1
#elif defined(__wasm64__)
#define IREE_UK_ARCH_WASM_64 1
#endif  // WASM

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(__i386) || defined(_M_IX86) || defined(_X86_)
#define IREE_UK_ARCH_X86_32 1
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || \
    defined(__amd64) || defined(_M_X64)
#define IREE_UK_ARCH_X86_64 1
#endif  // X86

//===----------------------------------------------------------------------===//
// Architecture bitness
//===----------------------------------------------------------------------===//

#if defined(IREE_UK_ARCH_ARM_64) || defined(IREE_UK_ARCH_RISCV_64) || \
    defined(IREE_UK_ARCH_WASM_64) || defined(IREE_UK_ARCH_X86_64)
#define IREE_UK_ARCH_IS_64_BIT
#elif defined(IREE_UK_ARCH_ARM_32) || defined(IREE_UK_ARCH_RISCV_32) || \
    defined(IREE_UK_ARCH_WASM_32) || defined(IREE_UK_ARCH_X86_32)
#define IREE_UK_ARCH_IS_32_BIT
#else
#error Unknown architecture
#endif

//===----------------------------------------------------------------------===//
// iree_uk_index_t: signed integer type, same size as MLIR `index`.
//===----------------------------------------------------------------------===//

// When ukernels are built as bitcode to embed in the compiler, the requirement
// here is that the size of iree_uk_index_t equals the size of the compiler's
// `index` type.
//
// So when here we define iree_uk_index_t as a pointer-sized type, there is an
// implicit assumption about how this ukernel code is built. When building as
// bitcode to embed in the compiler, this code must be built twice, once for
// some 32-bit-pointers architecture and once for some 64-bit-pointers
// architecture. Then the compiler must select the bitcode module that matches
// the size of the `index` type.

#if defined(IREE_UK_ARCH_IS_64_BIT)
typedef iree_uk_int64_t iree_uk_index_t;
#elif defined(IREE_UK_ARCH_IS_32_BIT)
typedef iree_uk_int32_t iree_uk_index_t;
#else
#error Unknown architecture
#endif

static inline void iree_uk_index_swap(iree_uk_index_t* a, iree_uk_index_t* b) {
  iree_uk_index_t t = *a;
  *a = *b;
  *b = t;
}

static inline iree_uk_index_t iree_uk_index_min(iree_uk_index_t a,
                                                iree_uk_index_t b) {
  return a <= b ? a : b;
}

static inline iree_uk_index_t iree_uk_index_max(iree_uk_index_t a,
                                                iree_uk_index_t b) {
  return a >= b ? a : b;
}

static inline iree_uk_index_t iree_uk_index_clamp(iree_uk_index_t val,
                                                  iree_uk_index_t min,
                                                  iree_uk_index_t max) {
  return iree_uk_index_min(max, iree_uk_index_max(min, val));
}

//===----------------------------------------------------------------------===//
// Local replacement for <assert.h>
//===----------------------------------------------------------------------===//

// Microkernel code needs to be stand-alone, not including the standard library
// (see comment in common.h). But it's hard to implement assertion failure
// without the standard library. So it's up to each piece of code that uses
// microkernels, to provide its own implementation of this function.
extern void iree_uk_assert_fail(const char* file, int line,
                                const char* function, const char* condition);

#ifndef NDEBUG
#define IREE_UK_ENABLE_ASSERTS
#endif

#ifdef IREE_UK_ENABLE_ASSERTS
#define IREE_UK_ASSERT(COND)                                               \
  do {                                                                     \
    if (!(COND)) iree_uk_assert_fail(__FILE__, __LINE__, __func__, #COND); \
  } while (0)
#else
#define IREE_UK_ASSERT(COND)
#endif

//===----------------------------------------------------------------------===//
// Element type IDs for the data accessed by microkernels.
//===----------------------------------------------------------------------===//

// Inspired by iree_hal_element_type_t, but more compact (8-bit instead of
// 32-bit), stand-alone (we couldn't use iree_hal_element_type_t at the moment
// anyway as that would #include more headers), and more specialized towards the
// subset of element types that we have in microkernels.
//
// The compactness is thought to be potentially valuable as many microkernels
// will have tuples of such element type ids and will perform if-else chains on
// the tuples, so if they can fit side-by-side in a single register, that will
// result in more compact code.
//
// Implementation note: we make this very bare-bones, with
// iree_uk_type_t just a typedef for iree_uk_uint8_t and
// the values given by macros, as opposed to trying to do something nicer, more
// strongly typed, etc, because of the following design goals:
// * Minimize divergence from iree_hal_element_type_t.
// * Minimize friction for microkernels authors. Examples:
//   * If people really care about writing switch statements as opposed to
//     if-else chains, it will be more convenient for them to have raw integers.
//     (C++ scoped enums would be perfect, but this is C code).
//   * If people ever need these type ids in assembly code, then the raw
//     numerical macros will be the only thing we'll be able to share with that
//     (as is the case today with  exported_bits.h).

// Defines the element type of a buffer passed to a microkernel.
//
// Used as a bit-field. Current layout:
// * Bits 4..7 encode the 'category', e.g. integer or floating-point.
//   See IREE_UK_TYPE_CATEGORY_MASK.
// * Bit 3 is currently unused and reserved. It should always be set to 0.
// * Bit 0..2 encode the bit-count-log2, i.e. the bit width, required to be
//   a power of 2. See IREE_UK_TYPE_BIT_COUNT_LOG2_MASK.
typedef iree_uk_uint8_t iree_uk_type_t;

// Mask and bit values for the 'category' field within an element type.
// The general schema is that we use low values, from 1 upward, for integer-ish
// categories and high values, from 0xF downward, for floating-point-ish
// categories. This way, we simultaneously we keep it easy to implement the
// "is floating-point" test and we keep it open how many values will be used for
// integer-ish vs float-ish categories.
#define IREE_UK_TYPE_CATEGORY_MASK 0xF0u
// None-category, only used for the none-element-type (value 0).
#define IREE_UK_TYPE_CATEGORY_NONE 0x00u
// Opaque means that the values are just bits. Use in microkernel that only copy
// elements, and do not perform arithmetic on them.
#define IREE_UK_TYPE_CATEGORY_OPAQUE 0x10u
// Signless integers. Use in microkernels that perform same-bit-width integer
// arithmetic that is insensitive to signedness. For example, same-bit-width
// element-wise integer add and mul ops.
#define IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS 0x20u
// Signed integers. Use in microkernels that are specifically performing signed
// integer arithmetic. For example, any mixed-bit-width op that involves a
// sign-extension (as in arith.extsi).
#define IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED 0x30u
// Unsigned integers. Similar comments as for signed integers.
#define IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED 0x40u
// "Brain" floating-point format. Currently only used for bfloat16.
#define IREE_UK_TYPE_CATEGORY_FLOAT_BRAIN 0xE0u
// IEEE754 floating-point format.
#define IREE_UK_TYPE_CATEGORY_FLOAT_IEEE 0xF0u

// Mask value for the 'bit-count-log2' field within an element type. 3 bits
// allow representing any power-of-two bit width from 1-bit to 128-bit, which
// matches what iree_hal_element_type_t can currently represent (as far as
// powers of two are concerned). If needed in the future, we could grow this
// by claiming the currently reserved bit 3.
#define IREE_UK_TYPE_BIT_COUNT_LOG2_MASK 0x07u

// Similar to iree_hal_element_types_t. We leave it a raw _e enum tag without a
// typedef because the enum type should never be used, only the enum values are
// expected to be used.
enum {
  IREE_UK_TYPE_NONE = IREE_UK_TYPE_CATEGORY_NONE | 0,
  IREE_UK_TYPE_OPAQUE_8 = IREE_UK_TYPE_CATEGORY_OPAQUE | 3,
  IREE_UK_TYPE_OPAQUE_16 = IREE_UK_TYPE_CATEGORY_OPAQUE | 4,
  IREE_UK_TYPE_OPAQUE_32 = IREE_UK_TYPE_CATEGORY_OPAQUE | 5,
  IREE_UK_TYPE_OPAQUE_64 = IREE_UK_TYPE_CATEGORY_OPAQUE | 6,
  IREE_UK_TYPE_INT_4 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS | 2,
  IREE_UK_TYPE_INT_8 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS | 3,
  IREE_UK_TYPE_INT_16 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS | 4,
  IREE_UK_TYPE_INT_32 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS | 5,
  IREE_UK_TYPE_INT_64 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS | 6,
  IREE_UK_TYPE_SINT_4 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 2,
  IREE_UK_TYPE_SINT_8 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 3,
  IREE_UK_TYPE_SINT_16 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 4,
  IREE_UK_TYPE_SINT_32 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 5,
  IREE_UK_TYPE_SINT_64 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 6,
  IREE_UK_TYPE_UINT_4 = IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED | 2,
  IREE_UK_TYPE_UINT_8 = IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED | 3,
  IREE_UK_TYPE_UINT_16 = IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED | 4,
  IREE_UK_TYPE_UINT_32 = IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED | 5,
  IREE_UK_TYPE_UINT_64 = IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED | 6,
  IREE_UK_TYPE_FLOAT_16 = IREE_UK_TYPE_CATEGORY_FLOAT_IEEE | 4,
  IREE_UK_TYPE_FLOAT_32 = IREE_UK_TYPE_CATEGORY_FLOAT_IEEE | 5,
  IREE_UK_TYPE_FLOAT_64 = IREE_UK_TYPE_CATEGORY_FLOAT_IEEE | 6,
  IREE_UK_TYPE_BFLOAT_16 = IREE_UK_TYPE_CATEGORY_FLOAT_BRAIN | 4,
};

IREE_UK_STATIC_ASSERT(IREE_UK_TYPE_NONE == 0);

// Accessors.
static inline iree_uk_uint8_t iree_uk_type_category(iree_uk_type_t t) {
  return t & IREE_UK_TYPE_CATEGORY_MASK;
}

static inline int iree_uk_type_bit_count_log2(iree_uk_type_t t) {
  return t & IREE_UK_TYPE_BIT_COUNT_LOG2_MASK;
}

// Mutate type category while keeping the same bit-width
static inline iree_uk_uint8_t iree_uk_type_mutate_category(
    iree_uk_type_t t, iree_uk_uint8_t new_category) {
  return new_category | iree_uk_type_bit_count_log2(t);
}

// Integer type helpers
static inline iree_uk_uint8_t iree_uk_type_is_integer(iree_uk_type_t t) {
  switch (iree_uk_type_category(t)) {
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS:
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED:
    case IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED:
      return true;
    default:
      return false;
  }
}

static inline iree_uk_uint8_t iree_uk_integer_type_as_signless(
    iree_uk_type_t t) {
  IREE_UK_ASSERT(iree_uk_type_is_integer(t));
  return iree_uk_type_mutate_category(t,
                                      IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS);
}

static inline iree_uk_uint8_t iree_uk_integer_type_as_signed(iree_uk_type_t t) {
  IREE_UK_ASSERT(iree_uk_type_is_integer(t));
  return iree_uk_type_mutate_category(t, IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED);
}

static inline iree_uk_uint8_t iree_uk_integer_type_as_unsigned(
    iree_uk_type_t t) {
  IREE_UK_ASSERT(iree_uk_type_is_integer(t));
  return iree_uk_type_mutate_category(t,
                                      IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED);
}

// Behavior is undefined if the bit-count is not a multiple of 8!
// The current implementation might return a negative value, but don't rely on
// that.
static inline int iree_uk_type_size_log2(iree_uk_type_t t) {
  int bit_count_log2 = iree_uk_type_bit_count_log2(t);
  IREE_UK_ASSERT(bit_count_log2 >= 3);
  return bit_count_log2 - 3;
}

static inline int iree_uk_type_bit_count(iree_uk_type_t t) {
  return 1 << iree_uk_type_bit_count_log2(t);
}

// Behavior is undefined if the bit-count is not a multiple of 8!
// Real C UB here (bit shift by negative amount), intentionally inviting the
// compiler to assume this can't happen.
static inline int iree_uk_type_size(iree_uk_type_t t) {
  return 1 << iree_uk_type_size_log2(t);
}

// Helper to correctly convert a bit-size to a byte-size, rounding up if the
// bit-size is not a multiple of 8.
static inline iree_uk_index_t iree_uk_bits_to_bytes_rounding_up(
    iree_uk_index_t bits) {
  return (bits + 7) / 8;
}

// Helper to correctly convert a bit-size to a byte-size, asserting that the
// bit-size is a multiple of 8.
static inline iree_uk_index_t iree_uk_bits_to_bytes_exact(
    iree_uk_index_t bits) {
  IREE_UK_ASSERT(!(bits % 8));
  return bits / 8;
}

//===----------------------------------------------------------------------===//
// Tuples of types, packed ("tied") into a word.
//===----------------------------------------------------------------------===//

// Note: the choice of word "tie" echoes C++ std::tie and generally tuple
// terminology. We used to call that "pack" but that was confusing as that word
// is also the name of a ukernel, iree_uk_pack.

typedef iree_uk_uint16_t iree_uk_type_pair_t;
typedef iree_uk_uint32_t iree_uk_type_triple_t;

// These need to be macros because they are used to define enum values.
#define IREE_UK_TIE_2_TYPES(T0, T1) ((T0) + ((T1) << 8))
#define IREE_UK_TIE_3_TYPES(T0, T1, T2) ((T0) + ((T1) << 8) + ((T2) << 16))

// Convenience macros to build tuples of literal types.
#define IREE_UK_TIE_2_TYPES_LITERAL(T0, T1) \
  IREE_UK_TIE_2_TYPES(IREE_UK_TYPE_##T0, IREE_UK_TYPE_##T1)
#define IREE_UK_TIE_3_TYPES_LITERAL(T0, T1, T2) \
  IREE_UK_TIE_3_TYPES(IREE_UK_TYPE_##T0, IREE_UK_TYPE_##T1, IREE_UK_TYPE_##T2)

static inline iree_uk_uint32_t iree_uk_tie_2_types(iree_uk_type_t t0,
                                                   iree_uk_type_t t1) {
  return IREE_UK_TIE_2_TYPES(t0, t1);
}

static inline iree_uk_uint32_t iree_uk_tie_3_types(iree_uk_type_t t0,
                                                   iree_uk_type_t t1,
                                                   iree_uk_type_t t2) {
  return IREE_UK_TIE_3_TYPES(t0, t1, t2);
}

static inline iree_uk_type_t iree_uk_untie_type(int pos,
                                                iree_uk_uint32_t word) {
  return (word >> (8 * pos)) & 0xFF;
}

//===----------------------------------------------------------------------===//
// Local replacement for <string.h>
//===----------------------------------------------------------------------===//

// The `restrict` here have the effect of enabling the compiler to rewrite this
// as a iree_uk_memcpy call.
static inline void iree_uk_memcpy(void* IREE_UK_RESTRICT dst,
                                  const void* IREE_UK_RESTRICT src,
                                  iree_uk_index_t size) {
  for (iree_uk_index_t i = 0; i < size; ++i)
    ((char*)dst)[i] = ((const char*)src)[i];
}

static inline void iree_uk_memset(void* buf, int val, iree_uk_index_t n) {
  // This naive loop is lifted to a memset by both clang and gcc on ARM64.
  for (iree_uk_index_t i = 0; i < n; ++i) ((char*)buf)[i] = val;
}

//===----------------------------------------------------------------------===//
// Count leading zeros (extracted from base/internal/math.h and adapted
// to be able to be used standalone).
//===----------------------------------------------------------------------===//

static inline int iree_uk_count_leading_zeros_u32(const iree_uk_uint32_t n) {
#if defined(_MSC_VER)
  unsigned long result = 0;  // NOLINT(runtime/int)
  if (_BitScanReverse(&result, n)) {
    return (int)(31 - result);
  }
  return 32;
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__LCZNT__)
  // NOTE: LZCNT is a risky instruction; it is not supported on architectures
  // before Haswell, yet it is encoded as 'rep bsr', which typically ignores
  // invalid rep prefixes, and interprets it as the 'bsr' instruction, which
  // returns the index of the value rather than the count, resulting in
  // incorrect code.
  return (int)__lzcnt32(n);
#endif  // defined(__LCZNT__)

  // Handle 0 as a special case because __builtin_clz(0) is undefined.
  if (n == 0) return 32;
  // Use __builtin_clz, which uses the following instructions:
  //  x86: bsr
  //  ARM64: clz
  //  PPC: cntlzd
  return (int)__builtin_clz(n);
#else
#error No clz for this arch.
#endif  // MSVC / GCC / CLANG
}

//===----------------------------------------------------------------------===//
// Power-of-two math helpers
//===----------------------------------------------------------------------===//

static inline bool iree_uk_is_po2_u32(const iree_uk_uint32_t n) {
  return !(n & (n - 1));
}

static inline int iree_uk_floor_log2_u32(const iree_uk_uint32_t n) {
  return 31 - iree_uk_count_leading_zeros_u32(n);
}

static inline int iree_uk_po2_log2_u32(const iree_uk_uint32_t n) {
  IREE_UK_ASSERT(iree_uk_is_po2_u32(n));
  return iree_uk_floor_log2_u32(n);
}

static inline int iree_uk_ceil_log2_u32(const iree_uk_uint32_t n) {
  return n <= 1 ? 0 : (1 + iree_uk_floor_log2_u32(n - 1));
}

//===----------------------------------------------------------------------===//
// Portable explicit prefetch hints
//
// Architecture-specific files may use architecture prefetch intrinsics instead.
//
// Any explicit prefetch should be justified by measurements on a specific CPU.
//===----------------------------------------------------------------------===//

#if IREE_UK_HAVE_BUILTIN(__builtin_prefetch) || defined(IREE_UK_COMPILER_GCC)
#define IREE_UK_PREFETCH_RO(ptr, hint) __builtin_prefetch((ptr), /*rw=*/0, hint)
#define IREE_UK_PREFETCH_RW(ptr, hint) __builtin_prefetch((ptr), /*rw=*/1, hint)
#else
#define IREE_UK_PREFETCH_RO(ptr, hint)
#define IREE_UK_PREFETCH_RW(ptr, hint)
#endif

// This is how the 0--3 locality hint is interpreted by both Clang and GCC on
// both arm64 and x86-64.
#define IREE_UK_PREFETCH_LOCALITY_NONE 0  // No locality. Data accessed once.
#define IREE_UK_PREFETCH_LOCALITY_L3 1    // Some locality. Try to keep in L3.
#define IREE_UK_PREFETCH_LOCALITY_L2 2    // More locality. Try to keep in L2.
#define IREE_UK_PREFETCH_LOCALITY_L1 3    // Most locality. Try to keep in L1.

//===----------------------------------------------------------------------===//
// 16-bit <-> 32-bit floating point conversions.
// Adapted from runtime/src/iree/base/internal/math.h.
//===----------------------------------------------------------------------===//

// NOTE: We used to have code here using built-in _Float16 type support.
// It worked well (https://godbolt.org/z/3a6WM39M1) until it didn't for
// some people (#14549). It's not worth the hassle, this is only used
// in slow generic fallbacks or test code, and we weren't able to use
// a builtin for bf16 anyway.

#define IREE_UK_FP_FORMAT_CONSTANTS(prefix, bits, ebits)            \
  const int prefix##exp_bits IREE_UK_ATTRIBUTE_UNUSED = ebits;      \
  const int prefix##mantissa_bits IREE_UK_ATTRIBUTE_UNUSED =        \
      bits - 1 - prefix##exp_bits;                                  \
  const int prefix##sign_shift IREE_UK_ATTRIBUTE_UNUSED = bits - 1; \
  const int prefix##exp_shift IREE_UK_ATTRIBUTE_UNUSED =            \
      prefix##mantissa_bits;                                        \
  const int prefix##sign_mask IREE_UK_ATTRIBUTE_UNUSED =            \
      1u << prefix##sign_shift;                                     \
  const int prefix##mantissa_mask IREE_UK_ATTRIBUTE_UNUSED =        \
      (1u << prefix##exp_shift) - 1;                                \
  const int prefix##exp_mask IREE_UK_ATTRIBUTE_UNUSED =             \
      (1u << prefix##sign_shift) - (1u << prefix##exp_shift);

static inline float iree_uk_generic_fp16_to_f32(iree_uk_uint16_t f16_value,
                                                int exp_bits) {
  IREE_UK_FP_FORMAT_CONSTANTS(f16_, 16, exp_bits)
  IREE_UK_FP_FORMAT_CONSTANTS(f32_, 32, 8)
  const iree_uk_uint32_t f16_sign = f16_value & f16_sign_mask;
  const iree_uk_uint32_t f32_sign = f16_sign
                                    << (f32_sign_shift - f16_sign_shift);
  const iree_uk_uint32_t f16_exp = f16_value & f16_exp_mask;
  const iree_uk_uint32_t f16_mantissa = f16_value & f16_mantissa_mask;
  iree_uk_uint32_t f32_exp = 0;
  iree_uk_uint32_t f32_mantissa = 0;
  if (f16_exp == f16_exp_mask) {
    // NaN or Inf case.
    f32_exp = f32_exp_mask;
    if (f16_mantissa) {
      // NaN. Generate a quiet NaN.
      f32_mantissa = f32_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (f16_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_f16_exp = f16_exp >> f16_exp_shift;
    int arithmetic_f32_exp = arithmetic_f16_exp + (1 << (f32_exp_bits - 1)) -
                             (1 << (f16_exp_bits - 1));
    f32_exp = arithmetic_f32_exp << f32_exp_shift;
    f32_mantissa = f16_mantissa << (f32_mantissa_bits - f16_mantissa_bits);
  }
  const iree_uk_uint32_t u32_value = f32_sign | f32_exp | f32_mantissa;
  float f32_value;
  iree_uk_memcpy(&f32_value, &u32_value, sizeof f32_value);
  return f32_value;
}

static inline iree_uk_uint16_t iree_uk_f32_to_generic_fp16(float value,
                                                           int exp_bits) {
  IREE_UK_FP_FORMAT_CONSTANTS(f16_, 16, exp_bits)
  IREE_UK_FP_FORMAT_CONSTANTS(f32_, 32, 8)
  iree_uk_uint32_t u32_value;
  iree_uk_memcpy(&u32_value, &value, sizeof value);
  const iree_uk_uint32_t f32_sign = u32_value & f32_sign_mask;
  const iree_uk_uint32_t f16_sign =
      f32_sign >> (f32_sign_shift - f16_sign_shift);
  const iree_uk_uint32_t f32_exp = u32_value & f32_exp_mask;
  const iree_uk_uint32_t f32_mantissa = u32_value & f32_mantissa_mask;
  iree_uk_uint32_t f16_exp = 0;
  iree_uk_uint32_t f16_mantissa = 0;
  if (f32_exp == f32_exp_mask) {
    // NaN or Inf case.
    f16_exp = f16_exp_mask;
    if (f32_mantissa) {
      // NaN. Generate a quiet NaN.
      f16_mantissa = f16_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (f32_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_exp = (f32_exp >> f32_exp_shift) - (1 << (f32_exp_bits - 1));
    if (arithmetic_exp >= (1 << (f16_exp_bits - 1))) {
      // Overflow. Generate Inf. Leave zero mantissa.
      f16_exp = f16_exp_mask;
    } else if (arithmetic_exp < -(1 << (f16_exp_bits - 1))) {
      // Underflow. Generate zero. Leave zero mantissa.
      f16_exp = 0;
    } else {
      // Normal case.
      // Implement round-to-nearest-even, by adding a bias before truncating.
      // truncating.
      int even_bit = 1u << (f32_mantissa_bits - f16_mantissa_bits);
      int odd_bit = even_bit >> 1;
      iree_uk_uint32_t biased_f32_mantissa =
          f32_mantissa +
          ((f32_mantissa & even_bit) ? (odd_bit) : (odd_bit - 1));
      // Adding the bias may cause an exponent increment.
      if (biased_f32_mantissa > f32_mantissa_mask) {
        // Note: software implementations that try to be fast tend to get this
        // conditional increment of exp and zeroing of mantissa for free by
        // simplying incrementing the whole uint32 encoding of the float value,
        // so that the mantissa overflows into the exponent bits.
        // This results in magical-looking code like in the following links.
        // We'd rather not care too much about performance of this function;
        // we should only care about fp16 performance on fp16 hardware, and
        // then, we should use hardware instructions.
        // https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/c10/util/BFloat16.h#L76
        // https://gitlab.com/libeigen/eigen/-/blob/21cd3fe20990a5ac1d683806f605110962aac3f1/Eigen/src/Core/arch/Default/BFloat16.h#L565
        biased_f32_mantissa = 0;
        ++arithmetic_exp;
      }
      // The exponent increment in the above if() branch may cause overflow.
      // This is exercised by converting 65520.0f from f32 to f16. No special
      // handling is needed for this case: the above if() branch already set
      // biased_f32_mantissa=0, so we will be generating a 0 mantissa, as
      // needed for infinite values.
      f16_exp = (arithmetic_exp + (1 << (f16_exp_bits - 1))) << f16_exp_shift;
      f16_mantissa =
          biased_f32_mantissa >> (f32_mantissa_bits - f16_mantissa_bits);
    }
  }
  iree_uk_uint16_t f16_value = f16_sign | f16_exp | f16_mantissa;
  return f16_value;
}

// Converts a fp16 value to a 32-bit C `float`.
static inline float iree_uk_f16_to_f32(iree_uk_uint16_t f16_value) {
  return iree_uk_generic_fp16_to_f32(f16_value, 5);
}

// Converts a 32-bit C `float` value to a fp16 value, rounding to nearest
// even.
static inline iree_uk_uint16_t iree_uk_f32_to_f16(float value) {
  return iree_uk_f32_to_generic_fp16(value, 5);
}

// Converts a bfloat16 value to a 32-bit C `float`.
static inline float iree_uk_bf16_to_f32(iree_uk_uint16_t bf16_value) {
  return iree_uk_generic_fp16_to_f32(bf16_value, 8);
}

// Converts a 32-bit C `float` value to a bfloat16 value, rounding to nearest
// even.
static inline iree_uk_uint16_t iree_uk_f32_to_bf16(float value) {
  return iree_uk_f32_to_generic_fp16(value, 8);
}

#endif  // IREE_BUILTINS_UKERNEL_COMMON_H_
