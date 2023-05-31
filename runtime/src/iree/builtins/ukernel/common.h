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

// Include common flag values, shared with the compiler.
#include "iree/builtins/ukernel/exported_bits.h"

// Include IREE_UK_STATIC_ASSERT.
#include "iree/builtins/ukernel/static_assert.h"

// Clang on Windows has __builtin_clzll; otherwise we need to use the
// windows intrinsic functions.
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#endif  // defined(_MSC_VER)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
#define IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(msc_ver) (msc_ver >= _MSC_VER)
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
#elif defined(__cplusplus)
#define IREE_UK_RESTRICT __restrict__
#else
#define IREE_UK_RESTRICT restrict
#endif  // IREE_UK_COMPILER_MSVC_VERSION_AT_LEAST(1900)

// Same as LLVM_BUILTIN_UNREACHABLE. Extremely dangerous. Use only in locations
// that are provably unreachable (+/- edge case of unreachable-past-assertions
// discussed below).
//
// The potential benefit of UNREACHABLE statements is code size and/or speed
// optimization. This is an arcane optimization. As such, each use must be
// carefully justified.
//
// There is the edge case of locations that are provably unreachable when
// optional validation code is enabled, but the validation code may also be
// disabled, making the location technically reachable. Typically: assertions.
// Use careful judgement for such cases.
//
// A typical use case in microkernels is as follows. A microkernel is
// parametrized by type triples packed into uint32s, and needs to have a switch
// statement on those:
//
// switch (params->type_triple) {
//   case iree_uk_mykernel_f32f32f32:  // 0xf5f5f5
//     return 123;
//   case iree_uk_mykernel_i8i8i32:  // 0x232325
//     return 321;
//   default:
//     return 0;
// }
//
// As long as the microkernel has validation code (running at least as Debug
// assertions) validating type_triple, and this code is already past that,
// and this switch statement covers all valid cases, the `default:` case should
// be unreachable. Adding an UNREACHABLE statement there can help with code
// size. This would be negligible if the case constants were small enough to
// fit in compare-with-immediate instructions, but the 24-bit type triple
// constants here would typically not, so without UNREACHABLE, the compiler has
// to fully implement each 24-bit literal separately.
//
// https://godbolt.org/z/hTv4qqbx9 shows a snipped similar as above where
// the __builtin_unreachable shrinks the AArch64 code from 11 to 7 instructions.
#if IREE_UK_HAVE_BUILTIN(__builtin_unreachable) || defined(IREE_UK_COMPILER_GCC)
#define IREE_UK_ASSUME_UNREACHABLE __builtin_unreachable()
#elif defined(IREE_UK_COMPILER_MSVC)
#define IREE_UK_ASSUME_UNREACHABLE __assume(false)
#else
#define IREE_UK_ASSUME_UNREACHABLE
#endif  // IREE_UK_HAVE_BUILTIN(__builtin_unreachable)

#define IREE_UK_ASSUME(condition) \
  do {                            \
    if (!(condition)) {           \
      IREE_UK_ASSUME_UNREACHABLE; \
    }                             \
  } while (false)

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

#if IREE_UK_HAVE_ATTRIBUTE(weak) || defined(IREE_UK_COMPILER_GCC)
#define IREE_UK_WEAK __attribute__((weak))
#define IREE_UK_HAVE_WEAK 1
#else
#define IREE_UK_WEAK
#define IREE_UK_HAVE_WEAK 0
#endif  // IREE_UK_HAVE_ATTRIBUTE(noinline)

//===----------------------------------------------------------------------===//
// Local replacement for stdbool.h
//===----------------------------------------------------------------------===//

#ifndef __cplusplus
// Exactly as in stdbool.h.
// As stdbool.h is only macros, not typedefs, and it is standardized how these
// macros expand, we can simply do them here. We still avoid #including it
// in case in some toolchain it might include unexpected other headers.
#define bool _Bool
#define true 1
#define false 0
#endif

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
#define IREE_UK_TYPE_CATEGORY_INTEGER 0x20u
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
  IREE_UK_TYPE_INT_8 = IREE_UK_TYPE_CATEGORY_INTEGER | 3,
  IREE_UK_TYPE_INT_16 = IREE_UK_TYPE_CATEGORY_INTEGER | 4,
  IREE_UK_TYPE_INT_32 = IREE_UK_TYPE_CATEGORY_INTEGER | 5,
  IREE_UK_TYPE_INT_64 = IREE_UK_TYPE_CATEGORY_INTEGER | 6,
  IREE_UK_TYPE_SINT_8 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 3,
  IREE_UK_TYPE_SINT_16 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 4,
  IREE_UK_TYPE_SINT_32 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 5,
  IREE_UK_TYPE_SINT_64 = IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED | 6,
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

// Behavior is undefined if the bit-count is not a multiple of 8!
// The current implementation might return a negative value, but don't rely on
// that.
static inline int iree_uk_type_size_log2(iree_uk_type_t t) {
  return iree_uk_type_bit_count_log2(t) - 3;
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
// as a memcpy call.
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
// Selection between inline asm and intrinsics code paths.
//===----------------------------------------------------------------------===//

// IREE_UK_ENABLE_INLINE_ASM controls whether we enabled inline assembly at all.
// Our inline asm is GCC-extended-syntax, supported by Clang and GCC.
#if defined(IREE_UK_COMPILER_CLANG_OR_GCC)
#define IREE_UK_ENABLE_INLINE_ASM
#endif

// Undefine this to save code size by dropping intrinsics code paths that are
// redundant with an inline asm code path. This shouldn't make a difference in
// bitcode with reflection, flags being compile-time constant, allowing the
// compiler to DCE unused intrinsics code paths no matter what.
#define IREE_UK_ENABLE_INTRINSICS_EVEN_WHEN_INLINE_ASM_AVAILABLE

// Some ukernels have both intrinsics and inline asm code paths. The
// IREE_UK_SELECT_INLINE_ASM_OR_INTRINSICS macro helps selecting the one to use
// based on what's enabled in the build. If both intrinsics and inline asm are
// enabled in the build, by default inline asm is preferred, unless
// `prefer_intrinsics` is true. Useful for testing.
#if defined(IREE_UK_ENABLE_INLINE_ASM) && \
    defined(IREE_UK_ENABLE_INTRINSICS_EVEN_WHEN_INLINE_ASM_AVAILABLE)
#define IREE_UK_HAVE_BOTH_INLINE_ASM_AND_INTRINSICS
#define IREE_UK_SELECT_INLINE_ASM_OR_INTRINSICS(inline_asm, intrinsics, \
                                                prefer_intrinsics)      \
  ((prefer_intrinsics) ? (intrinsics) : (inline_asm))
#elif defined(IREE_UK_ENABLE_INLINE_ASM)
#define IREE_UK_SELECT_INLINE_ASM_OR_INTRINSICS(inline_asm, intrinsics, \
                                                prefer_intrinsics)      \
  (inline_asm)
#else  // not defined(IREE_UK_ENABLE_INLINE_ASM)
#define IREE_UK_SELECT_INLINE_ASM_OR_INTRINSICS(inline_asm, intrinsics, \
                                                prefer_intrinsics)      \
  (intrinsics)
#endif  // not defined(IREE_UK_ENABLE_INLINE_ASM)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_COMMON_H_
