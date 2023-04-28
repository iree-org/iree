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
//    not for a complete target platform/OS/triple. In particular:
//    a. It's OK to have a `#ifdef __aarch64__` but not a `#ifdef __ANDROID__`.
// 3. Microkernels are pure/reentrant/stateless.
//    a. Pure: the only effect of calling a ukernel is to write to destination
//       buffers specified by pointers passed as ukernel arguments.
//    b. Reentrant: ukernels may be called concurrently with
//       themselves, other ukernels, or any other code, on any thread.
//    c. Stateless: ukernels can't mutate any global (or static local) variable.
//
// Explanation:
// 1. a. Microkernels will eventually be called from IREE LLVM-CPU codegen
//       modules. So we need to be able to build microkernels for all the target
//       architectures that iree-compile supports. If microkernels included
//       system headers, we would need to compile them not merely for each
//       target architecture but for each target triple, and we would need to
//       have the system headers for each of these.
// 1. b. Follows from a. because many standard C library headers #include
//       system headers. We can't keep track of which do. Even plausibly "pure"
//       ones such as <stdint.h> have been known to drag in surprising amounts.
// 1. c. Since we're only targeting a CPU architecture, not a complete target
//       platform/OS, we can't use any features that rely on the OS. For example
//       we can't use TLS (thread-local-storage) or Linux's auxiliary vector, or
//       syscalls.
//       * This means in particular that any CPU feature detection needs
//         to be made ahead of calling the ukernel, and the results passed as
//         ukernel args.
// 2. We don't want code to depend on platform `#ifdefs` beyond just target CPU
//    architecture ifdefs, in any way --- even if the code paths are not
//    interfacing with the OS (see 1.c.), it's still forbidden to have separate
//    code paths. When we will in the future call microkernels from IREE
//    LLVM-CPU codegen, this will make it legal for us to compile them only for
//    each target CPU architecture, which will be easier than having to compile
//    them separately for each supported target triple.
// 3. Microkernels are typically called on tiles, after the workload has been
//    tiled and distributed to several threads. Keeping microkernels pure,
//    reentrant and stateless keeps them automatically compatible with any
//    tiling and distribution that we may use in the future.
//
// FAQ:
// Q: Can a microkernel save, change, and restore the CPU float rounding mode?
//    A: Yes, as long as:
//       * It properly restores it in all its return paths.
//       * The CPU rounding mode is accessed in the microkernel's
//         own local code (as opposed to trying to use some standard library
//         header for that).
//       * The CPU architecture treats the rounding mode as a thread-local
//         setting (this tends to be the case on current CPU architectures).
// Q: How can a microkernel depend on CPU identification information?
//    A: Microkernels that need to know CPU identification information, such as
//       bits indicating support for optional SIMD ISA features, should take
//       such information as arguments. This moves the problem of obtaining the
//       CPU identification information to the caller. This serves multiple
//       purposes:
//       * This allows writing tests that exercise all variants supported by the
//         test machine, not just whichever variant would be selected for that
//         machine.
//       * On CPU architectures where only the OS can directly access CPU
//         identification bits (that includes ARM architectures), this is
//         basically required by rule 1.c. (forbidding microkernels from
//         querying the OS directly).
//         - While other CPU architectures like x86 allow userspace processes to
//           directly query CPU identification, it's best to keep all kernels
//           on all architectures aligned on this.
//         - While some OSes may trap CPU identification instructions to make
//           them appear as succeeding in userspace programs
//           (https://www.kernel.org/doc/html/latest/arm64/cpu-feature-registers.html),
//           there are portability, reliability and performance concerns with
//           that.

// Include the build-system-generated configured header and use it as the only
// source of information about the target we're compiling against, as opposed to
// including iree/base/target_platform.h.
//
// For example, using IREE_UK_ARCH_ARM_64 (from arch/config.h) rather than
// IREE_ARCH_ARM_64 (from target_platform.h) means that we can control from a
// single place in the build system whether we enable ARM_64-specific code paths
// or stick to generic code.
#include "iree/builtins/ukernel/arch/config.h"

// Now that we have IREE_UK_ARCH_ARM_64 et al defined, based on that we can
// include the architecture-specific configured header. What it defines is
// architecture-specific details that we don't necessarily need in this header,
// but having it included here ensures that all files consistently have this
// defined.
#if defined(IREE_UK_ARCH_ARM_64)
#include "iree/builtins/ukernel/arch/arm_64/config.h"
#elif defined(IREE_UK_ARCH_X86_64)
#include "iree/builtins/ukernel/arch/x86_64/config.h"
#endif

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
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define IREE_UK_RESTRICT __restrict
#elif defined(_MSC_VER)
#define IREE_UK_RESTRICT
#elif defined(__cplusplus)
#define IREE_UK_RESTRICT __restrict__
#else
#define IREE_UK_RESTRICT restrict
#endif  // _MSC_VER

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
#if IREE_UK_HAVE_BUILTIN(__builtin_unreachable) || defined(__GNUC__)
#define IREE_UK_ASSUME_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define IREE_UK_ASSUME_UNREACHABLE __assume(false)
#else
#define IREE_UK_ASSUME_UNREACHABLE
#endif

#define IREE_UK_ASSUME(condition) \
  do {                            \
    if (!(condition)) {           \
      IREE_UK_ASSUME_UNREACHABLE; \
    }                             \
  } while (false)

#if IREE_UK_HAVE_ATTRIBUTE(noinline) || \
    (defined(__GNUC__) && !defined(__clang__))
#define IREE_UK_ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
#define IREE_UK_ATTRIBUTE_NOINLINE
#endif  // IREE_UK_HAVE_ATTRIBUTE(noinline)

#if defined(__GNUC__) || defined(__clang__)
#define IREE_UK_LIKELY(x) (__builtin_expect(!!(x), 1))
#define IREE_UK_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define IREE_UK_LIKELY(x) (x)
#define IREE_UK_UNLIKELY(x) (x)
#endif  // IREE_HAVE_ATTRIBUTE(likely)

#if IREE_UK_HAVE_ATTRIBUTE(aligned) || \
    (defined(__GNUC__) && !defined(__clang__))
#define IREE_UK_ATTRIBUTE_ALIGNED(N) __attribute__((aligned(N)))
#else
#define IREE_UK_ATTRIBUTE_ALIGNED(N)
#endif  // IREE_UK_HAVE_ATTRIBUTE(noinline)

//===----------------------------------------------------------------------===//
// Local replacements for stdint.h types and constants
// Refer to the comment at the top of this file for why we can't include
// stdint.h.
//===----------------------------------------------------------------------===//

// These typedefs are making assumptions about the widths of standard C types.
// These assumptions are guarded by the IREE_UK_STATIC_ASSERT's below.
// If someday these assumptions fail, then we can always add #if's to control
// these typedefs, perhaps similarly to what is done for iree_uk_ssize_t
// below.
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
#define IREE_UK_UINT8_MAX 0xff
#define IREE_UK_UINT16_MAX 0xffff
#define IREE_UK_UINT32_MAX 0xffffffffU
#define IREE_UK_UINT64_MAX 0xffffffffffffffffULL

IREE_UK_STATIC_ASSERT(IREE_UK_INT8_MIN == -(1 << 7));
IREE_UK_STATIC_ASSERT(IREE_UK_INT16_MIN == -(1 << 15));
IREE_UK_STATIC_ASSERT(IREE_UK_INT32_MIN == -(1U << 31));
IREE_UK_STATIC_ASSERT(IREE_UK_INT64_MIN == -(1ULL << 63));
IREE_UK_STATIC_ASSERT(IREE_UK_INT8_MAX == (1 << 7) - 1);
IREE_UK_STATIC_ASSERT(IREE_UK_INT16_MAX == (1 << 15) - 1);
IREE_UK_STATIC_ASSERT(IREE_UK_INT32_MAX == (1U << 31) - 1);
IREE_UK_STATIC_ASSERT(IREE_UK_INT64_MAX == (1ULL << 63) - 1);
IREE_UK_STATIC_ASSERT(IREE_UK_UINT8_MAX == ((iree_uk_uint8_t)-1));
IREE_UK_STATIC_ASSERT(IREE_UK_UINT16_MAX == ((iree_uk_uint16_t)-1));
IREE_UK_STATIC_ASSERT(IREE_UK_UINT32_MAX == ((iree_uk_uint32_t)-1));
IREE_UK_STATIC_ASSERT(IREE_UK_UINT64_MAX == ((iree_uk_uint64_t)-1));

// Helper for microkernel input validation
#define IREE_UK_VALUE_IN_UNSIGNED_INT_RANGE(VALUE, BIT_COUNT) \
  (((VALUE) >= 0) && !((VALUE) >> (BIT_COUNT)))

//===----------------------------------------------------------------------===//
// Local replacement for ssize_t
//===----------------------------------------------------------------------===//

// Use iree_uk_ssize_t for all sizes that may need pointer width.
// For any argument that is known to fit in a specific size prefer that to
// ensure this code operates well on systems with small/weird widths (x32/ilp32,
// etc).
#if IREE_UK_POINTER_SIZE == 4
typedef iree_uk_int32_t iree_uk_ssize_t;
#elif IREE_UK_POINTER_SIZE == 8
typedef iree_uk_int64_t iree_uk_ssize_t;
#else
#error Unexpected pointer size
#endif

static inline void iree_uk_ssize_swap(iree_uk_ssize_t* a, iree_uk_ssize_t* b) {
  iree_uk_ssize_t t = *a;
  *a = *b;
  *b = t;
}

static inline iree_uk_ssize_t iree_uk_ssize_min(iree_uk_ssize_t a,
                                                iree_uk_ssize_t b) {
  return a <= b ? a : b;
}

static inline iree_uk_ssize_t iree_uk_ssize_max(iree_uk_ssize_t a,
                                                iree_uk_ssize_t b) {
  return a >= b ? a : b;
}

static inline iree_uk_ssize_t iree_uk_ssize_clamp(iree_uk_ssize_t val,
                                                  iree_uk_ssize_t min,
                                                  iree_uk_ssize_t max) {
  return iree_uk_ssize_min(max, iree_uk_ssize_max(min, val));
}

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
                                  iree_uk_ssize_t size) {
  for (iree_uk_ssize_t i = 0; i < size; ++i)
    ((char*)dst)[i] = ((const char*)src)[i];
}

static inline void iree_uk_memset(void* buf, int val, iree_uk_ssize_t n) {
  // This naive loop is lifted to a memset by both clang and gcc on ARM64.
  for (iree_uk_ssize_t i = 0; i < n; ++i) ((char*)buf)[i] = val;
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

#if IREE_UK_HAVE_BUILTIN(__builtin_prefetch) || defined(__GNUC__)
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
//
// Some ukernels have both intrinsics and inline asm code paths. This macro
// helps select the one to use based on what's enabled in the build. If both
// intrinsics and inline asm are enabled in the build, by default inline asm is
// preferred, unless `prefer_intrinsics` is true --- that's useful for testing.
//
// This is a macro in order to avoid referencing undefined function symbols.
//
// This currently falls back on intrinsics as always enabled -- we can always
// add an option for that, like IREE_UK_ENABLE_INLINE_ASM, if needed.
//===----------------------------------------------------------------------===//

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
