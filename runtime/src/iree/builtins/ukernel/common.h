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
// For example, using IREE_UKERNEL_ARCH_ARM_64 (from arch/config.h) rather than
// IREE_ARCH_ARM_64 (from target_platform.h) means that we can control from a
// single place in the build system whether we enable ARM_64-specific code paths
// or stick to generic code.
#include "iree/builtins/ukernel/arch/config.h"

// Include common flag values, shared with the compiler.
#include "iree/builtins/ukernel/exported_flag_bits.h"

// Include IREE_UKERNEL_STATIC_ASSERT.
#include "iree/builtins/ukernel/static_assert.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Attributes and metadata
//===----------------------------------------------------------------------===//

// Tagged on functions that are part of the public API.
// TODO(benvanik): use this to change symbol visibility? We don't want a library
// that embeds this one to transitively export functions but may want to have
// the functions exported based on how we link them. For now this is used as
// documentation.
#define IREE_UKERNEL_EXPORT

// Local fork of IREE_RESTRICT. We can't #include iree/base/attributes.h because
// it drags in platform headers, via target_platform.h. TODO, consider sharing
// this and other attributes that can be defined without any #include.
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define IREE_UKERNEL_RESTRICT __restrict
#elif defined(_MSC_VER)
#define IREE_UKERNEL_RESTRICT
#elif defined(__cplusplus)
#define IREE_UKERNEL_RESTRICT __restrict__
#else
#define IREE_UKERNEL_RESTRICT restrict
#endif  // _MSC_VER

//===----------------------------------------------------------------------===//
// Local replacements for stdint.h types and constants
// Refer to the comment at the top of this file for why we can't include
// stdint.h.
//===----------------------------------------------------------------------===//

// These typedefs are making assumptions about the widths of standard C types.
// These assumptions are guarded by the IREE_UKERNEL_STATIC_ASSERT's below.
// If someday these assumptions fail, then we can always add #if's to control
// these typedefs, perhaps similarly to what is done for iree_ukernel_ssize_t
// below.
typedef signed char iree_ukernel_int8_t;
typedef short iree_ukernel_int16_t;
typedef int iree_ukernel_int32_t;
typedef long long iree_ukernel_int64_t;
typedef unsigned char iree_ukernel_uint8_t;
typedef unsigned short iree_ukernel_uint16_t;
typedef unsigned int iree_ukernel_uint32_t;
typedef unsigned long long iree_ukernel_uint64_t;

IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_int8_t) == 1);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_int16_t) == 2);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_int32_t) == 4);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_int64_t) == 8);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_uint8_t) == 1);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_uint16_t) == 2);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_uint32_t) == 4);
IREE_UKERNEL_STATIC_ASSERT(sizeof(iree_ukernel_uint64_t) == 8);

#define IREE_UKERNEL_INT8_MIN (-127i8 - 1)
#define IREE_UKERNEL_INT16_MIN (-32767i16 - 1)
#define IREE_UKERNEL_INT32_MIN (-2147483647i32 - 1)
#define IREE_UKERNEL_INT64_MIN (-9223372036854775807i64 - 1)
#define IREE_UKERNEL_INT8_MAX 127i8
#define IREE_UKERNEL_INT16_MAX 32767i16
#define IREE_UKERNEL_INT32_MAX 2147483647i32
#define IREE_UKERNEL_INT64_MAX 9223372036854775807i64
#define IREE_UKERNEL_UINT8_MAX 0xffui8
#define IREE_UKERNEL_UINT16_MAX 0xffffui16
#define IREE_UKERNEL_UINT32_MAX 0xffffffffui32
#define IREE_UKERNEL_UINT64_MAX 0xffffffffffffffffui64

//===----------------------------------------------------------------------===//
// Local replacement for ssize_t
//===----------------------------------------------------------------------===//

// Use iree_ukernel_ssize_t for all sizes that may need pointer width.
// For any argument that is known to fit in a specific size prefer that to
// ensure this code operates well on systems with small/weird widths (x32/ilp32,
// etc).
#if IREE_UKERNEL_POINTER_SIZE == 4
typedef iree_ukernel_int32_t iree_ukernel_ssize_t;
#elif IREE_UKERNEL_POINTER_SIZE == 8
typedef iree_ukernel_int64_t iree_ukernel_ssize_t;
#else
#error Unexpected pointer size
#endif

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

// Status codes returned by a mmt4d operation.
enum iree_ukernel_status_t {
  iree_ukernel_status_ok = 0,
  iree_ukernel_status_bad_type,
  iree_ukernel_status_bad_flags,
  iree_ukernel_status_unsupported_huge_or_negative_dimension,
  iree_ukernel_status_unsupported_generic_tile_size,
};

typedef enum iree_ukernel_status_t iree_ukernel_status_t;

// Convert a status code to a human-readable string.
IREE_UKERNEL_EXPORT const char* iree_ukernel_status_message(
    iree_ukernel_status_t status);

#define IREE_UKERNEL_RETURN_IF_ERROR(X)     \
  do {                                      \
    iree_ukernel_status_t status = (X);     \
    if (status != iree_ukernel_status_ok) { \
      return status;                        \
    }                                       \
  } while (0)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_COMMON_H_
