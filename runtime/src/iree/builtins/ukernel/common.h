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
// This library is focused on supporting usage of tiled microkernels from both
// runtime libraries (VMVX via the IREE VM) and compiled libraries (LLVM CPU
// codegen). It is designed to compile standalone as well as to bitcode that
// can be linked into generated libraries and has support for specialization
// in the compiler. In general treat the code as portable across architectures
// but consistently built for bare-metal systems with no stdlib.
//
// Code here must not use any system headers - as almost all pull in bits/ and
// various other target-dependent definitions that make the resulting IR
// non-portable. This means there is no size_t, etc. Any definitions that may
// come from an std* file must be redefined here with care. Target-specific
// files may include target-specific headers if carefully managed.
//
// Code must also not use any mutable global or thread-local state ala
// errno/rounding modes/etc. Each of the functions in the library will be called
// concurrently from multiple threads and from multiple source modules. There
// must be no mutable static values anywhere.
//
// Avoid #ifdef entirely where possible: they indicate a leakage of host build
// configuration into what is supposed to be a portable module. Anything that
// requires target-specific conditional logic must be implemented via an extern
// that can be substituted by the IREE compiler when producing the final
// target-specific module.

// Include the build-system-generated configured header and use it as the only
// source of information about the target we're compiling against, as opposed to
// including iree/base/target_platform.h.
//
// For example, using IREE_UKERNEL_ARCH_ARM_64 (from arch/config.h) rather than
// IREE_ARCH_ARM_64 (from target_platform.h) means that we can control from a
// single place in the build system whether we enable ARM_64-specific code paths
// or stick to generic code.
#include "iree/builtins/ukernel/arch/config.h"

// We require that this header compile on bare-metal targets with no stdlib.
// These headers are clean:
#include "iree/base/attributes.h"

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

//===----------------------------------------------------------------------===//
// stdint.h
//===----------------------------------------------------------------------===//
// https://pubs.opengroup.org/onlinepubs/009604599/basedefs/stdint.h.html
// NOTE: prefer not using size_t/ptrdiff_t/etc (as they are target dependent).
// We avoid including the toolchain file as it may not match with the target
// information we have in the IREE compiler. They're also generally tire fires
// that significantly bloat the bitcode we build and link into binaries: stdint
// on Windows includes windows.h, for example.
//
// NOTE: callers must #include <stdint.h> before this header; unfortunately
// there's not a great way to redefine these in the absence of stdint.h that
// also operates with stdint.h.

#if !defined(INT8_MIN)

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define INT8_MIN (-127i8 - 1)
#define INT16_MIN (-32767i16 - 1)
#define INT32_MIN (-2147483647i32 - 1)
#define INT64_MIN (-9223372036854775807i64 - 1)
#define INT8_MAX 127i8
#define INT16_MAX 32767i16
#define INT32_MAX 2147483647i32
#define INT64_MAX 9223372036854775807i64
#define UINT8_MAX 0xffui8
#define UINT16_MAX 0xffffui16
#define UINT32_MAX 0xffffffffui32
#define UINT64_MAX 0xffffffffffffffffui64

#endif  // !INT8_MIN

// Use iree_ukernel_size_t for all sizes that may need pointer width.
// For any argument that is known to fit in a specific size prefer that to
// ensure this code operates well on systems with small/weird widths (x32/ilp32,
// etc).
// TODO: it's probably too surprising that a type named *_size_t is signed.
#if IREE_UKERNEL_POINTER_SIZE == 4
typedef int32_t iree_ukernel_size_t;
#elif IREE_UKERNEL_POINTER_SIZE == 8
typedef int64_t iree_ukernel_size_t;
#else
#error Unexpected pointer size
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_COMMON_H_
