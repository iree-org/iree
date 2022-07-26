// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_MMT4D_MMT4D_H_
#define IREE_BUILTINS_MMT4D_MMT4D_H_

//===----------------------------------------------------------------------===//
// MMT4D microkernel library
//===----------------------------------------------------------------------===//
// This library is focused on supporting usage of MMT4D for GEMMs from both
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

// We require that this header compile on bare-metal targets with no stdlib.
// These two headers are clean and do not include any other headers:
#include "iree/base/attributes.h"
#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Target architecture selection
//===----------------------------------------------------------------------===//
// The "generic" target is used if no other target is specified. All platforms
// should support the generic path and then optionally provide their own
// specializations as needed. The generic path can be forced by passing
// -DIREE_MMT4D_ARCH_GENERIC_32=1 or -DIREE_MMT4D_ARCH_GENERIC_64=1 to the
// compiler in addition to -DIREE_PLATFORM_GENERIC=1.

#if defined(IREE_MMT4D_ARCH_GENERIC_32)
#define IREE_MMT4D_SIZE_TYPE int32_t
#elif defined(IREE_MMT4D_ARCH_GENERIC_64)
#define IREE_MMT4D_SIZE_TYPE int64_t
#elif defined(IREE_ARCH_ARM_64)
#define IREE_MMT4D_ARCH_ARM_64 1
#define IREE_MMT4D_SIZE_TYPE int64_t
#else
#define IREE_MMT4D_ARCH_GENERIC_64 1
#define IREE_MMT4D_SIZE_TYPE int64_t
#endif  // IREE_ARCH_*

//===----------------------------------------------------------------------===//
// Attributes and metadata
//===----------------------------------------------------------------------===//

// Tagged on functions that are part of the public API.
// TODO(benvanik): use this to change symbol visibility? We don't want a library
// that embeds this one to transitively export functions but may want to have
// the functions exported based on how we link them. For now this is used as
// documentation.
#define IREE_MMT4D_EXPORT

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

// Use iree_mmt4d_size_t for all sizes that may need pointer width.
// For any argument that is known to fit in a specific size prefer that to
// ensure this code operates well on systems with small/weird widths (x32/ilp32,
// etc).
typedef IREE_MMT4D_SIZE_TYPE iree_mmt4d_size_t;

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Example matmul tile. This should be removed after we have any real methods
// that can be used to demonstrate how this all fits together. The shape of the
// method is close to what we need (proper buffer types, using the
// iree_mmt4d_size_t type for strides, and taking the minimal required metadata)
// but what we pass is dependent on the what the compiler produces.
//
// Returns 0 if the parameters were valid and the operation was performed.
// Non-zero results will fail the entire submission or lose the device and
// should be used as if an abort() ("no correct execution is possible after
// this point").
IREE_MMT4D_EXPORT int iree_mmt4d_example_matmul_f32(
    const float* lhs, iree_mmt4d_size_t lhs_stride, const float* rhs,
    iree_mmt4d_size_t rhs_stride, float* IREE_RESTRICT out,
    iree_mmt4d_size_t out_stride, int32_t m, int32_t n, int32_t k, float alpha,
    float beta);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_MMT4D_MMT4D_H_
