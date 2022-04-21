// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_DEVICE_DEVICE_H_
#define IREE_BUILTINS_DEVICE_DEVICE_H_

//===----------------------------------------------------------------------===//
// A simplified libc/libm-alike that is designed to compile to portable LLVM IR.
//===----------------------------------------------------------------------===//
// This library is focused on supporting the subset of LLVM's RuntimeLibcalls
// that we need in our embedded executable binaries. This means that things like
// printf, malloc, etc are excluded.
//
// See the full list of possible functions here:
// third_party/llvm-project/llvm/include/llvm/IR/RuntimeLibcalls.def
//
// Code here must not use any system headers - as almost all pull in bits/ and
// various other target-dependent definitions that make the resulting IR
// non-portable. This means there is no size_t, etc. Any definitions that may
// come from an std* file must be redefined here with care.
//
// Code must also not use any mutable global or thread-local state ala
// errno/rounding modes/etc. Each of the functions in the library will be called
// concurrently from multiple threads and from multiple source modules. There
// must be no mutable static values anywhere.
//
// Avoid #ifdef entirely: they indicate a leakage of host build configuration
// into what is supposed to be a portable module. Anything that requires
// target-specific conditional logic must be implemented via an extern that
// can be substituted by the IREE compiler when producing the final
// target-specific module.

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// IREE_DEVICE_STANDALONE:
// Define to have libdevice's implementation of builtins alias the standard
// names. If undefined then the host toolchain implementations will be used.

//===----------------------------------------------------------------------===//
// Attributes and metadata
//===----------------------------------------------------------------------===//

// Tagged on functions that are part of the public API.
#ifdef __cplusplus
#define IREE_DEVICE_EXPORT extern "C"
#else
#define IREE_DEVICE_EXPORT
#endif  // __cplusplus

// `restrict` keyword, not supported by some older compilers.
// We define our own macro in case dependencies use `restrict` differently.
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define IREE_DEVICE_RESTRICT __restrict
#elif defined(_MSC_VER)
#define IREE_DEVICE_RESTRICT
#elif defined(__cplusplus)
#define IREE_DEVICE_RESTRICT __restrict__
#else
#define IREE_DEVICE_RESTRICT restrict
#endif  // _MSC_VER

//===----------------------------------------------------------------------===//
// stdint.h
//===----------------------------------------------------------------------===//
// https://pubs.opengroup.org/onlinepubs/009604599/basedefs/stdint.h.html
// NOTE: no size_t/ptrdiff_t/etc (as they are target dependent).

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

//===----------------------------------------------------------------------===//
// Target-specific queries
//===----------------------------------------------------------------------===//
// These are substituted with values from the compiler and must not be specified
// here in C before we generate the IR.

// Do not use: here as an example. Remove once we have any other flag.
extern int libdevice_platform_example_flag;
// The value used when not coming from the compiler.
#define LIBDEVICE_PLATFORM_EXAMPLE_FLAG 0

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Converts a 16-bit floating-point value to a 32-bit C `float`.
IREE_DEVICE_EXPORT float iree_h2f_ieee(short param);

// Converts a 32-bit C `float` value to a 16-bit floating-point value.
IREE_DEVICE_EXPORT short iree_f2h_ieee(float param);

#endif  // IREE_BUILTINS_DEVICE_DEVICE_H_
