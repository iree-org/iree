// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
// Attributes and metadata
//===----------------------------------------------------------------------===//

// Tagged on functions that are part of the public API.
#define LIBRT_EXPORT __attribute__((visibility("hidden")))

//===----------------------------------------------------------------------===//
// stdint.h
//===----------------------------------------------------------------------===//
// https://pubs.opengroup.org/onlinepubs/009604599/basedefs/stdint.h.html
// NOTE: no size_t/ptrdiff_t/etc (as they are target dependent).

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

//===----------------------------------------------------------------------===//
// Target-specific queries
//===----------------------------------------------------------------------===//
// These are substituted with values from the compiler and must not be specified
// here in C before we generate the IR.

// Do not use: here as an example. Remove once we have any other flag.
extern int librt_platform_example_flag;
