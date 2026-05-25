// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Local definitions, substitute for standard headers.
//
// Our microkernels are compiled with `-nostdinc -ffreestanding`, so we do not
// include standard C library headers (not even `<stdint.h>`). This is to
// guarantee build portability: the compilation host's standard headers may
// not be configured for the CPU bitcode target, and pulling them in invariably
// runs into platform-specific surprises. Clang's *intrinsic* headers
// (`<immintrin.h>` etc.) are header-only and target-aware, so they are fine to
// include directly.

#ifndef COMPILER_PLUGINS_TARGET_LLVMCPU_BUILTINS_UKERNEL_COMMON_H_
#define COMPILER_PLUGINS_TARGET_LLVMCPU_BUILTINS_UKERNEL_COMMON_H_

//===----------------------------------------------------------------------===//
// Local replacements for stdint.h, using Clang's predefined type macros.
//===----------------------------------------------------------------------===//

typedef __INT8_TYPE__ int8_t;
typedef __INT16_TYPE__ int16_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

//===----------------------------------------------------------------------===//
// Ukernel function attributes.
//===----------------------------------------------------------------------===//

// Ukernels are designed to be inlined into their caller so the per-call
// constant arguments (e.g. `intrinsics_m`) drive specialization of unrolled
// inner loops via constant propagation + DCE. Always inline.
#define IREE_UK_ALWAYS_INLINE __attribute__((always_inline))

#endif // COMPILER_PLUGINS_TARGET_LLVMCPU_BUILTINS_UKERNEL_COMMON_H_
