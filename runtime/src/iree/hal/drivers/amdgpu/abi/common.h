// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Minimal type definitions and attributes shared between AMDGPU device code
// (bare-metal C compiled to GPU bitcode) and host code (standard C compiled
// for the CPU). This header is the root of the abi/ dependency tree and must
// have zero dependencies beyond system headers and the HSA type definitions.
//
// Device code gets bare-metal typedefs for fixed-width integers and the
// compiler attribute forms it needs. Host code gets the same logical macros
// backed by standard C11 and HSA headers.
//
// The abi/ headers define only struct layouts, enums, and constants that match
// what the hardware expects. No operations, no atomics, no device builtins.
// Those live in device/support/ which re-exports abi/ and adds the
// implementation machinery.

#ifndef IREE_HAL_DRIVERS_AMDGPU_ABI_COMMON_H_
#define IREE_HAL_DRIVERS_AMDGPU_ABI_COMMON_H_

//===----------------------------------------------------------------------===//
// Compiler Configuration
//===----------------------------------------------------------------------===//

#if defined(__AMDGPU__)
#define IREE_AMDGPU_TARGET_DEVICE 1
#else
#define IREE_AMDGPU_TARGET_HOST 1
#endif  // __AMDGPU__

#if defined(IREE_AMDGPU_TARGET_DEVICE)

typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

typedef int64_t ssize_t;
typedef uint64_t size_t;
typedef int64_t intptr_t;
typedef uint64_t uintptr_t;

#define UINT32_MAX 0xFFFFFFFFu
#define UINT64_MAX 0xFFFFFFFFFFFFFFFFull

#define NULL ((void*)0)

#else

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// HSA system type definitions. On the host side several abi/ types are
// typedef'd directly to their HSA equivalents (e.g. iree_hsa_signal_t) so
// that they can be used interchangeably with HSA API calls.
#include "third_party/hsa-runtime-headers/include/hsa/hsa.h"  // IWYU pragma: export

#endif  // IREE_AMDGPU_TARGET_DEVICE

// Both device and host targets use GCC/Clang so the compiler attribute syntax
// is identical. The AMDGPU driver is Linux-only and always compiled with Clang.
#define IREE_AMDGPU_RESTRICT __restrict__
#define IREE_AMDGPU_ALIGNAS(x) __attribute__((aligned(x)))
#define IREE_AMDGPU_ALIGNOF(x) __alignof__(x)
#define IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define IREE_AMDGPU_ATTRIBUTE_PACKED __attribute__((__packed__))

#if defined(__cplusplus)
#define IREE_AMDGPU_STATIC_ASSERT(expr, message) static_assert((expr), message)
#else
#define IREE_AMDGPU_STATIC_ASSERT(expr, message) _Static_assert((expr), message)
#endif  // __cplusplus

#if defined(IREE_AMDGPU_TARGET_DEVICE)
#define IREE_AMDGPU_OFFSETOF(type, field) __builtin_offsetof(type, field)
#else
#define IREE_AMDGPU_OFFSETOF(type, field) offsetof(type, field)
#endif  // IREE_AMDGPU_TARGET_DEVICE

// Tick in the agent domain.
// This can be converted to the system domain for correlation across agents and
// the host with hsa_amd_profiling_convert_tick_to_system_domain.
typedef uint64_t iree_amdgpu_device_tick_t;

#endif  // IREE_HAL_DRIVERS_AMDGPU_ABI_COMMON_H_
