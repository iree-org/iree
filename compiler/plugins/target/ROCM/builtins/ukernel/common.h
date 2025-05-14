// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Local definitions, substitute for standard headers.

// Our microkernels are completely self-contained, not even including standard
// C library headers such as stdint.h, as that invariably runs into build
// breakage on some compilation hosts where these headers happen to include
// platform stuff that wasn't expecting to get compiled for AMDGPU.

// This code is only ever getting compiled by recent Clang, in C23 mode, and for
// the AMDGPU target. This greatly simplifies this header; in particular, we
// can use any Clang built-in or predefined macro.

#ifndef COMPILER_PLUGINS_TARGET_ROCM_BUILTINS_UKERNEL_COMMON_H_
#define COMPILER_PLUGINS_TARGET_ROCM_BUILTINS_UKERNEL_COMMON_H_

//===----------------------------------------------------------------------===//
// Local replacements for stdint.h
//===----------------------------------------------------------------------===//

typedef __INT8_TYPE__ int8_t;
typedef __INT16_TYPE__ int16_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT64_TYPE__ int64_t;
typedef __UINT8_TYPE__ uint8_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT64_TYPE__ uint64_t;

#define INT8_MIN __INT8_MIN__
#define INT16_MIN __INT16_MIN__
#define INT32_MIN __INT32_MIN__
#define INT64_MIN __INT64_MIN__
#define INT8_MAX __INT8_MAX__
#define INT16_MAX __INT16_MAX__
#define INT32_MAX __INT32_MAX__
#define INT64_MAX __INT64_MAX__
#define UINT8_MAX __UINT8_MAX__
#define UINT16_MAX __UINT16_MAX__
#define UINT32_MAX __UINT32_MAX__
#define UINT64_MAX __UINT64_MAX__

// Note: intentionally NO size_t and other address-space-size-dependent types.
// Since on AMDGPU we are dealing with different address spaces with different
// pointer sizes (64-bit global vs. 32-bit LDS), each function can use int64 or
// int32 explicitly.

//===----------------------------------------------------------------------===//
// Local replacements for float.h
//===----------------------------------------------------------------------===//

#define FLT_EPSILON __FLT_EPSILON__
#define FLT_MIN __FLT_MIN__
#define FLT_MAX __FLT_MAX__

//===----------------------------------------------------------------------===//
// Vector typedefs
//===----------------------------------------------------------------------===//

typedef __attribute__((__vector_size__(4 * 4))) int32_t int32x4_t;

//===----------------------------------------------------------------------===//
// Declarations for Clangd, which may be slightly older than actual clang.
// Drop these as clangd versions used in practice gain these builtins.
// Unconditionally declaring these, regardless of clang version, ensures that
// any mistake in the declaration is caught by newer clangs.
//===----------------------------------------------------------------------===//

// Missing in clang 18. https://github.com/llvm/llvm-project/pull/80741.
unsigned int __builtin_amdgcn_wavefrontsize();

//===----------------------------------------------------------------------===//
// Local replacements for AMD device library headers
//===----------------------------------------------------------------------===//

_Float16 __ockl_wfred_max_f16(_Float16);
int64_t __ockl_wfred_min_i64(int64_t);
int32_t __ockl_wfred_min_i32(int32_t);

static inline void __threadfence_block() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

[[clang::convergent]] static inline void __syncthreads() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

//===----------------------------------------------------------------------===//
// Local replacements for HIP headers
//===----------------------------------------------------------------------===//

static inline int __lane_id() {
  return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
}

static inline int __shfl_xor_i(int var, int lane_mask) {
  const int width = __builtin_amdgcn_wavefrontsize();
  int self = __lane_id();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}

static inline float __shfl_xor_f(float var, int lane_mask) {
  union {
    int i;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor_i(tmp.i, lane_mask);
  return tmp.f;
}

static inline uint64_t __ballot(int predicate) {
  return __builtin_amdgcn_uicmp(
      predicate, 0, 33 /*ICMP_NE from llvm/include/llvm/IR/InstrTypes.h*/);
}

#endif // COMPILER_PLUGINS_TARGET_ROCM_BUILTINS_UKERNEL_COMMON_H_
