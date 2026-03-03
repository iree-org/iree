// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// C ABI for the MIPS custom matmul kernel.
//
// The kernel computes:
//   C[m, n] = sum_k A[m, k] * B[k, n]
//
// Each matrix is passed as a base pointer plus explicit strided-layout
// parameters that match what MLIR's memref.extract_strided_metadata produces.
// This matches the calling convention emitted by LowerMIPSToFuncCallPass.

#ifndef IREE_BUILTINS_MIPS_MY_MATMUL_KERNEL_H_
#define IREE_BUILTINS_MIPS_MY_MATMUL_KERNEL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// void my_matmul_kernel(
//   float *A, int64_t A_offset, int64_t A_stride0, int64_t A_stride1,
//   float *B, int64_t B_offset, int64_t B_stride0, int64_t B_stride1,
//   float *C, int64_t C_offset, int64_t C_stride0, int64_t C_stride1,
//   int64_t M, int64_t N, int64_t K);
void my_matmul_kernel(float *A, int64_t A_offset, int64_t A_stride0,
                      int64_t A_stride1, float *B, int64_t B_offset,
                      int64_t B_stride0, int64_t B_stride1, float *C,
                      int64_t C_offset, int64_t C_stride0, int64_t C_stride1,
                      int64_t M, int64_t N, int64_t K);

#ifdef __cplusplus
}
#endif

#endif // IREE_BUILTINS_MIPS_MY_MATMUL_KERNEL_H_
