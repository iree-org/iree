// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uCUDAGemmCutlass.cuh"

//===----------------------------------------------------------------------===//
// Template Helpers Microkernel
//===----------------------------------------------------------------------===//

#if defined(KERNEL_NAME)

#define CAT(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) \
  P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15
#define TEMPLATE(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, \
                 P15)                                                         \
  CAT(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15)

#define KERNELNAME                                                       \
  TEMPLATE(KERNEL_NAME, ELEMENT_A, ELEMENT_B, ELEMENT_C, TILE_M, TILE_N, \
           TILE_K, WARP_M, WARP_N, INST_M, INST_N, INST_K, STAGES,       \
           HAS_LINALG_FILL, WRITEBACK_TO_GLOBAL)

#define STRINGIZE(x) "[uGPU Generating] " #x
#define SSTRINGIZE(x) STRINGIZE(x)
#pragma message(SSTRINGIZE(KERNELNAME))

#ifdef CUDA_ENTRY_GLOBAL
#define CUDA_ENTRY __global__
#else
#if defined(__clang__)
#define CUDA_ENTRY __device__
#else
#define CUDA_ENTRY __device__
#endif
#endif

//===----------------------------------------------------------------------===//
// C Trampoline Microkernel
//===----------------------------------------------------------------------===//

extern "C" {
/* clang-format off */ 
CUDA_ENTRY void TEMPLATE(
    KERNEL_NAME, ELEMENT_A, ELEMENT_B, ELEMENT_C,
    TILE_M, TILE_N, TILE_K, 
    WARP_M, WARP_N,  
    INST_M, INST_N, INST_K, 
    STAGES, HAS_LINALG_FILL, WRITEBACK_TO_GLOBAL)(    
    ELEMENT_A * lhs_base, ELEMENT_A* lhs_aligned, int64_t lhs_offset, int64_t lhs_start, int64_t lhs_dim2, 
    ELEMENT_B * rhs_base, ELEMENT_B* rhs_aligned, int64_t rhs_offset, int64_t rhs_start, int64_t rhs_dim2, 
    ELEMENT_C * res_base, ELEMENT_C* res_aligned, int64_t res_offset, int64_t res_start, int64_t res_dim2,
    ELEMENT_C* shm_base, ELEMENT_C* shm_aligned, int64_t shm_offset, int64_t shm_start, int64_t shm_dim2, 
    ELEMENT_C* rsh_base, ELEMENT_C* rsh_aligned, int64_t rsh_start, int64_t rsh_dim2, 
    ELEMENT_C initValue) {
  
  ELEMENT_A* lhs_begin = &lhs_base[lhs_start];
  ELEMENT_B* rhs_begin = &rhs_base[rhs_start];
  ELEMENT_C* res_begin = &res_base[res_start];

#ifdef DEBUG
  if (threadIdx.x == 0)
    printf("<<< (%d,%d), (%d,%d) >>> %p[%ld] %p[%ld] %p[%ld] %p \t --- %p %p %p\n", 
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, 
           lhs_base, lhs_start, rhs_base, rhs_start, res_base, res_start, shm_base,
           lhs_begin, rhs_begin, res_begin);
#endif
  
  gemm_ukernel<
    ELEMENT_A, ELEMENT_B, ELEMENT_C,
    TILE_M, TILE_N, TILE_K, 
    WARP_M, WARP_N,  
    INST_M, INST_N, INST_K, 
    STAGES, HAS_LINALG_FILL, WRITEBACK_TO_GLOBAL>(
        lhs_begin, lhs_dim2,
        rhs_begin, rhs_dim2,
        res_begin, res_dim2,
        shm_base, initValue);
}
/* clang-format on */
}

#endif
