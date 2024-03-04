/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/device_functions.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/device_functions.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_FUNCTIONS_HPP__
#endif

#if !defined(__DEVICE_FUNCTIONS_HPP__)
#define __DEVICE_FUNCTIONS_HPP__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__CUDACC__)

#if defined(__CUDACC_RTC__)
#define __DEVICE_FUNCTIONS_DECL__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ __device__
#define __DEVICE_HOST_FUNCTIONS_STATIC_DECL__ __device__ __host__ __cudart_builtin__
#else
#define __DEVICE_FUNCTIONS_DECL__ __device__
#define __DEVICE_FUNCTIONS_STATIC_DECL__ static __inline__ __device__
#define __DEVICE_HOST_FUNCTIONS_STATIC_DECL__ static __inline__ __device__ __host__ __cudart_builtin__
#endif /* __CUDACC_RTC__ */

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

#undef __DEVICE_FUNCTIONS_DECL__
#undef __DEVICE_FUNCTIONS_STATIC_DECL__

#endif /* __cplusplus && __CUDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#ifdef __CUDACC__
# if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define __CUDA_AND_AT_LEAST_SM_90__
#endif /* defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) */
#endif /* __CUDACC__ */

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vimax_s32_relu(const int a, const int b){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm("{max.s32.relu %0, %1, %2;}" : "=r"(res) : "r"(a), "r"(b));
  return res;
#else
  // Host and older architecture code
  int ans = max(a, b);

  return (ans > 0) ? ans : 0;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimax_s16x2_relu(const unsigned int a, const unsigned int b){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm("{max.s16x2.relu %0, %1, %2;}" : "=r"(res) : "r"(a), "r"(b));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  // Get answer
  short ansS_lo = (short)max(aS_lo, bS_lo);
  short ansS_hi = (short)max(aS_hi, bS_hi);

  // relu
  if(ansS_lo < 0){ ansS_lo = 0; }
  if(ansS_hi < 0){ ansS_hi = 0; }

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vimin_s32_relu(const int a, const int b){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm("{min.s32.relu %0, %1, %2;}" : "=r"(res) : "r"(a), "r"(b));
  return res;
#else
  // Host and older architecture code
    int ans = min(a, b);
    
    return (ans > 0) ? ans : 0;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimin_s16x2_relu(const unsigned int a, const unsigned int b){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm("{min.s16x2.relu %0, %1, %2;}" : "=r"(res) : "r"(a), "r"(b));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  // Get answer
  short ansS_lo = (short)min(aS_lo, bS_lo);
  short ansS_hi = (short)min(aS_hi, bS_hi);

  // relu
  if(ansS_lo < 0){ ansS_lo = 0; }
  if(ansS_hi < 0){ ansS_hi = 0; }

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vimax3_s32(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "max.s32 t1, %1, %2; \n\t"
      "max.s32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return max(max(a, b), c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimax3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  // Future asm code (naming/syntax may change):
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "max.s16x2 t1, %1, %2; \n\t"
      "max.s16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)max(max(aS_lo, bS_lo), cS_lo);
  short ansS_hi = (short)max(max(aS_hi, bS_hi), cS_hi);

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimax3_u32(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
int res;
  asm ("{.reg .u32 t1; \n\t"
      "max.u32 t1, %1, %2; \n\t"
      "max.u32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return max(max(a, b), c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimax3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .b32 t1; \n\t"
      "max.u16x2 t1, %1, %2; \n\t"
      "max.u16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  // Get answer
  unsigned short ansU_lo = (unsigned short)max(max(aU_lo, bU_lo), cU_lo);
  unsigned short ansU_hi = (unsigned short)max(max(aU_hi, bU_hi), cU_hi);

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vimin3_s32(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "min.s32 t1, %1, %2; \n\t"
      "min.s32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return min(min(a, b), c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimin3_s16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .b32 t1; \n\t"
      "min.s16x2 t1, %1, %2; \n\t"
      "min.s16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)min(min(aS_lo, bS_lo), cS_lo);
  short ansS_hi = (short)min(min(aS_hi, bS_hi), cS_hi);

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimin3_u32(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .u32 t1; \n\t"
      "min.u32 t1, %1, %2; \n\t"
      "min.u32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return min(min(a, b), c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimin3_u16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .b32 t1; \n\t"
      "min.u16x2 t1, %1, %2; \n\t"
      "min.u16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  // Get answer
  unsigned short ansU_lo = (unsigned short)min(min(aU_lo, bU_lo), cU_lo);
  unsigned short ansU_hi = (unsigned short)min(min(aU_hi, bU_hi), cU_hi);

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vimax3_s32_relu(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "max.s32.relu t1, %1, %2; \n\t"
      "max.s32.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  int ans = max(max(a, b), c);

  return (ans > 0) ? ans : 0;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimax3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .b32 t1; \n\t"
      "max.s16x2.relu t1, %1, %2; \n\t"
      "max.s16x2.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)max(max(aS_lo, bS_lo), cS_lo);
  short ansS_hi = (short)max(max(aS_hi, bS_hi), cS_hi);

  // relu
  if(ansS_lo < 0){ansS_lo = 0;}
  if(ansS_hi < 0){ansS_hi = 0;}

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vimin3_s32_relu(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "min.s32.relu t1, %1, %2; \n\t"
      "min.s32.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  int ans = min(min(a, b), c);

  return (ans > 0) ? ans : 0;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vimin3_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .b32 t1; \n\t"
      "min.s16x2.relu t1, %1, %2; \n\t"
      "min.s16x2.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)min(min(aS_lo, bS_lo), cS_lo);
  short ansS_hi = (short)min(min(aS_hi, bS_hi), cS_hi);

  // relu
  if(ansS_lo < 0){ansS_lo = 0;}
  if(ansS_hi < 0){ansS_hi = 0;}

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __viaddmax_s32(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "add.s32 t1, %1, %2; \n\t"
      "max.s32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return max(a + b, c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmax_s16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "add.s16x2 t1, %1, %2; \n\t"
      "max.s16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)max((short)(aS_lo + bS_lo), cS_lo);
  short ansS_hi = (short)max((short)(aS_hi + bS_hi), cS_hi);

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;  
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmax_u32(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .u32 t1; \n\t"
      "add.u32 t1, %1, %2; \n\t"
      "max.u32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return max(a + b, c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmax_u16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "add.u16x2 t1, %1, %2; \n\t"
      "max.u16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  // Get answer
  unsigned short ansU_lo = (unsigned short)max((unsigned short)(aU_lo + bU_lo), cU_lo);
  unsigned short ansU_hi = (unsigned short)max((unsigned short)(aU_hi + bU_hi), cU_hi);

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __viaddmin_s32(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "add.s32 t1, %1, %2; \n\t"
      "min.s32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return min(a + b, c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmin_s16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "add.s16x2 t1, %1, %2; \n\t"
      "min.s16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)min((short)(aS_lo + bS_lo), cS_lo);
  short ansS_hi = (short)min((short)(aS_hi + bS_hi), cS_hi);

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmin_u32(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .u32 t1; \n\t"
      "add.u32 t1, %1, %2; \n\t"
      "min.u32 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  return min(a + b, c);
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmin_u16x2(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "add.u16x2 t1, %1, %2; \n\t"
      "min.u16x2 %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  // Get answer
  unsigned short ansU_lo = (unsigned short)min((unsigned short)(aU_lo + bU_lo), cU_lo);
  unsigned short ansU_hi = (unsigned short)min((unsigned short)(aU_hi + bU_hi), cU_hi);

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __viaddmax_s32_relu(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "add.s32 t1, %1, %2; \n\t"
      "max.s32.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  int ans = max(a + b, c);

  return (ans > 0) ? ans : 0;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmax_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "add.s16x2 t1, %1, %2; \n\t"
      "max.s16x2.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)max((short)(aS_lo + bS_lo), cS_lo);
  short ansS_hi = (short)max((short)(aS_hi + bS_hi), cS_hi);

  if(ansS_lo < 0){ansS_lo = 0;}
  if(ansS_hi < 0){ansS_hi = 0;}

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __viaddmin_s32_relu(const int a, const int b, const int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int res;
  asm ("{.reg .s32 t1; \n\t"
      "add.s32 t1, %1, %2; \n\t"
      "min.s32.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  int ans = min(a + b, c);

  return (ans > 0) ? ans : 0;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __viaddmin_s16x2_relu(const unsigned int a, const unsigned int b, const unsigned int c){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int res;
  asm ("{.reg .b32 t1; \n\t"
      "add.s16x2 t1, %1, %2; \n\t"
      "min.s16x2.relu %0, t1, %3;}\n\t"
      : "=r"(res) : "r"(a), "r"(b), "r"(c));
  return res;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  unsigned short cU_lo = (unsigned short)(c & 0xFFFFU);
  unsigned short cU_hi = (unsigned short)(c >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  short cS_lo = *(short*)& cU_lo;
  short cS_hi = *(short*)& cU_hi;

  // Get answer
  short ansS_lo = (short)min((short)(aS_lo + bS_lo), cS_lo);
  short ansS_hi = (short)min((short)(aS_hi + bS_hi), cS_hi);

  if(ansS_lo < 0){ansS_lo = 0;}
  if(ansS_hi < 0){ansS_hi = 0;}

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;
#endif
}

// vimax vimin with predicate
// *pred gets set to '(a >= b)'
__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vibmax_s32(const int a, const int b, bool* const pred){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int val;
  unsigned int predicate_local;
  asm ("{ .reg .pred __$temp1;\n\t"
      "  setp.ge.s32  __$temp1, %2, %3;\n\t"
      "  selp.s32 %0, %2, %3, __$temp1;\n\t"
      "  selp.s32 %1, 1, 0, __$temp1;}\n\t"
      : "=r"(val), "=r"(predicate_local) : "r"(a), "r"(b));

  *pred = (bool)predicate_local;
  return val;
#else
  // Host and older architecture code
  int ans = max(a, b);

  *pred = (a >= b);
  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vibmax_u32(const unsigned int a, const unsigned int b, bool* const pred){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int val;
  unsigned int predicate_local;
  asm ("{ .reg .pred __$temp1;\n\t"
      "  setp.ge.u32  __$temp1, %2, %3;\n\t"
      "  selp.u32 %0, %2, %3, __$temp1;\n\t"
      "  selp.u32 %1, 1, 0, __$temp1;}\n\t"
      : "=r"(val), "=r"(predicate_local) : "r"(a), "r"(b));

  *pred = (bool)predicate_local;
  return val;
#else
  // Host and older architecture code
  unsigned int ans = max(a, b);

  *pred = (a >= b);
  return ans;
#endif
}

// *pred gets set to '(a <= b)'
__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ int __vibmin_s32(const int a, const int b, bool* const pred){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  int val;
  unsigned int predicate_local;
  asm ("{ .reg .pred __$temp1;\n\t"
      "  setp.le.s32  __$temp1, %2, %3;\n\t"
      "  selp.s32 %0, %2, %3, __$temp1;\n\t"
      "  selp.s32 %1, 1, 0, __$temp1;}\n\t"
      : "=r"(val), "=r"(predicate_local) : "r"(a), "r"(b));

  *pred = (bool)predicate_local;
  return val;
#else
  // Host and older architecture code
  int ans = min(a, b);

  *pred = (a <= b);
  return ans;
#endif
}

// *pred gets set to '(a <= b)'
__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vibmin_u32(const unsigned int a, const unsigned int b, bool* const pred){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int val;
  unsigned int predicate_local;
  asm ("{ .reg .pred __$temp1;\n\t"
      "  setp.le.u32  __$temp1, %2, %3;\n\t"
      "  selp.u32 %0, %2, %3, __$temp1;\n\t"
      "  selp.u32 %1, 1, 0, __$temp1;}\n\t"
      : "=r"(val), "=r"(predicate_local) : "r"(a), "r"(b));

  *pred = (bool)predicate_local;
  return val;
#else
  // Host and older architecture code
  unsigned int ans = min(a, b);

  *pred = (a <= b);
  return ans;
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vibmax_s16x2(const unsigned int a, const unsigned int b, bool* const pred_hi, bool* const pred_lo){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int val;
  unsigned int predicate_local_hi;
  unsigned int predicate_local_lo;
  asm ("{.reg .pred pu, pv; \n\t"
      ".reg .s16 rs0, rs1, rs2, rs3; \n\t"
      "max.s16x2 %0, %3, %4; \n\t"
      "mov.b32 {rs0, rs1}, %0; \n\t"
      "mov.b32 {rs2, rs3}, %3; \n\t"
      "setp.eq.s16 pv, rs0, rs2; \n\t"
      "setp.eq.s16 pu, rs1, rs3; \n\t"
      "selp.b32 %1, 1, 0, pu; \n\t"
      "selp.b32 %2, 1, 0, pv;} \n\t"
      : "=r"(val), "=r"(predicate_local_hi),"=r"(predicate_local_lo) : "r"(a), "r"(b));

  *pred_hi = (bool)predicate_local_hi;
  *pred_lo = (bool)predicate_local_lo;
  return val;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  // Get answer
  short ansS_lo = (short)max(aS_lo, bS_lo);
  short ansS_hi = (short)max(aS_hi, bS_hi);

  *pred_hi = (aS_hi >= bS_hi);
  *pred_lo = (aS_lo >= bS_lo);

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;  
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vibmax_u16x2(const unsigned int a, const unsigned int b, bool* const pred_hi, bool* const pred_lo){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int val;
  unsigned int predicate_local_hi;
  unsigned int predicate_local_lo;
  asm ("{.reg .pred pu, pv; \n\t"
      ".reg .u16 rs0, rs1, rs2, rs3; \n\t"
      "max.u16x2 %0, %3, %4; \n\t"
      "mov.b32 {rs0, rs1}, %0; \n\t"
      "mov.b32 {rs2, rs3}, %3; \n\t"
      "setp.eq.u16 pv, rs0, rs2; \n\t"
      "setp.eq.u16 pu, rs1, rs3; \n\t"
      "selp.b32 %1, 1, 0, pu; \n\t"
      "selp.b32 %2, 1, 0, pv;} \n\t"
      : "=r"(val), "=r"(predicate_local_hi),"=r"(predicate_local_lo) : "r"(a), "r"(b));

  *pred_hi = (bool)predicate_local_hi;
  *pred_lo = (bool)predicate_local_lo;
  return val;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  // Get answer
  unsigned short ansU_lo = (unsigned short)max(aU_lo, bU_lo);
  unsigned short ansU_hi = (unsigned short)max(aU_hi, bU_hi);

  *pred_hi = (aU_hi >= bU_hi);
  *pred_lo = (aU_lo >= bU_lo);

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;  
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vibmin_s16x2(const unsigned int a, const unsigned int b, bool* const pred_hi, bool* const pred_lo){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int val;
  unsigned int predicate_local_hi;
  unsigned int predicate_local_lo;
  asm ("{.reg .pred pu, pv; \n\t"
      ".reg .u16 rs0, rs1, rs2, rs3; \n\t"
      "min.s16x2 %0, %3, %4; \n\t"
      "mov.b32 {rs0, rs1}, %0; \n\t"
      "mov.b32 {rs2, rs3}, %3; \n\t"
      "setp.eq.s16 pv, rs0, rs2; \n\t"
      "setp.eq.s16 pu, rs1, rs3; \n\t"
      "selp.b32 %1, 1, 0, pu; \n\t"
      "selp.b32 %2, 1, 0, pv;} \n\t"
      : "=r"(val), "=r"(predicate_local_hi),"=r"(predicate_local_lo) : "r"(a), "r"(b));

  *pred_hi = (bool)predicate_local_hi;
  *pred_lo = (bool)predicate_local_lo;
  return val;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  //cast to signed:
  short aS_lo = *(short*)& aU_lo;
  short aS_hi = *(short*)& aU_hi;

  short bS_lo = *(short*)& bU_lo;
  short bS_hi = *(short*)& bU_hi;

  // Get answer
  short ansS_lo = (short)min(aS_lo, bS_lo);
  short ansS_hi = (short)min(aS_hi, bS_hi);

  *pred_hi = (aS_hi <= bS_hi);
  *pred_lo = (aS_lo <= bS_lo);

  // Cast back to unsigned:
  unsigned short ansU_lo = *(unsigned short*)& ansS_lo;
  unsigned short ansU_hi = *(unsigned short*)& ansS_hi;

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;  
#endif
}

__DEVICE_HOST_FUNCTIONS_STATIC_DECL__ unsigned int __vibmin_u16x2(const unsigned int a, const unsigned int b, bool* const pred_hi, bool* const pred_lo){
#ifdef __CUDA_AND_AT_LEAST_SM_90__
  unsigned int val;
  unsigned int predicate_local_hi;
  unsigned int predicate_local_lo;
  asm ("{.reg .pred pu, pv; \n\t"
      ".reg .u16 rs0, rs1, rs2, rs3; \n\t"
      "min.u16x2 %0, %3, %4; \n\t"
      "mov.b32 {rs0, rs1}, %0; \n\t"
      "mov.b32 {rs2, rs3}, %3; \n\t"
      "setp.eq.u16 pv, rs0, rs2; \n\t"
      "setp.eq.u16 pu, rs1, rs3; \n\t"
      "selp.b32 %1, 1, 0, pu; \n\t"
      "selp.b32 %2, 1, 0, pv;} \n\t"
      : "=r"(val), "=r"(predicate_local_hi),"=r"(predicate_local_lo) : "r"(a), "r"(b));

  *pred_hi = (bool)predicate_local_hi;
  *pred_lo = (bool)predicate_local_lo;
  return val;
#else
  // Host and older architecture code
  // Separate our high and low bit:
  unsigned short aU_lo = (unsigned short)(a & 0xFFFFU);
  unsigned short aU_hi = (unsigned short)(a >> 16);

  unsigned short bU_lo = (unsigned short)(b & 0xFFFFU);
  unsigned short bU_hi = (unsigned short)(b >> 16);

  // Get answer
  unsigned short ansU_lo = (unsigned short)min(aU_lo, bU_lo);
  unsigned short ansU_hi = (unsigned short)min(aU_hi, bU_hi);

  *pred_hi = (aU_hi <= bU_hi);
  *pred_lo = (aU_lo <= bU_lo);

  // Put answer back together:
  unsigned int ans = ((unsigned int) ansU_lo) | (((unsigned int) ansU_hi) << 16);

  return ans;  
#endif
}

#ifdef __CUDA_AND_AT_LEAST_SM_90__
#undef __CUDA_AND_AT_LEAST_SM_90__
#endif

#undef __DEVICE_HOST_FUNCTIONS_STATIC_DECL__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#endif /* !__DEVICE_FUNCTIONS_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_FUNCTIONS_HPP__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DEVICE_FUNCTIONS_HPP__
#endif
