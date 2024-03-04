/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__DEVICE_ATOMIC_FUNCTIONS_H__)
#define __DEVICE_ATOMIC_FUNCTIONS_H__

#if defined(__CUDACC_RTC__)
#define __DEVICE_ATOMIC_FUNCTIONS_DECL__ __device__
#elif defined(_NVHPC_CUDA)
# define __DEVICE_ATOMIC_FUNCTIONS_DECL__ extern __device__ __cudart_builtin__
#else /* __CUDACC_RTC__ */
#define __DEVICE_ATOMIC_FUNCTIONS_DECL__ static __inline__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

/* Add !defined(_NVHPC_CUDA) to avoid empty function definition in PGI CUDA
 * C++ compiler where the macro __CUDA_ARCH__ is not defined. */
#if !defined(__CUDA_ARCH__) && !defined(_NVHPC_CUDA)
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicAdd(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicAdd(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicSub(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicSub(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicExch(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicExch(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ float atomicExch(float *address, float val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicMin(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicMin(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicMax(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicMax(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicInc(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicDec(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicAnd(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicAnd(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicOr(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicOr(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicXor(int *address, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicXor(unsigned int *address, unsigned int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ int atomicCAS(int *address, int compare, int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val) __DEF_IF_HOST

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

#if defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is not valid on compute_70 and above, and should be replaced with "#x"_sync()."\
    "To continue using "#x"(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."
#elif defined(_NVHPC_CUDA)
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is not valid on cc70 and above, and should be replaced with "#x"_sync()."
#else
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is deprecated in favor of "#x"_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."
#endif

extern "C"
{
extern __device__ __device_builtin__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__any)) int __any(int cond);
extern __device__ __device_builtin__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__all)) int __all(int cond);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned long long int atomicExch(unsigned long long int *address, unsigned long long int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ unsigned long long int atomicCAS(unsigned long long int *address, unsigned long long int compare, unsigned long long int val) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__any)) bool any(bool cond) __DEF_IF_HOST

__DEVICE_ATOMIC_FUNCTIONS_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__all)) bool all(bool cond) __DEF_IF_HOST

#undef __DEPRECATED__
#undef __WSB_DEPRECATION_MESSAGE

#endif /* __cplusplus && __CUDACC__ */

#undef __DEF_IF_HOST
#undef __DEVICE_ATOMIC_FUNCTIONS_DECL__

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
#include "device_atomic_functions.hpp"
#endif /* !__CUDACC_RTC__ && defined(__CUDA_ARCH__) */

#endif /* !__DEVICE_ATOMIC_FUNCTIONS_H__ */
