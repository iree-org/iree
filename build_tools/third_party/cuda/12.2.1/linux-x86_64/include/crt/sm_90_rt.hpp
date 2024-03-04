/*
 * Copyright 2022 NVIDIA Corporation.  All rights reserved.
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
#pragma message("crt/sm_90_rt.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead.")
#else
#warning "crt/sm_90_rt.hpp is an internal header file and must not be used directly.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_90_RT_HPP__
#endif

#if !defined(__SM_90_RT_HPP__)
#define __SM_90_RT_HPP__

#if defined(__CUDACC_RTC__)
#define __SM_90_RT_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_90_RT_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

/*******************************************************************************
*                                                                              *
*  Below are implementations of SM-9.0 builtin functions which are included as *
*  source (instead of being built in to the compiler)                          *
*                                                                              *
*******************************************************************************/
extern "C" {
  __device__ unsigned  __nv_isClusterShared_impl(const void *);
  __device__ void * __nv_cluster_map_shared_rank_impl(const void *, unsigned);
  __device__ unsigned __nv_cluster_query_shared_rank_impl(const void *);
  __device__ unsigned __nv_clusterDimIsSpecifed_impl();
  __device__ void __nv_clusterDim_impl(unsigned *, unsigned *, unsigned *);
  __device__ void __nv_clusterRelativeBlockIdx_impl(unsigned *, 
                                                    unsigned *, unsigned *);
  __device__ void __nv_clusterGridDimInClusters_impl(unsigned *, 
                                                     unsigned *, unsigned *);
  __device__ void __nv_clusterIdx_impl(unsigned *, unsigned *, unsigned *);
  __device__ unsigned __nv_clusterRelativeBlockRank_impl();
  __device__ unsigned __nv_clusterSizeInBlocks_impl();
  __device__ void __nv_cluster_barrier_arrive_impl();
  __device__ void __nv_cluster_barrier_arrive_relaxed_impl();
  __device__ void __nv_cluster_barrier_wait_impl();
  __device__ void __nv_threadfence_cluster_impl();

  __device__ __device_builtin__ float2 __f2AtomicAdd(float2 *, float2);
  __device__ __device_builtin__ float2 __f2AtomicAdd_block(float2 *, float2);
  __device__ __device_builtin__ float2 __f2AtomicAdd_system(float2 *, float2);
  __device__ __device_builtin__ float4 __f4AtomicAdd(float4 *, float4);
  __device__ __device_builtin__ float4 __f4AtomicAdd_block(float4 *, float4);
  __device__ __device_builtin__ float4 __f4AtomicAdd_system(float4 *, float4);
} // extern "C"

__SM_90_RT_DECL__  unsigned __isCtaShared(const void *ptr) 
{
  return __isShared(ptr);
}

__SM_90_RT_DECL__ unsigned __isClusterShared(const void *ptr) 
{
  return __nv_isClusterShared_impl(ptr);
}

__SM_90_RT_DECL__ void *__cluster_map_shared_rank(const void *ptr, 
                                                  unsigned target_block_rank)
{
  return __nv_cluster_map_shared_rank_impl(ptr, target_block_rank);
}

__SM_90_RT_DECL__ unsigned __cluster_query_shared_rank(const void *ptr)
{
  return __nv_cluster_query_shared_rank_impl(ptr);
}

__SM_90_RT_DECL__ uint2 __cluster_map_shared_multicast(const void *ptr, 
                                                 unsigned int cluster_cta_mask)
{
  return make_uint2((unsigned)__cvta_generic_to_shared(ptr), cluster_cta_mask);
}

__SM_90_RT_DECL__ unsigned __clusterDimIsSpecified()
{
  return __nv_clusterDimIsSpecifed_impl();
}  

__SM_90_RT_DECL__ dim3 __clusterDim()
{
  unsigned x, y, z;
  __nv_clusterDim_impl(&x, &y, &z);
  return dim3(x,y,z);
}

__SM_90_RT_DECL__ dim3 __clusterRelativeBlockIdx()
{
  unsigned x, y, z;
  __nv_clusterRelativeBlockIdx_impl(&x, &y, &z);
  return dim3(x,y,z);
}

__SM_90_RT_DECL__ dim3 __clusterGridDimInClusters()
{
  unsigned x, y, z;
  __nv_clusterGridDimInClusters_impl(&x, &y, &z);
  return dim3(x,y,z);
}

__SM_90_RT_DECL__ dim3 __clusterIdx()
{
  unsigned x, y, z;
  __nv_clusterIdx_impl(&x, &y, &z);
  return dim3(x,y,z);
}

__SM_90_RT_DECL__ unsigned __clusterRelativeBlockRank()
{
  return __nv_clusterRelativeBlockRank_impl();
}

__SM_90_RT_DECL__ unsigned __clusterSizeInBlocks()
{
  return __nv_clusterSizeInBlocks_impl();
}

__SM_90_RT_DECL__ void __cluster_barrier_arrive()
{
  __nv_cluster_barrier_arrive_impl();
}

__SM_90_RT_DECL__ void __cluster_barrier_arrive_relaxed()
{
  __nv_cluster_barrier_arrive_relaxed_impl();
}

__SM_90_RT_DECL__ void __cluster_barrier_wait()
{
  __nv_cluster_barrier_wait_impl();
}

__SM_90_RT_DECL__ void __threadfence_cluster()
{
  __nv_threadfence_cluster_impl();
}


/* Define __PTR for atomicAdd prototypes below, undef after done */
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/

__SM_90_RT_DECL__ float2 atomicAdd(float2 *address, float2 val) {
  return __f2AtomicAdd(address, val);
}

__SM_90_RT_DECL__ float2 atomicAdd_block(float2 *address, float2 val) {
  return __f2AtomicAdd_block(address, val);
}

__SM_90_RT_DECL__ float2 atomicAdd_system(float2 *address, float2 val) {
  return __f2AtomicAdd_system(address, val);
}

__SM_90_RT_DECL__ float4 atomicAdd(float4 *address, float4 val) {
  return __f4AtomicAdd(address, val);
}

__SM_90_RT_DECL__ float4 atomicAdd_block(float4 *address, float4 val) {
  return __f4AtomicAdd_block(address, val);
}

__SM_90_RT_DECL__ float4 atomicAdd_system(float4 *address, float4 val) {
  return __f4AtomicAdd_system(address, val);
}

#endif /* !__CUDA_ARCH__ || __CUDA_ARCH__ >= 900 */

#endif /* __cplusplus && __CUDACC__ */

#undef __SM_90_RT_DECL__

#endif /* !__SM_90_RT_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_90_RT_HPP__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_SM_90_RT_HPP__
#endif
