/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUDA_DEVICE_RUNTIME_API_H__)
#define __CUDA_DEVICE_RUNTIME_API_H__

#if defined(__CUDACC__) && !defined(__CUDACC_RTC__)
#include <stdlib.h>
#endif

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if !defined(CUDA_FORCE_CDP1_IF_SUPPORTED) && !defined(__CUDADEVRT_INTERNAL__) && !defined(_NVHPC_CUDA) && !(defined(_WIN32) && !defined(_WIN64))
#define __CUDA_INTERNAL_USE_CDP2
#endif

#if !defined(__CUDACC_RTC__)

#if !defined(__CUDACC_INTERNAL_NO_STUBS__) && !defined(__CUDACC_RDC__) && !defined(__CUDACC_EWP__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350) && !defined(__CUDADEVRT_INTERNAL__)

#if defined(__cplusplus)
extern "C" {
#endif

struct cudaFuncAttributes;


#ifndef __CUDA_INTERNAL_USE_CDP2
inline __device__  cudaError_t CUDARTAPI cudaMalloc(void **p, size_t s)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
  return cudaErrorUnknown;
}
#else // __CUDA_INTERNAL_USE_CDP2
inline __device__  cudaError_t CUDARTAPI __cudaCDP2Malloc(void **p, size_t s)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI __cudaCDP2FuncGetAttributes(struct cudaFuncAttributes *p, const void *c)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI __cudaCDP2DeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI __cudaCDP2GetDevice(int *device)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

inline __device__  cudaError_t CUDARTAPI __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
  return cudaErrorUnknown;
}
#endif // __CUDA_INTERNAL_USE_CDP2


#if defined(__cplusplus)
}
#endif

#endif /* !defined(__CUDACC_INTERNAL_NO_STUBS__) && !defined(__CUDACC_RDC__) &&  !defined(__CUDACC_EWP__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350) && !defined(__CUDADEVRT_INTERNAL__) */

#endif /* !defined(__CUDACC_RTC__) */

#if defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
# define __DEPRECATED__(msg)
#elif defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(__CUDA_ARCH__) && !defined(__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING)
# define __CDPRT_DEPRECATED(func_name) __DEPRECATED__("Use of "#func_name" from device code is deprecated. Moreover, such use will cause this module to fail to load on sm_90+ devices. If calls to "#func_name" from device code cannot be removed for older devices at this time, you may guard them with __CUDA_ARCH__ macros to remove them only for sm_90+ devices, making sure to generate code for compute_90 for the macros to take effect. Note that this mitigation will no longer work when support for "#func_name" from device code is eventually dropped for all devices. Disable this warning with -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING.")
#else
# define __CDPRT_DEPRECATED(func_name)
#endif

#if defined(__cplusplus) && defined(__CUDACC__)         /* Visible to nvcc front-end only */
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)   // Visible to SM>=3.5 and "__host__ __device__" only

#include "driver_types.h"
#include "crt/host_defines.h"

#define cudaStreamGraphTailLaunch             (cudaStream_t)0x0100000000000000
#define cudaStreamGraphFireAndForget          (cudaStream_t)0x0200000000000000
#define cudaStreamGraphFireAndForgetAsSibling (cudaStream_t)0x0300000000000000

#ifdef __CUDA_INTERNAL_USE_CDP2
#define cudaStreamTailLaunch                ((cudaStream_t)0x3) /**< Per-grid stream with a fire-and-forget synchronization behavior. Only applicable when used with CUDA Dynamic Parallelism. */
#define cudaStreamFireAndForget             ((cudaStream_t)0x4) /**< Per-grid stream with a tail launch semantics. Only applicable when used with CUDA Dynamic Parallelism. */
#endif

extern "C"
{

// Symbols beginning with __cudaCDP* should not be used outside
// this header file. Instead, compile with -DCUDA_FORCE_CDP1_IF_SUPPORTED if
// CDP1 support is required.

extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaDeviceSynchronizeDeprecationAvoidance(void);

#ifndef __CUDA_INTERNAL_USE_CDP2
//// CDP1 endpoints
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
#if (__CUDA_ARCH__ < 900) && (defined(CUDA_FORCE_CDP1_IF_SUPPORTED) || (defined(_WIN32) && !defined(_WIN64)))
// cudaDeviceSynchronize is removed on sm_90+
extern __device__ __cudart_builtin__ __CDPRT_DEPRECATED(cudaDeviceSynchronize) cudaError_t CUDARTAPI cudaDeviceSynchronize(void);
#endif
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void);
extern __device__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error);
extern __device__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync_ptsz(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync_ptsz(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags);
#endif // __CUDA_INTERNAL_USE_CDP2

//// CDP2 endpoints
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2DeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2DeviceGetLimit(size_t *pValue, enum cudaLimit limit);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2DeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2DeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2GetLastError(void);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2PeekAtLastError(void);
extern __device__ __cudart_builtin__ const char* CUDARTAPI __cudaCDP2GetErrorString(cudaError_t error);
extern __device__ __cudart_builtin__ const char* CUDARTAPI __cudaCDP2GetErrorName(cudaError_t error);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2GetDeviceCount(int *count);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2GetDevice(int *device);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2StreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2StreamDestroy(cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2StreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2StreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2EventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2EventRecord(cudaEvent_t event, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2EventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2EventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2EventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2EventDestroy(cudaEvent_t event);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2FuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Free(void *devPtr);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Malloc(void **devPtr, size_t size);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2MemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2MemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2MemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2MemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memset2DAsync_ptsz(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2Memset3DAsync_ptsz(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2RuntimeGetVersion(int *runtimeVersion);
extern __device__ __cudart_builtin__ void * CUDARTAPI __cudaCDP2GetParameterBuffer(size_t alignment, size_t size);
extern __device__ __cudart_builtin__ void * CUDARTAPI __cudaCDP2GetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2LaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2LaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2LaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2LaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags);


extern  __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) 
static inline  __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGraphLaunch_ptsz(cudaGraphExec_t graphExec, cudaStream_t stream)
{
    if (stream == 0) {
        stream = cudaStreamPerThread;
    }
    return  cudaGraphLaunch(graphExec, stream);
}
#endif

/**
  * \ingroup CUDART_GRAPH
  * \brief Get the currently running device graph id.
  *
  * Get the currently running device graph id.
  * \return Returns the current device graph id, 0 if the call is outside of a device graph.
  * \sa cudaLaunchDevice
  */
static inline __device__ __cudart_builtin__ cudaGraphExec_t CUDARTAPI cudaGetCurrentGraphExec(void)
{
    unsigned long long current_graph_exec;
    asm ("mov.u64 %0, %%current_graph_exec;" : "=l"(current_graph_exec));
    return (cudaGraphExec_t)current_graph_exec;
}

/**
  * \ingroup CUDART_EXECUTION
  * \brief Programmatic dependency trigger
  *
  * This device function ensures the programmatic launch completion edges /
  * events are fulfilled. See
  * ::cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticStreamSerialization
  * and ::cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticEvent for more
  * information. The event / edge kick off only happens when every CTAs
  * in the grid has either exited or called this function at least once,
  * otherwise the kick off happens automatically after all warps finishes
  * execution but before the grid completes. The kick off only enables
  * scheduling of the secondary kernel. It provides no memory visibility
  * guarantee itself. The user could enforce memory visibility by inserting a
  * memory fence of the correct scope.
  */
static inline __device__ __cudart_builtin__ void CUDARTAPI cudaTriggerProgrammaticLaunchCompletion(void)
{
    asm volatile("griddepcontrol.launch_dependents;":::);
}

/**
  * \ingroup CUDART_EXECUTION
  * \brief Programmatic grid dependency synchronization
  *
  * This device function will block the thread until all direct grid
  * dependencies have completed. This API is intended to use in conjuncture with
  * programmatic / launch event / dependency. See
  * ::cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticStreamSerialization
  * and ::cudaLaunchAttributeID::cudaLaunchAttributeProgrammaticEvent for more
  * information.
  */
static inline __device__ __cudart_builtin__ void CUDARTAPI cudaGridDependencySynchronize(void)
{
    asm volatile("griddepcontrol.wait;":::"memory");
}


//// CG API
extern __device__ __cudart_builtin__ unsigned long long CUDARTAPI cudaCGGetIntrinsicHandle(enum cudaCGScope scope);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGSynchronize(unsigned long long handle, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGSynchronizeGrid(unsigned long long handle, unsigned int flags);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGGetSize(unsigned int *numThreads, unsigned int *numGrids, unsigned long long handle);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaCGGetRank(unsigned int *threadRank, unsigned int *gridRank, unsigned long long handle);


//// CDP API

#ifdef __CUDA_ARCH__

#ifdef __CUDA_INTERNAL_USE_CDP2
static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    return __cudaCDP2DeviceGetAttribute(value, attr, device);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit)
{
    return __cudaCDP2DeviceGetLimit(pValue, limit);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    return __cudaCDP2DeviceGetCacheConfig(pCacheConfig);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig)
{
    return __cudaCDP2DeviceGetSharedMemConfig(pConfig);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
    return __cudaCDP2GetLastError();
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void)
{
    return __cudaCDP2PeekAtLastError();
}

static __inline__ __device__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
{
    return __cudaCDP2GetErrorString(error);
}

static __inline__ __device__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error)
{
    return __cudaCDP2GetErrorName(error);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    return __cudaCDP2GetDeviceCount(count);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    return __cudaCDP2GetDevice(device);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    return __cudaCDP2StreamCreateWithFlags(pStream, flags);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    return __cudaCDP2StreamDestroy(stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return __cudaCDP2StreamWaitEvent(stream, event, flags);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return __cudaCDP2StreamWaitEvent_ptsz(stream, event, flags);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    return __cudaCDP2EventCreateWithFlags(event, flags);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return __cudaCDP2EventRecord(event, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream)
{
    return __cudaCDP2EventRecord_ptsz(event, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    return __cudaCDP2EventRecordWithFlags(event, stream, flags);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    return __cudaCDP2EventRecordWithFlags_ptsz(event, stream, flags);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
    return __cudaCDP2EventDestroy(event);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    return __cudaCDP2FuncGetAttributes(attr, func);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    return __cudaCDP2Free(devPtr);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    return __cudaCDP2Malloc(devPtr, size);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return __cudaCDP2MemcpyAsync(dst, src, count, kind, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return __cudaCDP2MemcpyAsync_ptsz(dst, src, count, kind, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return __cudaCDP2Memcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync_ptsz(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    return __cudaCDP2Memcpy2DAsync_ptsz(dst, dpitch, src, spitch, width, height, kind, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream)
{
    return __cudaCDP2Memcpy3DAsync(p, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream)
{
    return __cudaCDP2Memcpy3DAsync_ptsz(p, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    return __cudaCDP2MemsetAsync(devPtr, value, count, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    return __cudaCDP2MemsetAsync_ptsz(devPtr, value, count, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return __cudaCDP2Memset2DAsync(devPtr, pitch, value, width, height, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync_ptsz(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return __cudaCDP2Memset2DAsync_ptsz(devPtr, pitch, value, width, height, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return __cudaCDP2Memset3DAsync(pitchedDevPtr, value, extent, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync_ptsz(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return __cudaCDP2Memset3DAsync_ptsz(pitchedDevPtr, value, extent, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
{
    return __cudaCDP2RuntimeGetVersion(runtimeVersion);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
    return __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSmemSize);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
    return __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSmemSize, flags);
}
#endif // __CUDA_INTERNAL_USE_CDP2

#endif // __CUDA_ARCH__


/**
 * \ingroup CUDART_EXECUTION
 * \brief Obtains a parameter buffer
 *
 * Obtains a parameter buffer which can be filled with parameters for a kernel launch.
 * Parameters passed to ::cudaLaunchDevice must be allocated via this function.
 *
 * This is a low level API and can only be accessed from Parallel Thread Execution (PTX).
 * CUDA user code should use <<< >>> to launch kernels.
 *
 * \param alignment - Specifies alignment requirement of the parameter buffer
 * \param size      - Specifies size requirement in bytes
 *
 * \return
 * Returns pointer to the allocated parameterBuffer
 * \notefnerr
 *
 * \sa cudaLaunchDevice
 */
#ifdef __CUDA_INTERNAL_USE_CDP2
static __inline__ __device__ __cudart_builtin__ void * CUDARTAPI cudaGetParameterBuffer(size_t alignment, size_t size)
{
    return __cudaCDP2GetParameterBuffer(alignment, size);
}
#else
extern __device__ __cudart_builtin__ void * CUDARTAPI cudaGetParameterBuffer(size_t alignment, size_t size);
#endif


/**
 * \ingroup CUDART_EXECUTION
 * \brief Launches a specified kernel
 *
 * Launches a specified kernel with the specified parameter buffer. A parameter buffer can be obtained
 * by calling ::cudaGetParameterBuffer().
 *
 * This is a low level API and can only be accessed from Parallel Thread Execution (PTX).
 * CUDA user code should use <<< >>> to launch the kernels.
 *
 * \param func            - Pointer to the kernel to be launched
 * \param parameterBuffer - Holds the parameters to the launched kernel. parameterBuffer can be NULL. (Optional)
 * \param gridDimension   - Specifies grid dimensions
 * \param blockDimension  - Specifies block dimensions
 * \param sharedMemSize   - Specifies size of shared memory
 * \param stream          - Specifies the stream to be used
 *
 * \return
 * ::cudaSuccess, ::cudaErrorInvalidDevice, ::cudaErrorLaunchMaxDepthExceeded, ::cudaErrorInvalidConfiguration,
 * ::cudaErrorStartupFailure, ::cudaErrorLaunchPendingCountExceeded, ::cudaErrorLaunchOutOfResources
 * \notefnerr
 * \n Please refer to Execution Configuration and Parameter Buffer Layout from the CUDA Programming
 * Guide for the detailed descriptions of launch configuration and parameter layout respectively.
 *
 * \sa cudaGetParameterBuffer
 */
#ifdef __CUDA_INTERNAL_USE_CDP2
static __inline__ __device__ __cudart_builtin__ void * CUDARTAPI cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize)
{
    return __cudaCDP2GetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize);
}
#else
extern __device__ __cudart_builtin__ void * CUDARTAPI cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize);
#endif


#ifdef __CUDA_INTERNAL_USE_CDP2
static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream)
{
    return __cudaCDP2LaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream);
}

static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream)
{
    return __cudaCDP2LaunchDeviceV2_ptsz(parameterBuffer, stream);
}
#else
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice_ptsz(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDeviceV2_ptsz(void *parameterBuffer, cudaStream_t stream);
#endif


#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) && defined(__CUDA_ARCH__)
    // When compiling for the device and per thread default stream is enabled, add
    // a static inline redirect to the per thread stream entry points.

    static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI
    cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream)
    {
#ifdef __CUDA_INTERNAL_USE_CDP2
        return __cudaCDP2LaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream);
#else
        return cudaLaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream);
#endif
    }

    static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI
    cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
    {
#ifdef __CUDA_INTERNAL_USE_CDP2
        return __cudaCDP2LaunchDeviceV2_ptsz(parameterBuffer, stream);
#else
        return cudaLaunchDeviceV2_ptsz(parameterBuffer, stream);
#endif
    }
#else // defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) && defined(__CUDA_ARCH__)
#ifdef __CUDA_INTERNAL_USE_CDP2
    static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream)
    {
        return __cudaCDP2LaunchDevice(func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream);
    }

    static __inline__ __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
    {
        return __cudaCDP2LaunchDeviceV2(parameterBuffer, stream);
    }
#else // __CUDA_INTERNAL_USE_CDP2
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDevice(void *func, void *parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned int sharedMemSize, cudaStream_t stream);
extern __device__ __cudart_builtin__ cudaError_t CUDARTAPI cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream);
#endif // __CUDA_INTERNAL_USE_CDP2
#endif // defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) && defined(__CUDA_ARCH__)


// These symbols should not be used outside of this header file.
#define __cudaCDP2DeviceGetAttribute
#define __cudaCDP2DeviceGetLimit
#define __cudaCDP2DeviceGetCacheConfig
#define __cudaCDP2DeviceGetSharedMemConfig
#define __cudaCDP2GetLastError
#define __cudaCDP2PeekAtLastError
#define __cudaCDP2GetErrorString
#define __cudaCDP2GetErrorName
#define __cudaCDP2GetDeviceCount
#define __cudaCDP2GetDevice
#define __cudaCDP2StreamCreateWithFlags
#define __cudaCDP2StreamDestroy
#define __cudaCDP2StreamWaitEvent
#define __cudaCDP2StreamWaitEvent_ptsz
#define __cudaCDP2EventCreateWithFlags
#define __cudaCDP2EventRecord
#define __cudaCDP2EventRecord_ptsz
#define __cudaCDP2EventRecordWithFlags
#define __cudaCDP2EventRecordWithFlags_ptsz
#define __cudaCDP2EventDestroy
#define __cudaCDP2FuncGetAttributes
#define __cudaCDP2Free
#define __cudaCDP2Malloc
#define __cudaCDP2MemcpyAsync
#define __cudaCDP2MemcpyAsync_ptsz
#define __cudaCDP2Memcpy2DAsync
#define __cudaCDP2Memcpy2DAsync_ptsz
#define __cudaCDP2Memcpy3DAsync
#define __cudaCDP2Memcpy3DAsync_ptsz
#define __cudaCDP2MemsetAsync
#define __cudaCDP2MemsetAsync_ptsz
#define __cudaCDP2Memset2DAsync
#define __cudaCDP2Memset2DAsync_ptsz
#define __cudaCDP2Memset3DAsync
#define __cudaCDP2Memset3DAsync_ptsz
#define __cudaCDP2RuntimeGetVersion
#define __cudaCDP2GetParameterBuffer
#define __cudaCDP2GetParameterBufferV2
#define __cudaCDP2LaunchDevice_ptsz
#define __cudaCDP2LaunchDeviceV2_ptsz
#define __cudaCDP2LaunchDevice
#define __cudaCDP2LaunchDeviceV2
#define __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor
#define __cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags

}

template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaMalloc(T **devPtr, size_t size);
template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, T *entry);
template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize);
template <typename T> static __inline__ __device__ __cudart_builtin__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned int flags);


#endif // !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
#endif /* defined(__cplusplus) && defined(__CUDACC__) */

#undef __DEPRECATED__
#undef __CDPRT_DEPRECATED
#undef __CUDA_INTERNAL_USE_CDP2

#endif /* !__CUDA_DEVICE_RUNTIME_API_H__ */
