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

#if !defined(__CUDA_RUNTIME_H__)
#define __CUDA_RUNTIME_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_H__
#endif

#define EXCLUDE_FROM_RTC
#if defined(__GNUC__)
#if defined(__clang__) || (!defined(__PGIC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
#pragma GCC diagnostic push
#endif
#if defined(__clang__) || (!defined(__PGIC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2)))
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4820)
#endif
#ifdef __QNX__
#if (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)
typedef unsigned size_t;
#endif
#endif
#undef EXCLUDE_FROM_RTC
/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "crt/host_config.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "library_types.h"
#if !defined(__CUDACC_RTC__)
#define EXCLUDE_FROM_RTC
#include "channel_descriptor.h"
#include "cuda_runtime_api.h"
#include "driver_functions.h"
#undef EXCLUDE_FROM_RTC
#endif /* !__CUDACC_RTC__ */
#include "crt/host_defines.h"
#ifdef __CUDACC_RTC__
#include "target"
#endif  /* defined(__CUDACC_RTC__) */


#include "vector_functions.h"

#if defined(__CUDACC__)

#if defined(__CUDACC_RTC__)
#include "nvrtc_device_runtime.h"
#include "crt/device_functions.h"
#include "crt/common_functions.h"
#include "device_launch_parameters.h"

#else /* !__CUDACC_RTC__ */
#define EXCLUDE_FROM_RTC
#include "crt/common_functions.h"
#include "crt/device_functions.h"
#include "device_launch_parameters.h"

#if defined(__CUDACC_EXTENDED_LAMBDA__)
#include <functional>
#include <utility>
struct  __device_builtin__ __nv_lambda_preheader_injection { };
#endif /* defined(__CUDACC_EXTENDED_LAMBDA__) */

#undef EXCLUDE_FROM_RTC
#endif /* __CUDACC_RTC__ */

#endif /* __CUDACC__ */

/** \cond impl_private */
#if defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif
/** \endcond impl_private */

#define EXCLUDE_FROM_RTC
#if defined(__cplusplus) && !defined(__CUDACC_RTC__)

#if __cplusplus >= 201103
#include <utility>
#endif

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * \addtogroup CUDART_HIGHLEVEL
 * @{
 */

/**
 *\brief Launches a device function
 *
 * The function invokes kernel \p func on \p gridDim (\p gridDim.x &times; \p gridDim.y
 * &times; \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x &times;
 * \p blockDim.y &times; \p blockDim.z) threads.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory (defaults to 0)
 * \param stream      - Stream identifier (defaults to NULL)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorSharedObjectInitFailed,
 * ::cudaErrorInvalidPtx,
 * ::cudaErrorUnsupportedPtxVersion,
 * ::cudaErrorNoKernelImageForDevice,
 * ::cudaErrorJitCompilerNotFound,
 * ::cudaErrorJitCompilationDisabled
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \ref ::cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C API)"
 */
template<class T>
static __inline__ __host__ cudaError_t cudaLaunchKernel(
  const T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
)
{
    return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
}


#if __cplusplus >= 201103 || defined(__DOXYGEN_ONLY__)
/**
 * \brief Launches a CUDA function with launch-time configuration
 *
 * Invokes the kernel \p func on \p config->gridDim (\p config->gridDim.x
 * &times; \p config->gridDim.y &times; \p config->gridDim.z) grid of blocks.
 * Each block contains \p config->blockDim (\p config->blockDim.x &times;
 * \p config->blockDim.y &times; \p config->blockDim.z) threads.
 *
 * \p config->dynamicSmemBytes sets the amount of dynamic shared memory that
 * will be available to each thread block.
 *
 * \p config->stream specifies a stream the invocation is associated to.
 *
 * Configuration beyond grid and block dimensions, dynamic shared memory size,
 * and stream can be provided with the following two fields of \p config:
 *
 * \p config->attrs is an array of \p config->numAttrs contiguous
 * ::cudaLaunchAttribute elements. The value of this pointer is not considered
 * if \p config->numAttrs is zero. However, in that case, it is recommended to
 * set the pointer to NULL.
 * \p config->numAttrs is the number of attributes populating the first
 * \p config->numAttrs positions of the \p config->attrs array.
 *
 * The kernel arguments should be passed as arguments to this function via the
 * \p args parameter pack.
 *
 * The C API version of this function, \p cudaLaunchKernelExC, is also available
 * for pre-C++11 compilers and for use cases where the ability to pass kernel
 * parameters via void* array is preferable.
 *
 * \param config - Launch configuration
 * \param func   - Kernel to launch
 * \param args   - Parameter pack of kernel parameters
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorSharedObjectInitFailed,
 * ::cudaErrorInvalidPtx,
 * ::cudaErrorUnsupportedPtxVersion,
 * ::cudaErrorNoKernelImageForDevice,
 * ::cudaErrorJitCompilerNotFound,
 * ::cudaErrorJitCompilationDisabled
 * \note_null_stream
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * \ref ::cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func, void **args) "cudaLaunchKernelEx (C API)",
 * ::cuLaunchKernelEx
 */
template<typename... ExpTypes, typename... ActTypes>
static __inline__ __host__ cudaError_t cudaLaunchKernelEx(
  const cudaLaunchConfig_t *config,
  void (*kernel)(ExpTypes...),
  ActTypes &&... args
)
{
    return [&](ExpTypes... coercedArgs){
        void *pArgs[] = { &coercedArgs... };
        return ::cudaLaunchKernelExC(config, (const void *)kernel, pArgs);
    }(std::forward<ActTypes>(args)...);
}
#endif

/**
 *\brief Launches a device function
 *
 * The function invokes kernel \p func on \p gridDim (\p gridDim.x &times; \p gridDim.y
 * &times; \p gridDim.z) grid of blocks. Each block contains \p blockDim (\p blockDim.x &times;
 * \p blockDim.y &times; \p blockDim.z) threads.
 *
 * The device on which this kernel is invoked must have a non-zero value for
 * the device attribute ::cudaDevAttrCooperativeLaunch.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
 *
 * The kernel cannot make use of CUDA dynamic parallelism.
 *
 * If the kernel has N parameters the \p args should point to array of N pointers.
 * Each pointer, from <tt>args[0]</tt> to <tt>args[N - 1]</tt>, point to the region
 * of memory from which the actual parameter will be copied.
 *
 * \p sharedMem sets the amount of dynamic shared memory that will be available to
 * each thread block.
 *
 * \p stream specifies a stream the invocation is associated to.
 *
 * \param func        - Device function symbol
 * \param gridDim     - Grid dimentions
 * \param blockDim    - Block dimentions
 * \param args        - Arguments
 * \param sharedMem   - Shared memory (defaults to 0)
 * \param stream      - Stream identifier (defaults to NULL)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidConfiguration,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorLaunchTimeout,
 * ::cudaErrorLaunchOutOfResources,
 * ::cudaErrorSharedObjectInitFailed
 * \notefnerr
 * \note_async
 * \note_null_stream
 * \note_init_rt
 * \note_callback
 *
 * \ref ::cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchCooperativeKernel (C API)"
 */
template<class T>
static __inline__ __host__ cudaError_t cudaLaunchCooperativeKernel(
  const T *func,
  dim3 gridDim,
  dim3 blockDim,
  void **args,
  size_t sharedMem = 0,
  cudaStream_t stream = 0
)
{
    return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream);
}

/**
 * \brief \hl Creates an event object with the specified flags
 *
 * Creates an event object with the specified flags. Valid flags include:
 * - ::cudaEventDefault: Default event creation flag.
 * - ::cudaEventBlockingSync: Specifies that event should use blocking
 *   synchronization. A host thread that uses ::cudaEventSynchronize() to wait
 *   on an event created with this flag will block until the event actually
 *   completes.
 * - ::cudaEventDisableTiming: Specifies that the created event does not need
 *   to record timing data.  Events created with this flag specified and
 *   the ::cudaEventBlockingSync flag not specified will provide the best
 *   performance when used with ::cudaStreamWaitEvent() and ::cudaEventQuery().
 *
 * \param event - Newly created event
 * \param flags - Flags for new event
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorLaunchFailure,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaEventCreate(cudaEvent_t*) "cudaEventCreate (C API)",
 * ::cudaEventCreateWithFlags, ::cudaEventRecord, ::cudaEventQuery,
 * ::cudaEventSynchronize, ::cudaEventDestroy, ::cudaEventElapsedTime,
 * ::cudaStreamWaitEvent
 */
static __inline__ __host__ cudaError_t cudaEventCreate(
  cudaEvent_t  *event,
  unsigned int  flags
)
{
  return ::cudaEventCreateWithFlags(event, flags);
}

/**
 * \brief Creates an executable graph from a graph
 *
 * Instantiates \p graph as an executable graph. The graph is validated for any
 * structural constraints or intra-node constraints which were not previously
 * validated. If instantiation is successful, a handle to the instantiated graph
 * is returned in \p pGraphExec.
 *
 * If there are any errors, diagnostic information may be returned in \p pErrorNode and
 * \p pLogBuffer. This is the primary way to inspect instantiation errors. The output
 * will be null terminated unless the diagnostics overflow
 * the buffer. In this case, they will be truncated, and the last byte can be
 * inspected to determine if truncation occurred.
 *
 * \param pGraphExec - Returns instantiated graph
 * \param graph      - Graph to instantiate
 * \param pErrorNode - In case of an instantiation error, this may be modified to
 *                      indicate a node contributing to the error
 * \param pLogBuffer   - A character buffer to store diagnostic messages
 * \param bufferSize  - Size of the log buffer in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphInstantiateWithFlags,
 * ::cudaGraphCreate,
 * ::cudaGraphUpload,
 * ::cudaGraphLaunch,
 * ::cudaGraphExecDestroy
 */
static __inline__ __host__ cudaError_t cudaGraphInstantiate(
  cudaGraphExec_t *pGraphExec,
  cudaGraph_t graph,
  cudaGraphNode_t *pErrorNode,
  char *pLogBuffer,
  size_t bufferSize
)
{
  (void)pErrorNode;
  (void)pLogBuffer;
  (void)bufferSize;
  return ::cudaGraphInstantiate(pGraphExec, graph, 0);
}

/**
 * \brief \hl Allocates page-locked memory on the host
 *
 * Allocates \p size bytes of host memory that is page-locked and accessible
 * to the device. The driver tracks the virtual memory ranges allocated with
 * this function and automatically accelerates calls to functions such as
 * ::cudaMemcpy(). Since the memory can be accessed directly by the device, it
 * can be read or written with much higher bandwidth than pageable memory
 * obtained with functions such as ::malloc(). Allocating excessive amounts of
 * pinned memory may degrade system performance, since it reduces the amount
 * of memory available to the system for paging. As a result, this function is
 * best used sparingly to allocate staging areas for data exchange between host
 * and device.
 *
 * The \p flags parameter enables different options to be specified that affect
 * the allocation, as follows.
 * - ::cudaHostAllocDefault: This flag's value is defined to be 0.
 * - ::cudaHostAllocPortable: The memory returned by this call will be
 * considered as pinned memory by all CUDA contexts, not just the one that
 * performed the allocation.
 * - ::cudaHostAllocMapped: Maps the allocation into the CUDA address space.
 * The device pointer to the memory may be obtained by calling
 * ::cudaHostGetDevicePointer().
 * - ::cudaHostAllocWriteCombined: Allocates the memory as write-combined (WC).
 * WC memory can be transferred across the PCI Express bus more quickly on some
 * system configurations, but cannot be read efficiently by most CPUs.  WC
 * memory is a good option for buffers that will be written by the CPU and read
 * by the device via mapped pinned memory or host->device transfers.
 *
 * All of these flags are orthogonal to one another: a developer may allocate
 * memory that is portable, mapped and/or write-combined with no restrictions.
 *
 * ::cudaSetDeviceFlags() must have been called with the ::cudaDeviceMapHost
 * flag in order for the ::cudaHostAllocMapped flag to have any effect.
 *
 * The ::cudaHostAllocMapped flag may be specified on CUDA contexts for devices
 * that do not support mapped pinned memory. The failure is deferred to
 * ::cudaHostGetDevicePointer() because the memory may be mapped into other
 * CUDA contexts via the ::cudaHostAllocPortable flag.
 *
 * Memory allocated by this function must be freed with ::cudaFreeHost().
 *
 * \param ptr   - Device pointer to allocated memory
 * \param size  - Requested allocation size in bytes
 * \param flags - Requested properties of allocated memory
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaSetDeviceFlags,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc
 */
static __inline__ __host__ cudaError_t cudaMallocHost(
  void         **ptr,
  size_t         size,
  unsigned int   flags
)
{
  return ::cudaHostAlloc(ptr, size, flags);
}

template<class T>
static __inline__ __host__ cudaError_t cudaHostAlloc(
  T            **ptr,
  size_t         size,
  unsigned int   flags
)
{
  return ::cudaHostAlloc((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __host__ cudaError_t cudaHostGetDevicePointer(
  T            **pDevice,
  void          *pHost,
  unsigned int   flags
)
{
  return ::cudaHostGetDevicePointer((void**)(void*)pDevice, pHost, flags);
}

/**
 * \brief Allocates memory that will be automatically managed by the Unified Memory system
 *
 * Allocates \p size bytes of managed memory on the device and returns in
 * \p *devPtr a pointer to the allocated memory. If the device doesn't support
 * allocating managed memory, ::cudaErrorNotSupported is returned. Support
 * for managed memory can be queried using the device attribute
 * ::cudaDevAttrManagedMemory. The allocated memory is suitably
 * aligned for any kind of variable. The memory is not cleared. If \p size
 * is 0, ::cudaMallocManaged returns ::cudaErrorInvalidValue. The pointer
 * is valid on the CPU and on all GPUs in the system that support managed memory.
 * All accesses to this pointer must obey the Unified Memory programming model.
 *
 * \p flags specifies the default stream association for this allocation.
 * \p flags must be one of ::cudaMemAttachGlobal or ::cudaMemAttachHost. The
 * default value for \p flags is ::cudaMemAttachGlobal.
 * If ::cudaMemAttachGlobal is specified, then this memory is accessible from
 * any stream on any device. If ::cudaMemAttachHost is specified, then the
 * allocation should not be accessed from devices that have a zero value for the
 * device attribute ::cudaDevAttrConcurrentManagedAccess; an explicit call to
 * ::cudaStreamAttachMemAsync will be required to enable access on such devices.
 *
 * If the association is later changed via ::cudaStreamAttachMemAsync to
 * a single stream, the default association, as specifed during ::cudaMallocManaged,
 * is restored when that stream is destroyed. For __managed__ variables, the
 * default association is always ::cudaMemAttachGlobal. Note that destroying a
 * stream is an asynchronous operation, and as a result, the change to default
 * association won't happen until all work in the stream has completed.
 *
 * Memory allocated with ::cudaMallocManaged should be released with ::cudaFree.
 *
 * Device memory oversubscription is possible for GPUs that have a non-zero value for the
 * device attribute ::cudaDevAttrConcurrentManagedAccess. Managed memory on
 * such GPUs may be evicted from device memory to host memory at any time by the Unified
 * Memory driver in order to make room for other allocations.
 *
 * In a multi-GPU system where all GPUs have a non-zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess, managed memory may not be populated when this
 * API returns and instead may be populated on access. In such systems, managed memory can
 * migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
 * maintain data locality and prevent excessive page faults to the extent possible. The application
 * can also guide the driver about memory usage patterns via ::cudaMemAdvise. The application
 * can also explicitly migrate memory to a desired processor's memory via
 * ::cudaMemPrefetchAsync.
 *
 * In a multi-GPU system where all of the GPUs have a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess and all the GPUs have peer-to-peer support
 * with each other, the physical storage for managed memory is created on the GPU which is active
 * at the time ::cudaMallocManaged is called. All other GPUs will reference the data at reduced
 * bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
 * memory among such GPUs.
 *
 * In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
 * where the value of the device attribute ::cudaDevAttrConcurrentManagedAccess
 * is zero for at least one of those GPUs, the location chosen for physical storage of managed
 * memory is system-dependent.
 * - On Linux, the location chosen will be device memory as long as the current set of active
 * contexts are on devices that either have peer-to-peer support with each other or have a
 * non-zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess.
 * If there is an active context on a GPU that does not have a non-zero value for that device
 * attribute and it does not have peer-to-peer support with the other devices that have active
 * contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
 * Note that this means that managed memory that is located in device memory is migrated to
 * host memory if a new context is created on a GPU that doesn't have a non-zero value for
 * the device attribute and does not support peer-to-peer with at least one of the other devices
 * that has an active context. This in turn implies that context creation may fail if there is
 * insufficient host memory to migrate all managed allocations.
 * - On Windows, the physical storage is always created in 'zero-copy' or host memory.
 * All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
 * circumstances, use of the environment variable CUDA_VISIBLE_DEVICES is recommended to
 * restrict CUDA to only use those GPUs that have peer-to-peer support.
 * Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
 * value to force the driver to always use device memory for physical storage.
 * When this environment variable is set to a non-zero value, all devices used in
 * that process that support managed memory have to be peer-to-peer compatible
 * with each other. The error ::cudaErrorInvalidDevice will be returned if a device
 * that supports managed memory is used and it is not peer-to-peer compatible with
 * any of the other managed memory supporting devices that were previously used in
 * that process, even if ::cudaDeviceReset has been called on those devices. These
 * environment variables are described in the CUDA programming guide under the
 * "CUDA environment variables" section.
 * - On ARM, managed memory is not available on discrete gpu with Drive PX-2.
 *
 * \param devPtr - Pointer to allocated device memory
 * \param size   - Requested allocation size in bytes
 * \param flags  - Must be either ::cudaMemAttachGlobal or ::cudaMemAttachHost (defaults to ::cudaMemAttachGlobal)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorMemoryAllocation,
 * ::cudaErrorNotSupported,
 * ::cudaErrorInvalidValue
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMallocPitch, ::cudaFree, ::cudaMallocArray, ::cudaFreeArray,
 * ::cudaMalloc3D, ::cudaMalloc3DArray,
 * \ref ::cudaMallocHost(void**, size_t) "cudaMallocHost (C API)",
 * ::cudaFreeHost, ::cudaHostAlloc, ::cudaDeviceGetAttribute, ::cudaStreamAttachMemAsync
 */
template<class T>
static __inline__ __host__ cudaError_t cudaMallocManaged(
  T            **devPtr,
  size_t         size,
  unsigned int   flags = cudaMemAttachGlobal
)
{
  return ::cudaMallocManaged((void**)(void*)devPtr, size, flags);
}

/**
 * \brief Advise about the usage of a given memory range.
 *
 * This is an alternate spelling for cudaMemAdvise made available through operator overloading.
 *
 * \sa ::cudaMemAdvise,
 * \ref ::cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, struct cudaMemLocation location)  "cudaMemAdvise (C API)"
 */
template<class T>
cudaError_t cudaMemAdvise(
  T                      *devPtr,
  size_t                 count,
  enum cudaMemoryAdvise  advice,
  struct cudaMemLocation location
)
{
  return ::cudaMemAdvise_v2((const void *)devPtr, count, advice, location);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMemPrefetchAsync(
  T                       *devPtr,
  size_t                  count,
  struct cudaMemLocation  location,
  unsigned int            flags,
  cudaStream_t            stream = 0
)
{
  return ::cudaMemPrefetchAsync_v2((const void *)devPtr, count, location, flags, stream);
}

/**
 * \brief Attach memory to a stream asynchronously
 *
 * Enqueues an operation in \p stream to specify stream association of
 * \p length bytes of memory starting from \p devPtr. This function is a
 * stream-ordered operation, meaning that it is dependent on, and will
 * only take effect when, previous work in stream has completed. Any
 * previous association is automatically replaced.
 *
 * \p devPtr must point to an one of the following types of memories:
 * - managed memory declared using the __managed__ keyword or allocated with
 *   ::cudaMallocManaged.
 * - a valid host-accessible region of system-allocated pageable memory. This
 *   type of memory may only be specified if the device associated with the
 *   stream reports a non-zero value for the device attribute
 *   ::cudaDevAttrPageableMemoryAccess.
 *
 * For managed allocations, \p length must be either zero or the entire
 * allocation's size. Both indicate that the entire allocation's stream
 * association is being changed. Currently, it is not possible to change stream
 * association for a portion of a managed allocation.
 *
 * For pageable allocations, \p length must be non-zero.
 *
 * The stream association is specified using \p flags which must be
 * one of ::cudaMemAttachGlobal, ::cudaMemAttachHost or ::cudaMemAttachSingle.
 * The default value for \p flags is ::cudaMemAttachSingle
 * If the ::cudaMemAttachGlobal flag is specified, the memory can be accessed
 * by any stream on any device.
 * If the ::cudaMemAttachHost flag is specified, the program makes a guarantee
 * that it won't access the memory on the device from any stream on a device that
 * has a zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess.
 * If the ::cudaMemAttachSingle flag is specified and \p stream is associated with
 * a device that has a zero value for the device attribute ::cudaDevAttrConcurrentManagedAccess,
 * the program makes a guarantee that it will only access the memory on the device
 * from \p stream. It is illegal to attach singly to the NULL stream, because the
 * NULL stream is a virtual global stream and not a specific stream. An error will
 * be returned in this case.
 *
 * When memory is associated with a single stream, the Unified Memory system will
 * allow CPU access to this memory region so long as all operations in \p stream
 * have completed, regardless of whether other streams are active. In effect,
 * this constrains exclusive ownership of the managed memory region by
 * an active GPU to per-stream activity instead of whole-GPU activity.
 *
 * Accessing memory on the device from streams that are not associated with
 * it will produce undefined results. No error checking is performed by the
 * Unified Memory system to ensure that kernels launched into other streams
 * do not access this region.
 *
 * It is a program's responsibility to order calls to ::cudaStreamAttachMemAsync
 * via events, synchronization or other means to ensure legal access to memory
 * at all times. Data visibility and coherency will be changed appropriately
 * for all kernels which follow a stream-association change.
 *
 * If \p stream is destroyed while data is associated with it, the association is
 * removed and the association reverts to the default visibility of the allocation
 * as specified at ::cudaMallocManaged. For __managed__ variables, the default
 * association is always ::cudaMemAttachGlobal. Note that destroying a stream is an
 * asynchronous operation, and as a result, the change to default association won't
 * happen until all work in the stream has completed.
 *
 * \param stream  - Stream in which to enqueue the attach operation
 * \param devPtr  - Pointer to memory (must be a pointer to managed memory or
 *                  to a valid host-accessible region of system-allocated
 *                  memory)
 * \param length  - Length of memory (defaults to zero)
 * \param flags   - Must be one of ::cudaMemAttachGlobal, ::cudaMemAttachHost or ::cudaMemAttachSingle (defaults to ::cudaMemAttachSingle)
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorNotReady,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidResourceHandle
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaStreamCreate, ::cudaStreamCreateWithFlags, ::cudaStreamWaitEvent, ::cudaStreamSynchronize, ::cudaStreamAddCallback, ::cudaStreamDestroy, ::cudaMallocManaged
 */
template<class T>
static __inline__ __host__ cudaError_t cudaStreamAttachMemAsync(
  cudaStream_t   stream,
  T              *devPtr,
  size_t         length = 0,
  unsigned int   flags  = cudaMemAttachSingle
)
{
  return ::cudaStreamAttachMemAsync(stream, (void*)devPtr, length, flags);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMalloc(
  T      **devPtr,
  size_t   size
)
{
  return ::cudaMalloc((void**)(void*)devPtr, size);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMallocHost(
  T            **ptr,
  size_t         size,
  unsigned int   flags = 0
)
{
  return cudaMallocHost((void**)(void*)ptr, size, flags);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMallocPitch(
  T      **devPtr,
  size_t  *pitch,
  size_t   width,
  size_t   height
)
{
  return ::cudaMallocPitch((void**)(void*)devPtr, pitch, width, height);
}

/**
 * \brief Allocate from a pool
 *
 * This is an alternate spelling for cudaMallocFromPoolAsync
 * made available through operator overloading.
 *
 * \sa ::cudaMallocFromPoolAsync,
 * \ref ::cudaMallocAsync(void** ptr, size_t size, cudaStream_t hStream)  "cudaMallocAsync (C API)"
 */
static __inline__ __host__ cudaError_t cudaMallocAsync(
  void        **ptr,
  size_t        size,
  cudaMemPool_t memPool,
  cudaStream_t  stream
)
{
  return ::cudaMallocFromPoolAsync(ptr, size, memPool, stream);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMallocAsync(
  T           **ptr,
  size_t        size,
  cudaMemPool_t memPool,
  cudaStream_t  stream
)
{
  return ::cudaMallocFromPoolAsync((void**)(void*)ptr, size, memPool, stream);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMallocAsync(
  T           **ptr,
  size_t        size,
  cudaStream_t  stream
)
{
  return ::cudaMallocAsync((void**)(void*)ptr, size, stream);
}

template<class T>
static __inline__ __host__ cudaError_t cudaMallocFromPoolAsync(
  T           **ptr,
  size_t        size,
  cudaMemPool_t memPool,
  cudaStream_t  stream
)
{
  return ::cudaMallocFromPoolAsync((void**)(void*)ptr, size, memPool, stream);
}

#if defined(__CUDACC__)

/**
 * \brief \hl Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToDevice.
 *
 * \param symbol - Device symbol reference
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
template<class T>
static __inline__ __host__ cudaError_t cudaMemcpyToSymbol(
  const T                   &symbol,
  const void                *src,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyHostToDevice
)
{
  return ::cudaMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}

/**
 * \brief \hl Copies data to the given symbol on the device
 *
 * Copies \p count bytes from the memory area pointed to by \p src
 * to the memory area \p offset bytes from the start of symbol
 * \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice or ::cudaMemcpyDeviceToDevice.
 *
 * ::cudaMemcpyToSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally
 * be associated to a stream by passing a non-zero \p stream argument. If
 * \p kind is ::cudaMemcpyHostToDevice and \p stream is non-zero, the copy
 * may overlap with operations in other streams.
 *
 * \param symbol - Device symbol reference
 * \param src    - Source memory address
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyFromSymbolAsync
 */
template<class T>
static __inline__ __host__ cudaError_t cudaMemcpyToSymbolAsync(
  const T                   &symbol,
  const void                *src,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyHostToDevice,
        cudaStream_t         stream = 0
)
{
  return ::cudaMemcpyToSymbolAsync((const void*)&symbol, src, count, offset, kind, stream);
}

/**
 * \brief \hl Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost or ::cudaMemcpyDeviceToDevice.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol reference
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_sync
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync, ::cudaMemcpyFromSymbolAsync
 */
template<class T>
static __inline__ __host__ cudaError_t cudaMemcpyFromSymbol(
        void                *dst,
  const T                   &symbol,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyDeviceToHost
)
{
  return ::cudaMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}

/**
 * \brief \hl Copies data from the given symbol on the device
 *
 * Copies \p count bytes from the memory area \p offset bytes
 * from the start of symbol \p symbol to the memory area pointed to by \p dst.
 * The memory areas may not overlap. \p symbol is a variable that resides in
 * global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost or ::cudaMemcpyDeviceToDevice.
 *
 * ::cudaMemcpyFromSymbolAsync() is asynchronous with respect to the host, so
 * the call may return before the copy is complete. The copy can optionally be
 * associated to a stream by passing a non-zero \p stream argument. If \p kind
 * is ::cudaMemcpyDeviceToHost and \p stream is non-zero, the copy may overlap
 * with operations in other streams.
 *
 * \param dst    - Destination memory address
 * \param symbol - Device symbol reference
 * \param count  - Size in bytes to copy
 * \param offset - Offset from start of symbol in bytes
 * \param kind   - Type of transfer
 * \param stream - Stream identifier
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorInvalidMemcpyDirection,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_async
 * \note_string_api_deprecation
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaMemcpy, ::cudaMemcpy2D,
 * ::cudaMemcpy2DToArray, ::cudaMemcpy2DFromArray,
 * ::cudaMemcpy2DArrayToArray, ::cudaMemcpyToSymbol,
 * ::cudaMemcpyFromSymbol, ::cudaMemcpyAsync, ::cudaMemcpy2DAsync,
 * ::cudaMemcpy2DToArrayAsync,
 * ::cudaMemcpy2DFromArrayAsync,
 * ::cudaMemcpyToSymbolAsync
 */
template<class T>
static __inline__ __host__ cudaError_t cudaMemcpyFromSymbolAsync(
        void                *dst,
  const T                   &symbol,
        size_t               count,
        size_t               offset = 0,
        enum cudaMemcpyKind  kind   = cudaMemcpyDeviceToHost,
        cudaStream_t         stream = 0
)
{
  return ::cudaMemcpyFromSymbolAsync(dst, (const void*)&symbol, count, offset, kind, stream);
}

/**
 * \brief Creates a memcpy node to copy to a symbol on the device and adds it to a graph
 *
 * Creates a new memcpy node to copy to \p symbol and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p src to the memory area pointed to by \p offset bytes from the start
 * of symbol \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::cudaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyToSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeFromSymbol,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemsetNode
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGraphAddMemcpyNodeToSymbol(
    cudaGraphNode_t *pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies,
    size_t numDependencies,
    const T &symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void*)&symbol, src, count, offset, kind);
}

/**
 * \brief Creates a memcpy node to copy from a symbol on the device and adds it to a graph
 *
 * Creates a new memcpy node to copy from \p symbol and adds it to \p graph with
 * \p numDependencies dependencies specified via \p pDependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p pDependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p pGraphNode.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p offset bytes from the start of symbol \p symbol to the memory area
 *  pointed to by \p dst. The memory areas may not overlap. \p symbol is a variable
 *  that resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * Memcpy nodes have some additional restrictions with regards to managed memory, if the
 * system contains at least one device which has a zero value for the device attribute
 * ::cudaDevAttrConcurrentManagedAccess.
 *
 * \param pGraphNode      - Returns newly created node
 * \param graph           - Graph to which to add the node
 * \param pDependencies   - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyFromSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeToSymbol,
 * ::cudaGraphMemcpyNodeGetParams,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphCreate,
 * ::cudaGraphDestroyNode,
 * ::cudaGraphAddChildGraphNode,
 * ::cudaGraphAddEmptyNode,
 * ::cudaGraphAddKernelNode,
 * ::cudaGraphAddHostNode,
 * ::cudaGraphAddMemsetNode
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGraphAddMemcpyNodeFromSymbol(
    cudaGraphNode_t* pGraphNode,
    cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies,
    size_t numDependencies,
    void* dst,
    const T &symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void*)&symbol, count, offset, kind);
}

/**
 * \brief Sets a memcpy node's parameters to copy to a symbol on the device
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p src to the memory area pointed to by \p offset bytes from the start
 * of symbol \p symbol. The memory areas may not overlap. \p symbol is a variable that
 * resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyHostToDevice, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of
 * transfer is inferred from the pointer values. However, ::cudaMemcpyDefault
 * is only allowed on systems that support unified virtual addressing.
 *
 * \param node            - Node to set the parameters for
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyToSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(
    cudaGraphNode_t node,
    const T &symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void*)&symbol, src, count, offset, kind);
}

/**
 * \brief Sets a memcpy node's parameters to copy from a symbol on the device
 *
 * Sets the parameters of memcpy node \p node to the copy described by the provided parameters.
 *
 * When the graph is launched, the node will copy \p count bytes from the memory area
 * pointed to by \p offset bytes from the start of symbol \p symbol to the memory area
 *  pointed to by \p dst. The memory areas may not overlap. \p symbol is a variable
 *  that resides in global or constant memory space. \p kind can be either
 * ::cudaMemcpyDeviceToHost, ::cudaMemcpyDeviceToDevice, or ::cudaMemcpyDefault.
 * Passing ::cudaMemcpyDefault is recommended, in which case the type of transfer
 * is inferred from the pointer values. However, ::cudaMemcpyDefault is only
 * allowed on systems that support unified virtual addressing.
 *
 * \param node            - Node to set the parameters for
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaMemcpyFromSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphMemcpyNodeGetParams
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(
    cudaGraphNode_t node,
    void* dst,
    const T &symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void*)&symbol, count, offset, kind);
}

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the device
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p src and \p symbol must be allocated from the same contexts as the original source and
 * destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::cudaErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param symbol          - Device symbol address
 * \param src             - Source memory address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeToSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphInstantiate,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    const T &symbol,
    const void* src,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
    return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void*)&symbol, src, count, offset, kind);
}

/**
 * \brief Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the device
 *
 * Updates the work represented by \p node in \p hGraphExec as though \p node had 
 * contained the given params at instantiation.  \p node must remain in the graph which was 
 * used to instantiate \p hGraphExec.  Changed edges to and from \p node are ignored.
 *
 * \p symbol and \p dst must be allocated from the same contexts as the original source and
 * destination memory.  The instantiation-time memory operands must be 1-dimensional.
 * Zero-length operations are not supported.
 *
 * The modifications only affect future launches of \p hGraphExec.  Already enqueued 
 * or running launches of \p hGraphExec are not affected by this call.  \p node is also 
 * not modified by this call.
 *
 * Returns ::cudaErrorInvalidValue if the memory operands' mappings changed or
 * the original memory operands are multidimensional.
 *
 * \param hGraphExec      - The executable graph in which to set the specified node
 * \param node            - Memcpy node from the graph which was used to instantiate graphExec
 * \param dst             - Destination memory address
 * \param symbol          - Device symbol address
 * \param count           - Size in bytes to copy
 * \param offset          - Offset from start of symbol in bytes
 * \param kind            - Type of transfer
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 * \note_graph_thread_safety
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaGraphAddMemcpyNode,
 * ::cudaGraphAddMemcpyNodeFromSymbol,
 * ::cudaGraphMemcpyNodeSetParams,
 * ::cudaGraphMemcpyNodeSetParamsFromSymbol,
 * ::cudaGraphInstantiate,
 * ::cudaGraphExecMemcpyNodeSetParams,
 * ::cudaGraphExecMemcpyNodeSetParamsToSymbol,
 * ::cudaGraphExecKernelNodeSetParams,
 * ::cudaGraphExecMemsetNodeSetParams,
 * ::cudaGraphExecHostNodeSetParams
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(
    cudaGraphExec_t hGraphExec,
    cudaGraphNode_t node,
    void* dst,
    const T &symbol,
    size_t count,
    size_t offset,
    enum cudaMemcpyKind kind)
{
  return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void*)&symbol, count, offset, kind);
}

// convenience function to avoid source breakage in c++ code
static __inline__ __host__ cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out)
{
    cudaGraphExecUpdateResultInfo resultInfo;
    cudaError_t status = cudaGraphExecUpdate(hGraphExec, hGraph, &resultInfo);
    if (hErrorNode_out) {
        *hErrorNode_out = resultInfo.errorNode;
    }
    if (updateResult_out) {
        *updateResult_out = resultInfo.result;
    }
    return status;
}

#if __cplusplus >= 201103

/**
 * \brief Creates a user object by wrapping a C++ object
 *
 * TODO detail
 *
 * \param object_out      - Location to return the user object handle
 * \param objectToWrap    - This becomes the \ptr argument to ::cudaUserObjectCreate. A
 *                          lambda will be passed for the \p destroy argument, which calls
 *                          delete on this object pointer.
 * \param initialRefcount - The initial refcount to create the object with, typically 1. The
 *                          initial references are owned by the calling thread.
 * \param flags           - Currently it is required to pass cudaUserObjectNoDestructorSync,
 *                          which is the only defined flag. This indicates that the destroy
 *                          callback cannot be waited on by any CUDA API. Users requiring
 *                          synchronization of the callback should signal its completion
 *                          manually.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidValue
 *
 * \sa
 * ::cudaUserObjectCreate
 */
template<class T>
static __inline__ __host__ cudaError_t cudaUserObjectCreate(
    cudaUserObject_t *object_out,
    T *objectToWrap,
    unsigned int initialRefcount,
    unsigned int flags)
{
    return ::cudaUserObjectCreate(
            object_out,
            objectToWrap,
            [](void *vpObj) { delete reinterpret_cast<T *>(vpObj); },
            initialRefcount,
            flags);
}

template<class T>
static __inline__ __host__ cudaError_t cudaUserObjectCreate(
    cudaUserObject_t *object_out,
    T *objectToWrap,
    unsigned int initialRefcount,
    cudaUserObjectFlags flags)
{
    return cudaUserObjectCreate(object_out, objectToWrap, initialRefcount, (unsigned int)flags);
}

#endif

/**
 * \brief \hl Finds the address associated with a CUDA symbol
 *
 * Returns in \p *devPtr the address of symbol \p symbol on the device.
 * \p symbol can either be a variable that resides in global or constant memory space.
 * If \p symbol cannot be found, or if \p symbol is not declared
 * in the global or constant memory space, \p *devPtr is unchanged and the error
 * ::cudaErrorInvalidSymbol is returned.
 *
 * \param devPtr - Return device pointer associated with symbol
 * \param symbol - Device symbol reference
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaGetSymbolAddress(void**, const void*) "cudaGetSymbolAddress (C API)",
 * \ref ::cudaGetSymbolSize(size_t*, const T&) "cudaGetSymbolSize (C++ API)"
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGetSymbolAddress(
        void **devPtr,
  const T     &symbol
)
{
  return ::cudaGetSymbolAddress(devPtr, (const void*)&symbol);
}

/**
 * \brief \hl Finds the size of the object associated with a CUDA symbol
 *
 * Returns in \p *size the size of symbol \p symbol. \p symbol must be a
 * variable that resides in global or constant memory space.
 * If \p symbol cannot be found, or if \p symbol is not declared
 * in global or constant memory space, \p *size is unchanged and the error
 * ::cudaErrorInvalidSymbol is returned.
 *
 * \param size   - Size of object associated with symbol
 * \param symbol - Device symbol reference
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidSymbol,
 * ::cudaErrorNoKernelImageForDevice
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa \ref ::cudaGetSymbolAddress(void**, const T&) "cudaGetSymbolAddress (C++ API)",
 * \ref ::cudaGetSymbolSize(size_t*, const void*) "cudaGetSymbolSize (C API)"
 */
template<class T>
static __inline__ __host__ cudaError_t cudaGetSymbolSize(
        size_t *size,
  const T      &symbol
)
{
  return ::cudaGetSymbolSize(size, (const void*)&symbol);
}

/**
 * \brief \hl Sets the preferred cache configuration for a device function
 *
 * On devices where the L1 cache and shared memory use the same hardware
 * resources, this sets through \p cacheConfig the preferred cache configuration
 * for the function specified via \p func. This is only a preference. The
 * runtime will use the requested configuration if possible, but it is free to
 * choose a different configuration if required to execute \p func.
 *
 * \p func must be a pointer to a function that executes on the device.
 * The parameter specified by \p func must be declared as a \p __global__
 * function. If the specified function does not exist,
 * then ::cudaErrorInvalidDeviceFunction is returned.
 *
 * This setting does nothing on devices where the size of the L1 cache and
 * shared memory are fixed.
 *
 * Launching a kernel with a different preference than the most recent
 * preference setting may insert a device-side synchronization point.
 *
 * The supported cache configurations are:
 * - ::cudaFuncCachePreferNone: no preference for shared memory or L1 (default)
 * - ::cudaFuncCachePreferShared: prefer larger shared memory and smaller L1 cache
 * - ::cudaFuncCachePreferL1: prefer larger L1 cache and smaller shared memory
 *
 * \param func        - device function pointer
 * \param cacheConfig - Requested cache configuration
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(const void*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, T*) "cudaFuncGetAttributes (C++ API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * ::cudaThreadGetCacheConfig,
 * ::cudaThreadSetCacheConfig
 */
template<class T>
static __inline__ __host__ cudaError_t cudaFuncSetCacheConfig(
  T                  *func,
  enum cudaFuncCache  cacheConfig
)
{
  return ::cudaFuncSetCacheConfig((const void*)func, cacheConfig);
}

template<class T>
static __inline__ __host__ cudaError_t cudaFuncSetSharedMemConfig(
  T                        *func,
  enum cudaSharedMemConfig  config
)
{
  return ::cudaFuncSetSharedMemConfig((const void*)func, config);
}

#endif // __CUDACC__

/**
 * \brief Returns occupancy for a device function
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calulated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSize
 * \sa ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 * \sa ::cudaOccupancyAvailableDynamicSMemPerBlock
 */
template<class T>
static __inline__ __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int   *numBlocks,
    T      func,
    int    blockSize,
    size_t dynamicSMemSize)
{
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, cudaOccupancyDefault);
}

/**
 * \brief Returns occupancy for a device function with the specified flags
 *
 * Returns in \p *numBlocks the maximum number of active blocks per
 * streaming multiprocessor for the device function.
 *
 * The \p flags parameter controls how special cases are handled. Valid flags include:
 *
 * - ::cudaOccupancyDefault: keeps the default behavior as
 *   ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *
 * - ::cudaOccupancyDisableCachingOverride: suppresses the default behavior
 *   on platform where global caching affects occupancy. On such platforms, if caching
 *   is enabled, but per-block SM resource usage would result in zero occupancy, the
 *   occupancy calculator will calculate the occupancy as if caching is disabled.
 *   Setting this flag makes the occupancy calculator to return 0 in such cases.
 *   More information can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param numBlocks       - Returned occupancy
 * \param func            - Kernel function for which occupancy is calulated
 * \param blockSize       - Block size the kernel is intended to be launched with
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param flags           - Requested behavior for the occupancy calculator
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * \sa ::cudaOccupancyMaxPotentialBlockSize
 * \sa ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 * \sa ::cudaOccupancyAvailableDynamicSMemPerBlock
 */
template<class T>
static __inline__ __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int         *numBlocks,
    T            func,
    int          blockSize,
    size_t       dynamicSMemSize,
    unsigned int flags)
{
    return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, flags);
}

/**
 * Helper functor for cudaOccupancyMaxPotentialBlockSize
 */
class __cudaOccupancyB2DHelper {
  size_t n;
public:
  inline __host__ CUDART_DEVICE __cudaOccupancyB2DHelper(size_t n_) : n(n_) {}
  inline __host__ CUDART_DEVICE size_t operator()(int)
  {
      return n;
  }
};

/**
 * \brief Returns grid and block size that achieves maximum potential occupancy for a device function
 *
 * Returns in \p *minGridSize and \p *blocksize a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps with the smallest number
 * of blocks).
 *
 * The \p flags parameter controls how special cases are handled. Valid flags include:
 *
 * - ::cudaOccupancyDefault: keeps the default behavior as
 *   ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 *
 * - ::cudaOccupancyDisableCachingOverride: This flag suppresses the default behavior
 *   on platform where global caching affects occupancy. On such platforms, if caching
 *   is enabled, but per-block SM resource usage would result in zero occupancy, the
 *   occupancy calculator will calculate the occupancy as if caching is disabled.
 *   Setting this flag makes the occupancy calculator to return 0 in such cases.
 *   More information can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the best potential occupancy
 * \param blockSize   - Returned block size
 * \param func        - Device function symbol
 * \param blockSizeToDynamicSMemSize - A unary function / functor that takes block size, and returns the size, in bytes, of dynamic shared memory needed for a block
 * \param blockSizeLimit  - The maximum block size \p func is designed to work with. 0 means no limit.
 * \param flags       - Requested behavior for the occupancy calculator
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSize
 * \sa ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 * \sa ::cudaOccupancyAvailableDynamicSMemPerBlock
 */

template<typename UnaryFunction, class T>
static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
    int           *minGridSize,
    int           *blockSize,
    T              func,
    UnaryFunction  blockSizeToDynamicSMemSize,
    int            blockSizeLimit = 0,
    unsigned int   flags = 0)
{
    cudaError_t status;

    // Device and function properties
    int                       device;
    struct cudaFuncAttributes attr;

    // Limits
    int maxThreadsPerMultiProcessor;
    int warpSize;
    int devMaxThreadsPerBlock;
    int multiProcessorCount;
    int funcMaxThreadsPerBlock;
    int occupancyLimit;
    int granularity;

    // Recorded maximum
    int maxBlockSize = 0;
    int numBlocks    = 0;
    int maxOccupancy = 0;

    // Temporary
    int blockSizeToTryAligned;
    int blockSizeToTry;
    int blockSizeLimitAligned;
    int occupancyInBlocks;
    int occupancyInThreads;
    size_t dynamicSMemSize;

    ///////////////////////////
    // Check user input
    ///////////////////////////

    if (!minGridSize || !blockSize || !func) {
        return cudaErrorInvalidValue;
    }

    //////////////////////////////////////////////
    // Obtain device and function properties
    //////////////////////////////////////////////

    status = ::cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &maxThreadsPerMultiProcessor,
        cudaDevAttrMaxThreadsPerMultiProcessor,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &warpSize,
        cudaDevAttrWarpSize,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &devMaxThreadsPerBlock,
        cudaDevAttrMaxThreadsPerBlock,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaDeviceGetAttribute(
        &multiProcessorCount,
        cudaDevAttrMultiProcessorCount,
        device);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaFuncGetAttributes(&attr, func);
    if (status != cudaSuccess) {
        return status;
    }
    
    funcMaxThreadsPerBlock = attr.maxThreadsPerBlock;

    /////////////////////////////////////////////////////////////////////////////////
    // Try each block size, and pick the block size with maximum occupancy
    /////////////////////////////////////////////////////////////////////////////////

    occupancyLimit = maxThreadsPerMultiProcessor;
    granularity    = warpSize;

    if (blockSizeLimit == 0) {
        blockSizeLimit = devMaxThreadsPerBlock;
    }

    if (devMaxThreadsPerBlock < blockSizeLimit) {
        blockSizeLimit = devMaxThreadsPerBlock;
    }

    if (funcMaxThreadsPerBlock < blockSizeLimit) {
        blockSizeLimit = funcMaxThreadsPerBlock;
    }

    blockSizeLimitAligned = ((blockSizeLimit + (granularity - 1)) / granularity) * granularity;

    for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
        // This is needed for the first iteration, because
        // blockSizeLimitAligned could be greater than blockSizeLimit
        //
        if (blockSizeLimit < blockSizeToTryAligned) {
            blockSizeToTry = blockSizeLimit;
        } else {
            blockSizeToTry = blockSizeToTryAligned;
        }
        
        dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);

        status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &occupancyInBlocks,
            func,
            blockSizeToTry,
            dynamicSMemSize,
            flags);

        if (status != cudaSuccess) {
            return status;
        }

        occupancyInThreads = blockSizeToTry * occupancyInBlocks;

        if (occupancyInThreads > maxOccupancy) {
            maxBlockSize = blockSizeToTry;
            numBlocks    = occupancyInBlocks;
            maxOccupancy = occupancyInThreads;
        }

        // Early out if we have reached the maximum
        //
        if (occupancyLimit == maxOccupancy) {
            break;
        }
    }

    ///////////////////////////
    // Return best available
    ///////////////////////////

    // Suggested min grid size to achieve a full machine launch
    //
    *minGridSize = numBlocks * multiProcessorCount;
    *blockSize = maxBlockSize;

    return status;
}

/**
 * \brief Returns grid and block size that achieves maximum potential occupancy for a device function
 *
 * Returns in \p *minGridSize and \p *blocksize a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps with the smallest number
 * of blocks).
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the best potential occupancy
 * \param blockSize   - Returned block size
 * \param func        - Device function symbol
 * \param blockSizeToDynamicSMemSize - A unary function / functor that takes block size, and returns the size, in bytes, of dynamic shared memory needed for a block
 * \param blockSizeLimit  - The maximum block size \p func is designed to work with. 0 means no limit.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSize
 * \sa ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 * \sa ::cudaOccupancyAvailableDynamicSMemPerBlock
 */

template<typename UnaryFunction, class T>
static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    int           *minGridSize,
    int           *blockSize,
    T              func,
    UnaryFunction  blockSizeToDynamicSMemSize,
    int            blockSizeLimit = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, cudaOccupancyDefault);
}

/**
 * \brief Returns grid and block size that achieves maximum potential occupancy for a device function
 *
 * Returns in \p *minGridSize and \p *blocksize a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps with the smallest number
 * of blocks).
 *
 * Use \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem if the
 * amount of per-block dynamic shared memory changes with different
 * block sizes.
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the best potential occupancy
 * \param blockSize   - Returned block size
 * \param func        - Device function symbol
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param blockSizeLimit  - The maximum block size \p func is designed to work with. 0 means no limit.
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 * \sa ::cudaOccupancyAvailableDynamicSMemPerBlock
 */
template<class T>
static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int    *minGridSize,
    int    *blockSize,
    T       func,
    size_t  dynamicSMemSize = 0,
    int     blockSizeLimit = 0)
{
  return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit, cudaOccupancyDefault);
}

/**
 * \brief Returns dynamic shared memory available per block when launching \p numBlocks blocks on SM.
 *
 * Returns in \p *dynamicSmemSize the maximum size of dynamic shared memory to allow \p numBlocks blocks per SM. 
 *
 * \param dynamicSmemSize - Returned maximum dynamic shared memory 
 * \param func            - Kernel function for which occupancy is calculated
 * \param numBlocks       - Number of blocks to fit on SM 
 * \param blockSize       - Size of the block
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxPotentialBlockSize
 * \sa ::cudaOccupancyMaxPotentialBlockSizeWithFlags
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 */
template<class T>
static __inline__ __host__ cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(
    size_t *dynamicSmemSize,
    T      func,
    int    numBlocks,
    int    blockSize)
{
    return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void*)func, numBlocks, blockSize);
}

/**
 * \brief Returns grid and block size that achived maximum potential occupancy for a device function with the specified flags
 *
 * Returns in \p *minGridSize and \p *blocksize a suggested grid /
 * block size pair that achieves the best potential occupancy
 * (i.e. the maximum number of active warps with the smallest number
 * of blocks).
 *
 * The \p flags parameter controls how special cases are handle. Valid flags include:
 *
 * - ::cudaOccupancyDefault: keeps the default behavior as
 *   ::cudaOccupancyMaxPotentialBlockSize
 *
 * - ::cudaOccupancyDisableCachingOverride: This flag suppresses the default behavior
 *   on platform where global caching affects occupancy. On such platforms, if caching
 *   is enabled, but per-block SM resource usage would result in zero occupancy, the
 *   occupancy calculator will calculate the occupancy as if caching is disabled.
 *   Setting this flag makes the occupancy calculator to return 0 in such cases.
 *   More information can be found about this feature in the "Unified L1/Texture Cache"
 *   section of the Maxwell tuning guide.
 *
 * Use \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem if the
 * amount of per-block dynamic shared memory changes with different
 * block sizes.
 *
 * \param minGridSize - Returned minimum grid size needed to achieve the best potential occupancy
 * \param blockSize   - Returned block size
 * \param func        - Device function symbol
 * \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 * \param blockSizeLimit  - The maximum block size \p func is designed to work with. 0 means no limit.
 * \param flags       - Requested behavior for the occupancy calculator
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDevice,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa ::cudaOccupancyMaxPotentialBlockSize
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * \sa ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMem
 * \sa ::cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags
 * \sa ::cudaOccupancyAvailableDynamicSMemPerBlock
 */
template<class T>
static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(
    int    *minGridSize,
    int    *blockSize,
    T      func,
    size_t dynamicSMemSize = 0,
    int    blockSizeLimit = 0,
    unsigned int flags = 0)
{
    return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, __cudaOccupancyB2DHelper(dynamicSMemSize), blockSizeLimit, flags);
}

/**
 * \brief Given the kernel function (\p func) and launch configuration
 * (\p config), return the maximum cluster size in \p *clusterSize.
 *
 * The cluster dimensions in \p config are ignored. If func has a required
 * cluster size set (see ::cudaFuncGetAttributes),\p *clusterSize will reflect 
 * the required cluster size.
 *
 * By default this function will always return a value that's portable on
 * future hardware. A higher value may be returned if the kernel function
 * allows non-portable cluster sizes.
 *
 * This function will respect the compile time launch bounds.
 *
 * \param clusterSize - Returned maximum cluster size that can be launched
 *                      for the given kernel function and launch configuration
 * \param func        - Kernel function for which maximum cluster
 *                      size is calculated
 * \param config      - Launch configuration for the given kernel function
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaFuncGetAttributes
 */
template<class T>
static __inline__ __host__ cudaError_t cudaOccupancyMaxPotentialClusterSize(
    int *clusterSize,
    T *func,
    const cudaLaunchConfig_t *config)
{
    return ::cudaOccupancyMaxPotentialClusterSize(clusterSize, (const void*)func, config);
}

/**
 * \brief Given the kernel function (\p func) and launch configuration
 * (\p config), return the maximum number of clusters that could co-exist
 * on the target device in \p *numClusters.
 *
 * If the function has required cluster size already set (see
 * ::cudaFuncGetAttributes), the cluster size from config must either be
 * unspecified or match the required size.
 * Without required sizes, the cluster size must be specified in config,
 * else the function will return an error.
 *
 * Note that various attributes of the kernel function may affect occupancy
 * calculation. Runtime environment may affect how the hardware schedules
 * the clusters, so the calculated occupancy is not guaranteed to be achievable.
 *
 * \param numClusters - Returned maximum number of clusters that
 *                      could co-exist on the target device
 * \param func        - Kernel function for which maximum number
 *                      of clusters are calculated
 * \param config      - Launch configuration for the given kernel function
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue,
 * ::cudaErrorInvalidClusterSize,
 * ::cudaErrorUnknown,
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \sa
 * ::cudaFuncGetAttributes
 */
template<class T>
static __inline__ __host__ cudaError_t cudaOccupancyMaxActiveClusters(
    int *numClusters,
    T *func,
    const cudaLaunchConfig_t *config)
{
    return ::cudaOccupancyMaxActiveClusters(numClusters, (const void*)func, config);
}

#if defined __CUDACC__

/**
 * \brief \hl Find out attributes for a given function
 *
 * This function obtains the attributes of a function specified via \p entry.
 * The parameter \p entry must be a pointer to a function that executes
 * on the device. The parameter specified by \p entry must be declared as a \p __global__
 * function. The fetched attributes are placed in \p attr. If the specified
 * function does not exist, then ::cudaErrorInvalidDeviceFunction is returned.
 *
 * Note that some function attributes such as
 * \ref ::cudaFuncAttributes::maxThreadsPerBlock "maxThreadsPerBlock"
 * may vary based on the device that is currently being used.
 *
 * \param attr  - Return pointer to function's attributes
 * \param entry - Function to get attributes of
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost
 */
template<class T>
static __inline__ __host__ cudaError_t cudaFuncGetAttributes(
  struct cudaFuncAttributes *attr,
  T                         *entry
)
{
  return ::cudaFuncGetAttributes(attr, (const void*)entry);
}

/**
 * \brief \hl Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p entry.
 * The parameter \p entry must be a pointer to a function that executes
 * on the device. The parameter specified by \p entry must be declared as a \p __global__
 * function. The enumeration defined by \p attr is set to the value defined by \p value.
 * If the specified function does not exist, then ::cudaErrorInvalidDeviceFunction is returned.
 * If the specified attribute cannot be written, or if the value is incorrect, 
 * then ::cudaErrorInvalidValue is returned.
 *
 * Valid values for \p attr are:
 * - ::cudaFuncAttributeMaxDynamicSharedMemorySize - The requested maximum size in bytes of dynamically-allocated shared memory. The sum of this value and the function attribute ::sharedSizeBytes
 *   cannot exceed the device attribute ::cudaDevAttrMaxSharedMemoryPerBlockOptin. The maximal size of requestable dynamic shared memory may differ by GPU architecture.
 * - ::cudaFuncAttributePreferredSharedMemoryCarveout - On devices where the L1 cache and shared memory use the same hardware resources, 
 *   this sets the shared memory carveout preference, in percent of the total shared memory. See ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
 *   This is only a hint, and the driver can choose a different ratio if required to execute the function.
 * - ::cudaFuncAttributeRequiredClusterWidth: The required cluster width in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return cudaErrorNotPermitted.
 * - ::cudaFuncAttributeRequiredClusterHeight: The required cluster height in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return cudaErrorNotPermitted.
 * - ::cudaFuncAttributeRequiredClusterDepth: The required cluster depth in
 *   blocks. The width, height, and depth values must either all be 0 or all be
 *   positive. The validity of the cluster dimensions is checked at launch time.
 *   If the value is set during compile time, it cannot be set at runtime.
 *   Setting it at runtime will return cudaErrorNotPermitted.
 * - ::cudaFuncAttributeClusterSchedulingPolicyPreference: The block
 *   scheduling policy of a function. The value type is cudaClusterSchedulingPolicy.
 *
 * \param entry - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue
 * \notefnerr
 * \note_init_rt
 * \note_callback
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig (C++ API)",
 * \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost
 */
template<class T>
static __inline__ __host__ cudaError_t cudaFuncSetAttribute(
  T                         *entry,
  enum cudaFuncAttribute    attr,
  int                       value
)
{
  return ::cudaFuncSetAttribute((const void*)entry, attr, value);
}

/**
 * \brief Get pointer to device kernel that matches entry function \p entryFuncAddr
  *
  * Returns in \p kernelPtr the device kernel corresponding to the entry function \p entryFuncAddr.
  *
  * \param kernelPtr          - Returns the device kernel
  * \param entryFuncAddr      - Address of device entry function to search kernel for
  *
  * \return
  * ::cudaSuccess
  *
  * \sa
  * \ref ::cudaGetKernel(cudaKernel_t *kernelPtr, const void *entryFuncAddr) "cudaGetKernel (C API)"
  */
template<class T>
static  __inline__ __host__ cudaError_t cudaGetKernel(
  cudaKernel_t *kernelPtr,
  const T *entryFuncAddr
)
{
  return ::cudaGetKernel(kernelPtr, (const void *)entryFuncAddr);
}

#endif /* __CUDACC__ */

/** @} */ /* END CUDART_HIGHLEVEL */

#endif /* __cplusplus && !__CUDACC_RTC__ */

#if !defined(__CUDACC_RTC__)
#if defined(__GNUC__)
#if defined(__clang__) || (!defined(__PGIC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))
#pragma GCC diagnostic pop
#endif
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

#undef EXCLUDE_FROM_RTC
#undef __CUDA_DEPRECATED

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_RUNTIME_H__
#endif

#endif /* !__CUDA_RUNTIME_H__ */
