/*
 * Copyright 2010-2021 NVIDIA Corporation.  All rights reserved.
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

#if !defined(_CUPTI_RESULT_H_)
#define _CUPTI_RESULT_H_

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else
#define CUPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup CUPTI_RESULT_API CUPTI Result Codes
 * Error and result codes returned by CUPTI functions.
 * @{
 */

/**
 * \brief CUPTI result codes.
 *
 * Error and result codes returned by CUPTI functions.
 */
typedef enum {
    /**
     * No error.
     */
    CUPTI_SUCCESS                                       = 0,
    /**
     * One or more of the parameters is invalid.
     */
    CUPTI_ERROR_INVALID_PARAMETER                       = 1,
    /**
     * The device does not correspond to a valid CUDA device.
     */
    CUPTI_ERROR_INVALID_DEVICE                          = 2,
    /**
     * The context is NULL or not valid.
     */
    CUPTI_ERROR_INVALID_CONTEXT                         = 3,
    /**
     * The event domain id is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID                 = 4,
    /**
     * The event id is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_ID                        = 5,
    /**
     * The event name is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_NAME                      = 6,
    /**
     * The current operation cannot be performed due to dependency on
     * other factors.
     */
    CUPTI_ERROR_INVALID_OPERATION                       = 7,
    /**
     * Unable to allocate enough memory to perform the requested
     * operation.
     */
    CUPTI_ERROR_OUT_OF_MEMORY                           = 8,
    /**
     * An error occurred on the performance monitoring hardware.
     */
    CUPTI_ERROR_HARDWARE                                = 9,
    /**
     * The output buffer size is not sufficient to return all
     * requested data.
     */
    CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT           = 10,
    /**
     * API is not implemented.
     */
    CUPTI_ERROR_API_NOT_IMPLEMENTED                     = 11,
    /**
     * The maximum limit is reached.
     */
    CUPTI_ERROR_MAX_LIMIT_REACHED                       = 12,
    /**
     * The object is not yet ready to perform the requested operation.
     */
    CUPTI_ERROR_NOT_READY                               = 13,
    /**
     * The current operation is not compatible with the current state
     * of the object
     */
    CUPTI_ERROR_NOT_COMPATIBLE                          = 14,
    /**
     * CUPTI is unable to initialize its connection to the CUDA
     * driver.
     */
    CUPTI_ERROR_NOT_INITIALIZED                         = 15,
    /**
     * The metric id is invalid.
     */
    CUPTI_ERROR_INVALID_METRIC_ID                        = 16,
    /**
     * The metric name is invalid.
     */
    CUPTI_ERROR_INVALID_METRIC_NAME                      = 17,
    /**
     * The queue is empty.
     */
    CUPTI_ERROR_QUEUE_EMPTY                              = 18,
    /**
     * Invalid handle (internal?).
     */
    CUPTI_ERROR_INVALID_HANDLE                           = 19,
    /**
     * Invalid stream.
     */
    CUPTI_ERROR_INVALID_STREAM                           = 20,
    /**
     * Invalid kind.
     */
    CUPTI_ERROR_INVALID_KIND                             = 21,
    /**
     * Invalid event value.
     */
    CUPTI_ERROR_INVALID_EVENT_VALUE                      = 22,
    /**
     * CUPTI is disabled due to conflicts with other enabled profilers
     */
    CUPTI_ERROR_DISABLED                                 = 23,
    /**
     * Invalid module.
     */
    CUPTI_ERROR_INVALID_MODULE                           = 24,
    /**
     * Invalid metric value.
     */
    CUPTI_ERROR_INVALID_METRIC_VALUE                     = 25,
    /**
     * The performance monitoring hardware is in use by other client.
     */
    CUPTI_ERROR_HARDWARE_BUSY                            = 26,
    /**
     * The attempted operation is not supported on the current
     * system or device.
     */
    CUPTI_ERROR_NOT_SUPPORTED                            = 27,
    /**
     * Unified memory profiling is not supported on the system.
     * Potential reason could be unsupported OS or architecture.
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED               = 28,
    /**
     * Unified memory profiling is not supported on the device
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE     = 29,
    /**
     * Unified memory profiling is not supported on a multi-GPU
     * configuration without P2P support between any pair of devices
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES = 30,
    /**
     * Unified memory profiling is not supported under the
     * Multi-Process Service (MPS) environment. CUDA 7.5 removes this
     * restriction.
     */
    CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS      = 31,
    /**
     * In CUDA 9.0, devices with compute capability 7.0 don't
     * support CDP tracing
     */
    CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED                = 32,
    /**
     * Profiling on virtualized GPU is not supported.
     */
    CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED         = 33,
    /**
     * Profiling results might be incorrect for CUDA applications
     * compiled with nvcc version older than 9.0 for devices with
     * compute capability 6.0 and 6.1.
     * Profiling session will continue and CUPTI will notify it using this error code.
     * User is advised to recompile the application code with nvcc version 9.0 or later.
     * Ignore this warning if code is already compiled with the recommended nvcc version.
     */
    CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE             = 34,
    /**
     * User doesn't have sufficient privileges which are required to
     * start the profiling session.
     * One possible reason for this may be that the NVIDIA driver or your system
     * administrator may have restricted access to the NVIDIA GPU performance counters.
     * To learn how to resolve this issue and find more information, please visit
     * https://developer.nvidia.com/CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
     */
    CUPTI_ERROR_INSUFFICIENT_PRIVILEGES                  = 35,
    /**
     * Legacy CUPTI Profiling API i.e. event API from the header cupti_events.h and
     * metric API from the header cupti_metrics.h are not compatible with the
     * Profiling API in the header cupti_profiler_target.h and Perfworks metrics API
     * in the headers nvperf_host.h and nvperf_target.h.
     */
    CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED             = 36,
    /**
     * Missing definition of the OpenACC API routine in the linked OpenACC library.
     *
     * One possible reason is that OpenACC library is linked statically in the
     * user application, which might not have the definition of all the OpenACC
     * API routines needed for the OpenACC profiling, as compiler might ignore
     * definitions for the functions not used in the application. This issue
     * can be mitigated by linking the OpenACC library dynamically.
     */
    CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE                = 37,
    /**
     * Legacy CUPTI Profiling API i.e. event API from the header cupti_events.h and
     * metric API from the header cupti_metrics.h are not supported on devices with
     * compute capability 7.5 and higher (i.e. Turing and later GPU architectures).
     * These API will be deprecated in a future CUDA release. These are replaced by
     * Profiling API in the header cupti_profiler_target.h and Perfworks metrics API
     * in the headers nvperf_host.h and nvperf_target.h.
     */
    CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED            = 38,
    /**
     * CUPTI doesn't allow multiple callback subscribers. Only a single subscriber
     * can be registered at a time.
     * Same error code is used when application is launched using NVIDIA tools
     * like nvprof, Visual Profiler, Nsight Systems, Nsight Compute, cuda-gdb and
     * cuda-memcheck.
     */
    CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED       = 39,
    /**
     * Profiling on virtualized GPU is not allowed by hypervisor.
     */
    CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES = 40,
    /**
     * Profiling and tracing are not allowed when confidential computing mode
     * is enabled.
     */
    CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED = 41,
    /**
     * CUPTI does not support NVIDIA Crypto Mining Processors (CMP).
     * For more information, please visit https://developer.nvidia.com/ERR_NVCMPGPU
    */
    CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED = 42,
    /**
     * An unknown internal error has occurred.
     */
    CUPTI_ERROR_UNKNOWN                                  = 999,
    CUPTI_ERROR_FORCE_INT                                = 0x7fffffff
} CUptiResult;

/**
 * \brief Get the descriptive string for a CUptiResult.
 *
 * Return the descriptive string for a CUptiResult in \p *str.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param result The result to get the string for
 * \param str Returns the string
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p str is NULL or \p
 * result is not a valid CUptiResult
 */
CUptiResult CUPTIAPI cuptiGetResultString(CUptiResult result, const char **str);

/** @} */ /* END CUPTI_RESULT_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_RESULT_H_*/
