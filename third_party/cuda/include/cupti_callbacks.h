/*
 * Copyright 2010-2020 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUPTI_CALLBACKS_H__)
#define __CUPTI_CALLBACKS_H__

#include <cuda.h>
// #include <builtin_types.h>
#include <string.h>
#include <cuda_stdint.h>
#include <cupti_result.h>

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
 * \defgroup CUPTI_CALLBACK_API CUPTI Callback API
 * Functions, types, and enums that implement the CUPTI Callback API.
 * @{
 */

/**
 * \brief Specifies the point in an API call that a callback is issued.
 *
 * Specifies the point in an API call that a callback is issued. This
 * value is communicated to the callback function via \ref
 * CUpti_CallbackData::callbackSite.
 */
typedef enum {
  /**
   * The callback is at the entry of the API call.
   */
  CUPTI_API_ENTER                 = 0,
  /**
   * The callback is at the exit of the API call.
   */
  CUPTI_API_EXIT                  = 1,
  CUPTI_API_CBSITE_FORCE_INT     = 0x7fffffff
} CUpti_ApiCallbackSite;

/**
 * \brief Callback domains.
 *
 * Callback domains. Each domain represents callback points for a
 * group of related API functions or CUDA driver activity.
 */
typedef enum {
  /**
   * Invalid domain.
   */
  CUPTI_CB_DOMAIN_INVALID           = 0,
  /**
   * Domain containing callback points for all driver API functions.
   */
  CUPTI_CB_DOMAIN_DRIVER_API        = 1,
  /**
   * Domain containing callback points for all runtime API
   * functions.
   */
  CUPTI_CB_DOMAIN_RUNTIME_API       = 2,
  /**
   * Domain containing callback points for CUDA resource tracking.
   */
  CUPTI_CB_DOMAIN_RESOURCE          = 3,
  /**
   * Domain containing callback points for CUDA synchronization.
   */
  CUPTI_CB_DOMAIN_SYNCHRONIZE       = 4,
  /**
   * Domain containing callback points for NVTX API functions.
   */
  CUPTI_CB_DOMAIN_NVTX              = 5,
  CUPTI_CB_DOMAIN_SIZE              = 6,
  CUPTI_CB_DOMAIN_FORCE_INT         = 0x7fffffff
} CUpti_CallbackDomain;

/**
 * \brief Callback IDs for resource domain.
 *
 * Callback IDs for resource domain, CUPTI_CB_DOMAIN_RESOURCE.  This
 * value is communicated to the callback function via the \p cbid
 * parameter.
 */
typedef enum {
  /**
   * Invalid resource callback ID.
   */
  CUPTI_CBID_RESOURCE_INVALID                               = 0,
  /**
   * A new context has been created.
   */
  CUPTI_CBID_RESOURCE_CONTEXT_CREATED                       = 1,
  /**
   * A context is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING              = 2,
  /**
   * A new stream has been created.
   */
  CUPTI_CBID_RESOURCE_STREAM_CREATED                        = 3,
  /**
   * A stream is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING               = 4,
  /**
   * The driver has finished initializing.
   */
  CUPTI_CBID_RESOURCE_CU_INIT_FINISHED                      = 5,
  /**
   * A module has been loaded.
   */
  CUPTI_CBID_RESOURCE_MODULE_LOADED                         = 6,
  /**
   * A module is about to be unloaded.
   */
  CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING                = 7,
  /**
   * The current module which is being profiled.
   */
  CUPTI_CBID_RESOURCE_MODULE_PROFILED                       = 8,
  /**
   * CUDA graph has been created.
   */
  CUPTI_CBID_RESOURCE_GRAPH_CREATED                         = 9,
  /**
   * CUDA graph is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING                = 10,
  /**
   * CUDA graph is cloned.
   */
  CUPTI_CBID_RESOURCE_GRAPH_CLONED                          = 11,
  /**
   * CUDA graph node is about to be created
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING             = 12,
  /**
   * CUDA graph node is created.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED                     = 13,
  /**
   * CUDA graph node is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING            = 14,
  /**
   * Dependency on a CUDA graph node is created.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED          = 15,
  /**
   * Dependency on a CUDA graph node is destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16,
  /**
   * An executable CUDA graph is about to be created.
   */
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING             = 17,
  /**
   * An executable CUDA graph is created.
   */
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED                     = 18,
  /**
   * An executable CUDA graph is about to be destroyed.
   */
  CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING            = 19,
  /**
   * CUDA graph node is cloned.
   */
  CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED                      = 20,

  CUPTI_CBID_RESOURCE_SIZE,
  CUPTI_CBID_RESOURCE_FORCE_INT                   = 0x7fffffff
} CUpti_CallbackIdResource;

/**
 * \brief Callback IDs for synchronization domain.
 *
 * Callback IDs for synchronization domain,
 * CUPTI_CB_DOMAIN_SYNCHRONIZE.  This value is communicated to the
 * callback function via the \p cbid parameter.
 */
typedef enum {
  /**
   * Invalid synchronize callback ID.
   */
  CUPTI_CBID_SYNCHRONIZE_INVALID                  = 0,
  /**
   * Stream synchronization has completed for the stream.
   */
  CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED      = 1,
  /**
   * Context synchronization has completed for the context.
   */
  CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED     = 2,
  CUPTI_CBID_SYNCHRONIZE_SIZE,
  CUPTI_CBID_SYNCHRONIZE_FORCE_INT                = 0x7fffffff
} CUpti_CallbackIdSync;

/**
 * \brief Data passed into a runtime or driver API callback function.
 *
 * Data passed into a runtime or driver API callback function as the
 * \p cbdata argument to \ref CUpti_CallbackFunc. The \p cbdata will
 * be this type for \p domain equal to CUPTI_CB_DOMAIN_DRIVER_API or
 * CUPTI_CB_DOMAIN_RUNTIME_API. The callback data is valid only within
 * the invocation of the callback function that is passed the data. If
 * you need to retain some data for use outside of the callback, you
 * must make a copy of that data. For example, if you make a shallow
 * copy of CUpti_CallbackData within a callback, you cannot
 * dereference \p functionParams outside of that callback to access
 * the function parameters. \p functionName is an exception: the
 * string pointed to by \p functionName is a global constant and so
 * may be accessed outside of the callback.
 */
typedef struct {
  /**
   * Point in the runtime or driver function from where the callback
   * was issued.
   */
  CUpti_ApiCallbackSite callbackSite;

  /**
   * Name of the runtime or driver API function which issued the
   * callback. This string is a global constant and so may be
   * accessed outside of the callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the runtime or driver API
   * call. See generated_cuda_runtime_api_meta.h and
   * generated_cuda_meta.h for structure definitions for the
   * parameters for each runtime and driver API function.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the runtime or driver API
   * call. This field is only valid within the exit::CUPTI_API_EXIT
   * callback. For a runtime API \p functionReturnValue points to a
   * \p cudaError_t. For a driver API \p functionReturnValue points
   * to a \p CUresult.
   */
  void *functionReturnValue;

  /**
   * Name of the symbol operated on by the runtime or driver API
   * function which issued the callback. This entry is valid only for
   * driver and runtime launch callbacks, where it returns the name of
   * the kernel.
   */
  const char *symbolName;

  /**
   * Driver context current to the thread, or null if no context is
   * current. This value can change from the entry to exit callback
   * of a runtime API function if the runtime initializes a context.
   */
  CUcontext context;

  /**
   * Unique ID for the CUDA context associated with the thread. The
   * UIDs are assigned sequentially as contexts are created and are
   * unique within a process.
   */
  uint32_t contextUid;

  /**
   * Pointer to data shared between the entry and exit callbacks of
   * a given runtime or drive API function invocation. This field
   * can be used to pass 64-bit values from the entry callback to
   * the corresponding exit callback.
   */
  uint64_t *correlationData;

  /**
   * The activity record correlation ID for this callback. For a
   * driver domain callback (i.e. \p domain
   * CUPTI_CB_DOMAIN_DRIVER_API) this ID will equal the correlation ID
   * in the CUpti_ActivityAPI record corresponding to the CUDA driver
   * function call. For a runtime domain callback (i.e. \p domain
   * CUPTI_CB_DOMAIN_RUNTIME_API) this ID will equal the correlation
   * ID in the CUpti_ActivityAPI record corresponding to the CUDA
   * runtime function call. Within the callback, this ID can be
   * recorded to correlate user data with the activity record. This
   * field is new in 4.1.
   */
  uint32_t correlationId;

} CUpti_CallbackData;

/**
 * \brief Data passed into a resource callback function.
 *
 * Data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The callback
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * For CUPTI_CBID_RESOURCE_CONTEXT_CREATED and
   * CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING, the context being
   * created or destroyed. For CUPTI_CBID_RESOURCE_STREAM_CREATED and
   * CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the context
   * containing the stream being created or destroyed.
   */
  CUcontext context;

  union {
    /**
     * For CUPTI_CBID_RESOURCE_STREAM_CREATED and
     * CUPTI_CBID_RESOURCE_STREAM_DESTROY_STARTING, the stream being
     * created or destroyed.
     */
    CUstream stream;
  } resourceHandle;

  /**
   * Reserved for future use.
   */
  void *resourceDescriptor;
} CUpti_ResourceData;


/**
 * \brief Module data passed into a resource callback function.
 *
 * CUDA module data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The module
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */

typedef struct {
  /**
   * Identifier to associate with the CUDA module.
   */
    uint32_t moduleId;

  /**
   * The size of the cubin.
   */
    size_t cubinSize;

  /**
   * Pointer to the associated cubin.
   */
    const char *pCubin;
} CUpti_ModuleResourceData;

/**
 * \brief CUDA graphs data passed into a resource callback function.
 *
 * CUDA graphs data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The graph
 * data is valid only within the invocation of the callback function
 * that is passed the data. If you need to retain some data for use
 * outside of the callback, you must make a copy of that data.
 */

typedef struct {
  /**
   * CUDA graph
   */
    CUgraph graph;
  /**
   * The original CUDA graph from which \param graph is cloned
   */
    CUgraph originalGraph;
  /**
   * CUDA graph node
   */
    CUgraphNode node;
  /**
   * The original CUDA graph node from which \param node is cloned
   */
    CUgraphNode originalNode;
  /**
   * Type of the \param node
   */
    CUgraphNodeType nodeType;
  /**
   * The dependent graph node
   * The size of the array is \param numDependencies.
   */
    CUgraphNode dependency;
  /**
   * CUDA executable graph
   */
    CUgraphExec graphExec;
} CUpti_GraphData;

/**
 * \brief Data passed into a synchronize callback function.
 *
 * Data passed into a synchronize callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_SYNCHRONIZE. The
 * callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * The context of the stream being synchronized.
   */
  CUcontext context;
  /**
   * The stream being synchronized.
   */
  CUstream  stream;
} CUpti_SynchronizeData;

/**
 * \brief Data passed into a NVTX callback function.
 *
 * Data passed into a NVTX callback function as the \p cbdata argument
 * to \ref CUpti_CallbackFunc. The \p cbdata will be this type for \p
 * domain equal to CUPTI_CB_DOMAIN_NVTX. Unless otherwise notes, the
 * callback data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
typedef struct {
  /**
   * Name of the NVTX API function which issued the callback. This
   * string is a global constant and so may be accessed outside of the
   * callback.
   */
  const char *functionName;

  /**
   * Pointer to the arguments passed to the NVTX API call. See
   * generated_nvtx_meta.h for structure definitions for the
   * parameters for each NVTX API function.
   */
  const void *functionParams;

  /**
   * Pointer to the return value of the NVTX API call. See
   * nvToolsExt.h for each NVTX API function's return value.
   */
  const void *functionReturnValue;
} CUpti_NvtxData;

/**
 * \brief An ID for a driver API, runtime API, resource or
 * synchronization callback.
 *
 * An ID for a driver API, runtime API, resource or synchronization
 * callback. Within a driver API callback this should be interpreted
 * as a CUpti_driver_api_trace_cbid value (these values are defined in
 * cupti_driver_cbid.h). Within a runtime API callback this should be
 * interpreted as a CUpti_runtime_api_trace_cbid value (these values
 * are defined in cupti_runtime_cbid.h). Within a resource API
 * callback this should be interpreted as a \ref
 * CUpti_CallbackIdResource value. Within a synchronize API callback
 * this should be interpreted as a \ref CUpti_CallbackIdSync value.
 */
typedef uint32_t CUpti_CallbackId;

/**
 * \brief Function type for a callback.
 *
 * Function type for a callback. The type of the data passed to the
 * callback in \p cbdata depends on the \p domain. If \p domain is
 * CUPTI_CB_DOMAIN_DRIVER_API or CUPTI_CB_DOMAIN_RUNTIME_API the type
 * of \p cbdata will be CUpti_CallbackData. If \p domain is
 * CUPTI_CB_DOMAIN_RESOURCE the type of \p cbdata will be
 * CUpti_ResourceData. If \p domain is CUPTI_CB_DOMAIN_SYNCHRONIZE the
 * type of \p cbdata will be CUpti_SynchronizeData. If \p domain is
 * CUPTI_CB_DOMAIN_NVTX the type of \p cbdata will be CUpti_NvtxData.
 *
 * \param userdata User data supplied at subscription of the callback
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param cbdata Data passed to the callback.
 */
typedef void (CUPTIAPI *CUpti_CallbackFunc)(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void *cbdata);

/**
 * \brief A callback subscriber.
 */
typedef struct CUpti_Subscriber_st *CUpti_SubscriberHandle;

/**
 * \brief Pointer to an array of callback domains.
 */
typedef CUpti_CallbackDomain *CUpti_DomainTable;

/**
 * \brief Get the available callback domains.
 *
 * Returns in \p *domainTable an array of size \p *domainCount of all
 * the available callback domains.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param domainCount Returns number of callback domains
 * \param domainTable Returns pointer to array of available callback domains
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialize CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p domainCount or \p domainTable are NULL
 */
CUptiResult CUPTIAPI cuptiSupportedDomains(size_t *domainCount,
                                           CUpti_DomainTable *domainTable);

/**
 * \brief Initialize a callback subscriber with a callback function
 * and user data.
 *
 * Initializes a callback subscriber with a callback function and
 * (optionally) a pointer to user data. The returned subscriber handle
 * can be used to enable and disable the callback for specific domains
 * and callback IDs.
 * \note Only a single subscriber can be registered at a time. To ensure
 * that no other CUPTI client interrupts the profiling session, it's the
 * responsibility of all the CUPTI clients to call this function before
 * starting the profling session. In case profiling session is already
 * started by another CUPTI client, this function returns the error code
 * CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED.
 * Note that this function returns the same error when application is
 * launched using NVIDIA tools like nvprof, Visual Profiler, Nsight Systems,
 * Nsight Compute, cuda-gdb and cuda-memcheck.
 * \note This function does not enable any callbacks.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param subscriber Returns handle to initialize subscriber
 * \param callback The callback function
 * \param userdata A pointer to user data. This data will be passed to
 * the callback function via the \p userdata paramater.
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialize CUPTI
 * \retval CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if there is already a CUPTI subscriber
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber is NULL
 */
CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                                    CUpti_CallbackFunc callback,
                                    void *userdata);

/**
 * \brief Unregister a callback subscriber.
 *
 * Removes a callback subscriber so that no future callbacks will be
 * issued to that subscriber.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param subscriber Handle to the initialize subscriber
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber is NULL or not initialized
 */
CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);

/**
 * \brief Get the current enabled/disabled state of a callback for a specific
 * domain and function ID.
 *
 * Returns non-zero in \p *enable if the callback for a domain and
 * callback ID is enabled, and zero if not enabled.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackState(sub,
 * d, c) and cuptiEnableCallback(sub, d, c) are called concurrently,
 * the results are undefined.
 *
 * \param enable Returns non-zero if callback enabled, zero if not enabled
 * \param subscriber Handle to the initialize subscriber
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p enabled is NULL, or if \p
 * subscriber, \p domain or \p cbid is invalid.
 */
CUptiResult CUPTIAPI cuptiGetCallbackState(uint32_t *enable,
                                           CUpti_SubscriberHandle subscriber,
                                           CUpti_CallbackDomain domain,
                                           CUpti_CallbackId cbid);

/**
 * \brief Enable or disabled callbacks for a specific domain and
 * callback ID.
 *
 * Enable or disabled callbacks for a subscriber for a specific domain
 * and callback ID.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackState(sub,
 * d, c) and cuptiEnableCallback(sub, d, c) are called concurrently,
 * the results are undefined.
 *
 * \param enable New enable state for the callback. Zero disables the
 * callback, non-zero enables the callback.
 * \param subscriber - Handle to callback subscription
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber, \p domain or \p
 * cbid is invalid.
 */
CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid);

/**
 * \brief Enable or disabled all callbacks for a specific domain.
 *
 * Enable or disabled all callbacks for a specific domain.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackEnabled(sub,
 * d, *) and cuptiEnableDomain(sub, d) are called concurrently, the
 * results are undefined.
 *
 * \param enable New enable state for all callbacks in the
 * domain. Zero disables all callbacks, non-zero enables all
 * callbacks.
 * \param subscriber - Handle to callback subscription
 * \param domain The domain of the callback
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber or \p domain is invalid
 */
CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain);

/**
 * \brief Enable or disable all callbacks in all domains.
 *
 * Enable or disable all callbacks in all domains.
 *
 * \note \b Thread-safety: a subscriber must serialize access to
 * cuptiGetCallbackState, cuptiEnableCallback, cuptiEnableDomain, and
 * cuptiEnableAllDomains. For example, if cuptiGetCallbackState(sub,
 * d, *) and cuptiEnableAllDomains(sub) are called concurrently, the
 * results are undefined.
 *
 * \param enable New enable state for all callbacks in all
 * domain. Zero disables all callbacks, non-zero enables all
 * callbacks.
 * \param subscriber - Handle to callback subscription
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_NOT_INITIALIZED if unable to initialized CUPTI
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p subscriber is invalid
 */
CUptiResult CUPTIAPI cuptiEnableAllDomains(uint32_t enable,
                                           CUpti_SubscriberHandle subscriber);

/**
 * \brief Get the name of a callback for a specific domain and callback ID.
 *
 * Returns a pointer to the name c_string in \p **name.
 *
 * \note \b Names are available only for the DRIVER and RUNTIME domains.
 *
 * \param domain The domain of the callback
 * \param cbid The ID of the callback
 * \param name Returns pointer to the name string on success, NULL otherwise
 *
 * \retval CUPTI_SUCCESS on success
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p name is NULL, or if
 * \p domain or \p cbid is invalid.
 */
CUptiResult CUPTIAPI cuptiGetCallbackName(CUpti_CallbackDomain domain,
                                          uint32_t cbid,
                                          const char **name);

/** @} */ /* END CUPTI_CALLBACK_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif  // file guard
