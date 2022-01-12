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

#if !defined(_CUPTI_EVENTS_H_)
#define _CUPTI_EVENTS_H_

#include <cuda.h>
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
 * \defgroup CUPTI_EVENT_API CUPTI Event API
 * Functions, types, and enums that implement the CUPTI Event API.
 *
 * \note CUPTI event API from the header cupti_events.h are not supported on devices
 * with compute capability 7.5 and higher (i.e. Turing and later GPU architectures).
 * These API will be deprecated in a future CUDA release. These are replaced by
 * Profiling API in the header cupti_profiler_target.h and Perfworks metrics API
 * in the headers nvperf_host.h and nvperf_target.h which are supported on
 * devices with compute capability 7.0 and higher (i.e. Volta and later GPU
 * architectures).
 *
 * @{
 */

/**
 * \brief ID for an event.
 *
 * An event represents a countable activity, action, or occurrence on
 * the device.
 */
typedef uint32_t CUpti_EventID;

/**
 * \brief ID for an event domain.
 *
 * ID for an event domain. An event domain represents a group of
 * related events. A device may have multiple instances of a domain,
 * indicating that the device can simultaneously record multiple
 * instances of each event within that domain.
 */
typedef uint32_t CUpti_EventDomainID;

/**
 * \brief A group of events.
 *
 * An event group is a collection of events that are managed
 * together. All events in an event group must belong to the same
 * domain.
 */
typedef void *CUpti_EventGroup;

/**
 * \brief Device class.
 *
 * Enumeration of device classes for device attribute
 * CUPTI_DEVICE_ATTR_DEVICE_CLASS.
 */
typedef enum {
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_TESLA              = 0,
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_QUADRO             = 1,
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_GEFORCE            = 2,
  CUPTI_DEVICE_ATTR_DEVICE_CLASS_TEGRA              = 3,
} CUpti_DeviceAttributeDeviceClass;

/**
 * \brief Device attributes.
 *
 * CUPTI device attributes. These attributes can be read using \ref
 * cuptiDeviceGetAttribute.
 */
typedef enum {
  /**
   * Number of event IDs for a device. Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_MAX_EVENT_ID                            = 1,
  /**
   * Number of event domain IDs for a device. Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID                     = 2,
  /**
   * Get global memory bandwidth in Kbytes/sec. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_GLOBAL_MEMORY_BANDWIDTH                 = 3,
  /**
   * Get theoretical maximum number of instructions per cycle. Value
   * is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_INSTRUCTION_PER_CYCLE                   = 4,
  /**
   * Get theoretical maximum number of single precision instructions
   * that can be executed per second. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_INSTRUCTION_THROUGHPUT_SINGLE_PRECISION = 5,
  /**
   * Get number of frame buffers for device.  Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_FRAME_BUFFERS                       = 6,
  /**
   * Get PCIE link rate in Mega bits/sec for device. Return 0 if bus-type
   * is non-PCIE. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_PCIE_LINK_RATE                          = 7,
  /**
   * Get PCIE link width for device. Return 0 if bus-type
   * is non-PCIE. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_PCIE_LINK_WIDTH                         = 8,
  /**
   * Get PCIE generation for device. Return 0 if bus-type
   * is non-PCIE. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_PCIE_GEN                                = 9,
  /**
   * Get the class for the device. Value is a
   * CUpti_DeviceAttributeDeviceClass.
   */
  CUPTI_DEVICE_ATTR_DEVICE_CLASS                            = 10,
  /**
   * Get the peak single precision flop per cycle. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_FLOP_SP_PER_CYCLE                       = 11,
  /**
   * Get the peak double precision flop per cycle. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_FLOP_DP_PER_CYCLE                       = 12,
  /**
   * Get number of L2 units. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_L2_UNITS                           = 13,
  /**
   * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_SHARED
   * preference. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_SHARED = 14,
  /**
   * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_L1
   * preference. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_L1 = 15,
  /**
   * Get the maximum shared memory for the CU_FUNC_CACHE_PREFER_EQUAL
   * preference. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_MAX_SHARED_MEMORY_CACHE_CONFIG_PREFER_EQUAL = 16,
  /**
   * Get the peak half precision flop per cycle. Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_FLOP_HP_PER_CYCLE                       = 17,
  /**
   * Check if Nvlink is connected to device. Returns 1, if at least one
   * Nvlink is connected to the device, returns 0 otherwise.
   * Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_NVLINK_PRESENT                          = 18,
    /**
   * Check if Nvlink is present between GPU and CPU. Returns Bandwidth,
   * in Bytes/sec, if Nvlink is present, returns 0 otherwise.
   * Value is a uint64_t.
   */
  CUPTI_DEVICE_ATTR_GPU_CPU_NVLINK_BW                       = 19,
  /**
   * Check if NVSwitch is present in the underlying topology.
   * Returns 1, if present, returns 0 otherwise.
   * Value is a uint32_t.
   */
  CUPTI_DEVICE_ATTR_NVSWITCH_PRESENT                        = 20,
  CUPTI_DEVICE_ATTR_FORCE_INT                               = 0x7fffffff,
} CUpti_DeviceAttribute;

/**
 * \brief Event domain attributes.
 *
 * Event domain attributes. Except where noted, all the attributes can
 * be read using either \ref cuptiDeviceGetEventDomainAttribute or
 * \ref cuptiEventDomainGetAttribute.
 */
typedef enum {
  /**
   * Event domain name. Value is a null terminated const c-string.
   */
  CUPTI_EVENT_DOMAIN_ATTR_NAME                 = 0,
  /**
   * Number of instances of the domain for which event counts will be
   * collected.  The domain may have additional instances that cannot
   * be profiled (see CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT).
   * Can be read only with \ref
   * cuptiDeviceGetEventDomainAttribute. Value is a uint32_t.
   */
  CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT       = 1,
  /**
   * Total number of instances of the domain, including instances that
   * cannot be profiled.  Use CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT
   * to get the number of instances that can be profiled. Can be read
   * only with \ref cuptiDeviceGetEventDomainAttribute. Value is a
   * uint32_t.
   */
  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT = 3,
  /**
   * Collection method used for events contained in the event domain.
   * Value is a \ref CUpti_EventCollectionMethod.
   */
  CUPTI_EVENT_DOMAIN_ATTR_COLLECTION_METHOD    = 4,

  CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT      = 0x7fffffff,
} CUpti_EventDomainAttribute;

/**
 * \brief The collection method used for an event.
 *
 * The collection method indicates how an event is collected.
 */
typedef enum {
  /**
   * Event is collected using a hardware global performance monitor.
   */
  CUPTI_EVENT_COLLECTION_METHOD_PM                  = 0,
  /**
   * Event is collected using a hardware SM performance monitor.
   */
  CUPTI_EVENT_COLLECTION_METHOD_SM                  = 1,
  /**
   * Event is collected using software instrumentation.
   */
  CUPTI_EVENT_COLLECTION_METHOD_INSTRUMENTED        = 2,
  /**
   * Event is collected using NvLink throughput counter method.
   */
  CUPTI_EVENT_COLLECTION_METHOD_NVLINK_TC           = 3,
  CUPTI_EVENT_COLLECTION_METHOD_FORCE_INT           = 0x7fffffff
} CUpti_EventCollectionMethod;

/**
 * \brief Event group attributes.
 *
 * Event group attributes. These attributes can be read using \ref
 * cuptiEventGroupGetAttribute. Attributes marked [rw] can also be
 * written using \ref cuptiEventGroupSetAttribute.
 */
typedef enum {
  /**
   * The domain to which the event group is bound. This attribute is
   * set when the first event is added to the group.  Value is a
   * CUpti_EventDomainID.
   */
  CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID              = 0,
  /**
   * [rw] Profile all the instances of the domain for this
   * eventgroup. This feature can be used to get load balancing
   * across all instances of a domain. Value is an integer.
   */
  CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES = 1,
  /**
   * [rw] Reserved for user data.
   */
  CUPTI_EVENT_GROUP_ATTR_USER_DATA                    = 2,
  /**
   * Number of events in the group. Value is a uint32_t.
   */
  CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS                   = 3,
  /**
   * Enumerates events in the group. Value is a pointer to buffer of
   * size sizeof(CUpti_EventID) * num_of_events in the eventgroup.
   * num_of_events can be queried using
   * CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS.
   */
  CUPTI_EVENT_GROUP_ATTR_EVENTS                       = 4,
  /**
   * Number of instances of the domain bound to this event group that
   * will be counted.  Value is a uint32_t.
   */
  CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT               = 5,
  /**
   * Event group scope can be set to CUPTI_EVENT_PROFILING_SCOPE_DEVICE or
   * CUPTI_EVENT_PROFILING_SCOPE_CONTEXT for an eventGroup, before
   * adding any event.
   * Sets the scope of eventgroup as CUPTI_EVENT_PROFILING_SCOPE_DEVICE or
   * CUPTI_EVENT_PROFILING_SCOPE_CONTEXT when the scope of the events
   * that will be added is CUPTI_EVENT_PROFILING_SCOPE_BOTH.
   * If profiling scope of event is either
   * CUPTI_EVENT_PROFILING_SCOPE_DEVICE or CUPTI_EVENT_PROFILING_SCOPE_CONTEXT
   * then setting this attribute will not affect the default scope.
   * It is not allowed to add events of different scope to same eventgroup.
   * Value is a uint32_t.
   */
  CUPTI_EVENT_GROUP_ATTR_PROFILING_SCOPE               = 6,
  CUPTI_EVENT_GROUP_ATTR_FORCE_INT                     = 0x7fffffff,
} CUpti_EventGroupAttribute;

/**
* \brief Profiling scope for event.
*
* Profiling scope of event indicates if the event can be collected at context
* scope or device scope or both i.e. it can be collected at any of context or
* device scope.
*/
typedef enum {
  /**
   * Event is collected at context scope.
   */
  CUPTI_EVENT_PROFILING_SCOPE_CONTEXT                 = 0,
  /**
   * Event is collected at device scope.
   */
  CUPTI_EVENT_PROFILING_SCOPE_DEVICE                  = 1,
  /**
   * Event can be collected at device or context scope.
   * The scope can be set using \ref cuptiEventGroupSetAttribute API.
   */
  CUPTI_EVENT_PROFILING_SCOPE_BOTH                    = 2,
  CUPTI_EVENT_PROFILING_SCOPE_FORCE_INT               = 0x7fffffff
} CUpti_EventProfilingScope;

/**
 * \brief Event attributes.
 *
 * Event attributes. These attributes can be read using \ref
 * cuptiEventGetAttribute.
 */
typedef enum {
  /**
   * Event name. Value is a null terminated const c-string.
   */
  CUPTI_EVENT_ATTR_NAME              = 0,
  /**
   * Short description of event. Value is a null terminated const
   * c-string.
   */
  CUPTI_EVENT_ATTR_SHORT_DESCRIPTION = 1,
  /**
   * Long description of event. Value is a null terminated const
   * c-string.
   */
  CUPTI_EVENT_ATTR_LONG_DESCRIPTION  = 2,
  /**
   * Category of event. Value is CUpti_EventCategory.
   */
  CUPTI_EVENT_ATTR_CATEGORY          = 3,
  /**
   * Profiling scope of the events. It can be either device or context or both.
   * Value is a \ref CUpti_EventProfilingScope.
   */
  CUPTI_EVENT_ATTR_PROFILING_SCOPE   = 5,

  CUPTI_EVENT_ATTR_FORCE_INT         = 0x7fffffff,
} CUpti_EventAttribute;

/**
 * \brief Event collection modes.
 *
 * The event collection mode determines the period over which the
 * events within the enabled event groups will be collected.
 */
typedef enum {
  /**
   * Events are collected for the entire duration between the
   * cuptiEventGroupEnable and cuptiEventGroupDisable calls.
   * Event values are reset when the events are read.
   * For CUDA toolkit v6.0 and older this was the default mode.
   */
  CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS          = 0,
  /**
   * Events are collected only for the durations of kernel executions
   * that occur between the cuptiEventGroupEnable and
   * cuptiEventGroupDisable calls. Event collection begins when a
   * kernel execution begins, and stops when kernel execution
   * completes. Event values are reset to zero when each kernel
   * execution begins. If multiple kernel executions occur between the
   * cuptiEventGroupEnable and cuptiEventGroupDisable calls then the
   * event values must be read after each kernel launch if those
   * events need to be associated with the specific kernel launch.
   * Note that collection in this mode may significantly change the
   * overall performance characteristics of the application because
   * kernel executions that occur between the cuptiEventGroupEnable and
   * cuptiEventGroupDisable calls are serialized on the GPU.
   * This is the default mode from CUDA toolkit v6.5
   */
  CUPTI_EVENT_COLLECTION_MODE_KERNEL              = 1,
  CUPTI_EVENT_COLLECTION_MODE_FORCE_INT           = 0x7fffffff
} CUpti_EventCollectionMode;

/**
 * \brief An event category.
 *
 * Each event is assigned to a category that represents the general
 * type of the event. A event's category is accessed using \ref
 * cuptiEventGetAttribute and the CUPTI_EVENT_ATTR_CATEGORY attribute.
 */
typedef enum {
  /**
   * An instruction related event.
   */
  CUPTI_EVENT_CATEGORY_INSTRUCTION     = 0,
  /**
   * A memory related event.
   */
  CUPTI_EVENT_CATEGORY_MEMORY          = 1,
  /**
   * A cache related event.
   */
  CUPTI_EVENT_CATEGORY_CACHE           = 2,
  /**
   * A profile-trigger event.
   */
  CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER = 3,
  /**
   * A system event.
   */
  CUPTI_EVENT_CATEGORY_SYSTEM  = 4,
  CUPTI_EVENT_CATEGORY_FORCE_INT       = 0x7fffffff
} CUpti_EventCategory;

/**
 * \brief The overflow value for a CUPTI event.
 *
 * The CUPTI event value that indicates an overflow.
 */
#define CUPTI_EVENT_OVERFLOW ((uint64_t)0xFFFFFFFFFFFFFFFFULL)

/**
 * \brief The value that indicates the event value is invalid
 */
#define CUPTI_EVENT_INVALID ((uint64_t)0xFFFFFFFFFFFFFFFEULL)

/**
 * \brief Flags for cuptiEventGroupReadEvent an
 * cuptiEventGroupReadAllEvents.
 *
 * Flags for \ref cuptiEventGroupReadEvent an \ref
 * cuptiEventGroupReadAllEvents.
 */
typedef enum {
  /**
   * No flags.
   */
  CUPTI_EVENT_READ_FLAG_NONE          = 0,
  CUPTI_EVENT_READ_FLAG_FORCE_INT     = 0x7fffffff,
} CUpti_ReadEventFlags;


/**
 * \brief A set of event groups.
 *
 * A set of event groups. When returned by \ref
 * cuptiEventGroupSetsCreate and \ref cuptiMetricCreateEventGroupSets
 * a set indicates that event groups that can be enabled at the same
 * time (i.e. all the events in the set can be collected
 * simultaneously).
 */
typedef struct {
  /**
   * The number of event groups in the set.
   */
  uint32_t numEventGroups;
  /**
   * An array of \p numEventGroups event groups.
   */
  CUpti_EventGroup *eventGroups;
} CUpti_EventGroupSet;

/**
 * \brief A set of event group sets.
 *
 * A set of event group sets. When returned by \ref
 * cuptiEventGroupSetsCreate and \ref cuptiMetricCreateEventGroupSets
 * a CUpti_EventGroupSets indicates the number of passes required to
 * collect all the events, and the event groups that should be
 * collected during each pass.
 */
typedef struct {
  /**
   * Number of event group sets.
   */
  uint32_t numSets;
  /**
   * An array of \p numSets event group sets.
   */
  CUpti_EventGroupSet *sets;
} CUpti_EventGroupSets;

/**
 * \brief Set the event collection mode.
 *
 * Set the event collection mode for a \p context.  The \p mode
 * controls the event collection behavior of all events in event
 * groups created in the \p context. This API is invalid in kernel
 * replay mode.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context
 * \param mode The event collection mode
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_OPERATION if called when replay mode is enabled
 * \retval CUPTI_ERROR_NOT_SUPPORTED if mode is not supported on the device
 */

CUptiResult CUPTIAPI cuptiSetEventCollectionMode(CUcontext context,
                                                 CUpti_EventCollectionMode mode);

/**
 * \brief Read a device attribute.
 *
 * Read a device attribute and return it in \p *value.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param attrib The attribute to read
 * \param valueSize Size of buffer pointed by the value, and
 * returns the number of bytes written to \p value
 * \param value Returns the value of the attribute
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not a device attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiDeviceGetAttribute(CUdevice device,
                                             CUpti_DeviceAttribute attrib,
                                             size_t *valueSize,
                                             void *value);

/**
 * \brief Read a device timestamp.
 *
 * Returns the device timestamp in \p *timestamp. The timestamp is
 * reported in nanoseconds and indicates the time since the device was
 * last reset.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context A context on the device from which to get the timestamp
 * \param timestamp Returns the device timestamp
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_PARAMETER is \p timestamp is NULL

 * **DEPRECATED** This API is deprecated as of CUDA 11.3
 */
CUptiResult CUPTIAPI cuptiDeviceGetTimestamp(CUcontext context,
                                             uint64_t *timestamp);

/**
 * \brief Get the number of domains for a device.
 *
 * Returns the number of domains in \p numDomains for a device.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param numDomains Returns the number of domains
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numDomains is NULL
 */
CUptiResult CUPTIAPI cuptiDeviceGetNumEventDomains(CUdevice device,
                                                   uint32_t *numDomains);

/**
 * \brief Get the event domains for a device.
 *
 * Returns the event domains IDs in \p domainArray for a device.  The
 * size of the \p domainArray buffer is given by \p
 * *arraySizeBytes. The size of the \p domainArray buffer must be at
 * least \p numdomains * sizeof(CUpti_EventDomainID) or else all
 * domains will not be returned. The value returned in \p
 * *arraySizeBytes contains the number of bytes returned in \p
 * domainArray.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param arraySizeBytes The size of \p domainArray in bytes, and
 * returns the number of bytes written to \p domainArray
 * \param domainArray Returns the IDs of the event domains for the device
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or
 * \p domainArray are NULL
 */
CUptiResult CUPTIAPI cuptiDeviceEnumEventDomains(CUdevice device,
                                                 size_t *arraySizeBytes,
                                                 CUpti_EventDomainID *domainArray);

/**
 * \brief Read an event domain attribute.
 *
 * Returns an event domain attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param eventDomain ID of the event domain
 * \param attrib The event domain attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event domain attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiDeviceGetEventDomainAttribute(CUdevice device,
                                                        CUpti_EventDomainID eventDomain,
                                                        CUpti_EventDomainAttribute attrib,
                                                        size_t *valueSize,
                                                        void *value);

/**
 * \brief Get the number of event domains available on any device.
 *
 * Returns the total number of event domains available on any
 * CUDA-capable device.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param numDomains Returns the number of domains
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numDomains is NULL
 */
CUptiResult CUPTIAPI cuptiGetNumEventDomains(uint32_t *numDomains);

/**
 * \brief Get the event domains available on any device.
 *
 * Returns all the event domains available on any CUDA-capable device.
 * Event domain IDs are returned in \p domainArray. The size of the \p
 * domainArray buffer is given by \p *arraySizeBytes. The size of the
 * \p domainArray buffer must be at least \p numDomains *
 * sizeof(CUpti_EventDomainID) or all domains will not be
 * returned. The value returned in \p *arraySizeBytes contains the
 * number of bytes returned in \p domainArray.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param arraySizeBytes The size of \p domainArray in bytes, and
 * returns the number of bytes written to \p domainArray
 * \param domainArray Returns all the event domains
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or
 * \p domainArray are NULL
 */
CUptiResult CUPTIAPI cuptiEnumEventDomains(size_t *arraySizeBytes,
                                           CUpti_EventDomainID *domainArray);

/**
 * \brief Read an event domain attribute.
 *
 * Returns an event domain attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventDomain ID of the event domain
 * \param attrib The event domain attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event domain attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventDomainGetAttribute(CUpti_EventDomainID eventDomain,
                                                  CUpti_EventDomainAttribute attrib,
                                                  size_t *valueSize,
                                                  void *value);

/**
 * \brief Get number of events in a domain.
 *
 * Returns the number of events in \p numEvents for a domain.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventDomain ID of the event domain
 * \param numEvents Returns the number of events in the domain
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p numEvents is NULL
 */
CUptiResult CUPTIAPI cuptiEventDomainGetNumEvents(CUpti_EventDomainID eventDomain,
                                                  uint32_t *numEvents);

/**
 * \brief Get the events in a domain.
 *
 * Returns the event IDs in \p eventArray for a domain.  The size of
 * the \p eventArray buffer is given by \p *arraySizeBytes. The size
 * of the \p eventArray buffer must be at least \p numdomainevents *
 * sizeof(CUpti_EventID) or else all events will not be returned. The
 * value returned in \p *arraySizeBytes contains the number of bytes
 * returned in \p eventArray.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventDomain ID of the event domain
 * \param arraySizeBytes The size of \p eventArray in bytes, and
 * returns the number of bytes written to \p eventArray
 * \param eventArray Returns the IDs of the events in the domain
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p arraySizeBytes or \p
 * eventArray are NULL
 */
CUptiResult CUPTIAPI cuptiEventDomainEnumEvents(CUpti_EventDomainID eventDomain,
                                                size_t *arraySizeBytes,
                                                CUpti_EventID *eventArray);

/**
 * \brief Get an event attribute.
 *
 * Returns an event attribute in \p *value. The size of the \p
 * value buffer is given by \p *valueSize. The value returned in \p
 * *valueSize contains the number of bytes returned in \p value.
 *
 * If the attribute value is a c-string that is longer than \p
 * *valueSize, then only the first \p *valueSize characters will be
 * returned and there will be no terminating null byte.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param event ID of the event
 * \param attrib The event attribute to read
 * \param valueSize The size of the \p value buffer in bytes, and
 * returns the number of bytes written to \p value
 * \param value Returns the attribute's value
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventGetAttribute(CUpti_EventID event,
                                            CUpti_EventAttribute attrib,
                                            size_t *valueSize,
                                            void *value);

/**
 * \brief Find an event by name.
 *
 * Find an event by name and return the event ID in \p *event.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param device The CUDA device
 * \param eventName The name of the event to find
 * \param event Returns the ID of the found event or undefined if
 * unable to find the event
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_DEVICE
 * \retval CUPTI_ERROR_INVALID_EVENT_NAME if unable to find an event
 * with name \p eventName. In this case \p *event is undefined
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventName or \p event are NULL
 */
CUptiResult CUPTIAPI cuptiEventGetIdFromName(CUdevice device,
                                             const char *eventName,
                                             CUpti_EventID *event);

/**
 * \brief Create a new event group for a context.
 *
 * Creates a new event group for \p context and returns the new group
 * in \p *eventGroup.
 * \note \p flags are reserved for future use and should be set to zero.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context for the event group
 * \param eventGroup Returns the new event group
 * \param flags Reserved - must be zero
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_OUT_OF_MEMORY
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupCreate(CUcontext context,
                                           CUpti_EventGroup *eventGroup,
                                           uint32_t flags);

/**
 * \brief Destroy an event group.
 *
 * Destroy an \p eventGroup and free its resources. An event group
 * cannot be destroyed if it is enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group to destroy
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if the event group is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupDestroy(CUpti_EventGroup eventGroup);

/**
 * \brief Read an event group attribute.
 *
 * Read an event group attribute and return it in \p *value.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.).
 *
 * \param eventGroup The event group
 * \param attrib The attribute to read
 * \param valueSize Size of buffer pointed by the value, and
 * returns the number of bytes written to \p value
 * \param value Returns the value of the attribute
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an eventgroup attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT For non-c-string
 * attribute values, indicates that the \p value buffer is too small
 * to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventGroupGetAttribute(CUpti_EventGroup eventGroup,
                                                 CUpti_EventGroupAttribute attrib,
                                                 size_t *valueSize,
                                                 void *value);

/**
 * \brief Write an event group attribute.
 *
 * Write an event group attribute.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 * \param attrib The attribute to write
 * \param valueSize The size, in bytes, of the value
 * \param value The attribute value to write
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p valueSize or \p value
 * is NULL, or if \p attrib is not an event group attribute, or if
 * \p attrib is not a writable attribute
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT Indicates that
 * the \p value buffer is too small to hold the attribute value.
 */
CUptiResult CUPTIAPI cuptiEventGroupSetAttribute(CUpti_EventGroup eventGroup,
                                                 CUpti_EventGroupAttribute attrib,
                                                 size_t valueSize,
                                                 void *value);

/**
 * \brief Add an event to an event group.
 *
 * Add an event to an event group. The event add can fail for a number of reasons:
 * \li The event group is enabled
 * \li The event does not belong to the same event domain as the
 * events that are already in the event group
 * \li Device limitations on the events that can belong to the same group
 * \li The event group is full
 *
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 * \param event The event to add to the group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_OUT_OF_MEMORY
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is enabled
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if \p event belongs to a
 * different event domain than the events already in \p eventGroup, or
 * if a device limitation prevents \p event from being collected at
 * the same time as the events already in \p eventGroup
 * \retval CUPTI_ERROR_MAX_LIMIT_REACHED if \p eventGroup is full
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupAddEvent(CUpti_EventGroup eventGroup,
                                             CUpti_EventID event);

/**
 * \brief Remove an event from an event group.
 *
 * Remove \p event from the an event group. The event cannot be
 * removed if the event group is enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 * \param event The event to remove from the group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupRemoveEvent(CUpti_EventGroup eventGroup,
                                                CUpti_EventID event);

/**
 * \brief Remove all events from an event group.
 *
 * Remove all events from an event group. Events cannot be removed if
 * the event group is enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupRemoveAllEvents(CUpti_EventGroup eventGroup);

/**
 * \brief Zero all the event counts in an event group.
 *
 * Zero all the event counts in an event group.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.).
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupResetAllEvents(CUpti_EventGroup eventGroup);

/**
 * \brief Enable an event group.
 *
 * Enable an event group. Enabling an event group zeros the value of
 * all the events in the group and then starts collection of those
 * events.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_NOT_READY if \p eventGroup does not contain any events
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if \p eventGroup cannot be
 * enabled due to other already enabled event groups
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 * \retval CUPTI_ERROR_HARDWARE_BUSY if another client is profiling
 * and hardware is busy
 */
CUptiResult CUPTIAPI cuptiEventGroupEnable(CUpti_EventGroup eventGroup);

/**
 * \brief Disable an event group.
 *
 * Disable an event group. Disabling an event group stops collection
 * of events contained in the group.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroup The event group
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupDisable(CUpti_EventGroup eventGroup);

/**
 * \brief Read the value for an event in an event group.
 *
 * Read the value for an event in an event group. The event value is
 * returned in the \p eventValueBuffer buffer. \p
 * eventValueBufferSizeBytes indicates the size of the \p
 * eventValueBuffer buffer. The buffer must be at least sizeof(uint64)
 * if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is not set
 * on the group containing the event.  The buffer must be at least
 * (sizeof(uint64) * number of domain instances) if
 * ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is set on the
 * group.
 *
 * If any instance of an event counter overflows, the value returned
 * for that event instance will be ::CUPTI_EVENT_OVERFLOW.
 *
 * The only allowed value for \p flags is ::CUPTI_EVENT_READ_FLAG_NONE.
 *
 * Reading an event from a disabled event group is not allowed. After
 * being read, an event's value is reset to zero.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.). If \ref cuptiEventGroupResetAllEvents is
 * called simultaneously with this function, then returned event
 * values are undefined.
 *
 * \param eventGroup The event group
 * \param flags Flags controlling the reading mode
 * \param event The event to read
 * \param eventValueBufferSizeBytes The size of \p eventValueBuffer
 * in bytes, and returns the number of bytes written to \p
 * eventValueBuffer
 * \param eventValueBuffer Returns the event value(s)
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is disabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup, \p
 * eventValueBufferSizeBytes or \p eventValueBuffer is NULL
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if size of \p eventValueBuffer
 * is not sufficient
 */
CUptiResult CUPTIAPI cuptiEventGroupReadEvent(CUpti_EventGroup eventGroup,
                                              CUpti_ReadEventFlags flags,
                                              CUpti_EventID event,
                                              size_t *eventValueBufferSizeBytes,
                                              uint64_t *eventValueBuffer);

/**
 * \brief Read the values for all the events in an event group.
 *
 * Read the values for all the events in an event group. The event
 * values are returned in the \p eventValueBuffer buffer. \p
 * eventValueBufferSizeBytes indicates the size of \p
 * eventValueBuffer.  The buffer must be at least (sizeof(uint64) *
 * number of events in group) if
 * ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is not set on
 * the group containing the events.  The buffer must be at least
 * (sizeof(uint64) * number of domain instances * number of events in
 * group) if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is
 * set on the group.
 *
 * The data format returned in \p eventValueBuffer is:
 *    - domain instance 0: event0 event1 ... eventN
 *    - domain instance 1: event0 event1 ... eventN
 *    - ...
 *    - domain instance M: event0 event1 ... eventN
 *
 * The event order in \p eventValueBuffer is returned in \p
 * eventIdArray. The size of \p eventIdArray is specified in \p
 * eventIdArraySizeBytes. The size should be at least
 * (sizeof(CUpti_EventID) * number of events in group).
 *
 * If any instance of any event counter overflows, the value returned
 * for that event instance will be ::CUPTI_EVENT_OVERFLOW.
 *
 * The only allowed value for \p flags is ::CUPTI_EVENT_READ_FLAG_NONE.
 *
 * Reading events from a disabled event group is not allowed. After
 * being read, an event's value is reset to zero.
 * \note \b Thread-safety: this function is thread safe but client
 * must guard against simultaneous destruction or modification of \p
 * eventGroup (for example, client must guard against simultaneous
 * calls to \ref cuptiEventGroupDestroy, \ref cuptiEventGroupAddEvent,
 * etc.), and must guard against simultaneous destruction of the
 * context in which \p eventGroup was created (for example, client
 * must guard against simultaneous calls to cudaDeviceReset,
 * cuCtxDestroy, etc.). If \ref cuptiEventGroupResetAllEvents is
 * called simultaneously with this function, then returned event
 * values are undefined.
 *
 * \param eventGroup The event group
 * \param flags Flags controlling the reading mode
 * \param eventValueBufferSizeBytes The size of \p eventValueBuffer in
 * bytes, and returns the number of bytes written to \p
 * eventValueBuffer
 * \param eventValueBuffer Returns the event values
 * \param eventIdArraySizeBytes The size of \p eventIdArray in bytes,
 * and returns the number of bytes written to \p eventIdArray
 * \param eventIdArray Returns the IDs of the events in the same order
 * as the values return in eventValueBuffer.
 * \param numEventIdsRead Returns the number of event IDs returned
 * in \p eventIdArray
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_OPERATION if \p eventGroup is disabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroup, \p
 * eventValueBufferSizeBytes, \p eventValueBuffer, \p
 * eventIdArraySizeBytes, \p eventIdArray or \p numEventIdsRead is
 * NULL
 * \retval CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT if size of \p eventValueBuffer
 * or \p eventIdArray is not sufficient
 */
CUptiResult CUPTIAPI cuptiEventGroupReadAllEvents(CUpti_EventGroup       eventGroup,
                                                  CUpti_ReadEventFlags   flags,
                                                  size_t                 *eventValueBufferSizeBytes,
                                                  uint64_t               *eventValueBuffer,
                                                  size_t                 *eventIdArraySizeBytes,
                                                  CUpti_EventID          *eventIdArray,
                                                  size_t                 *numEventIdsRead);

/**
 * \brief For a set of events, get the grouping that indicates the
 * number of passes and the event groups necessary to collect the
 * events.
 *
 * The number of events that can be collected simultaneously varies by
 * device and by the type of the events. When events can be collected
 * simultaneously, they may need to be grouped into multiple event
 * groups because they are from different event domains. This function
 * takes a set of events and determines how many passes are required
 * to collect all those events, and which events can be collected
 * simultaneously in each pass.
 *
 * The CUpti_EventGroupSets returned in \p eventGroupPasses indicates
 * how many passes are required to collect the events with the \p
 * numSets field. Within each event group set, the \p sets array
 * indicates the event groups that should be collected on each pass.
 * \note \b Thread-safety: this function is thread safe, but client
 * must guard against another thread simultaneously destroying \p
 * context.
 *
 * \param context The context for event collection
 * \param eventIdArraySizeBytes Size of \p eventIdArray in bytes
 * \param eventIdArray Array of event IDs that need to be grouped
 * \param eventGroupPasses Returns a CUpti_EventGroupSets object that
 * indicates the number of passes required to collect the events and
 * the events to collect on each pass
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_EVENT_ID
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventIdArray or
 * \p eventGroupPasses is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupSetsCreate(CUcontext context,
                                               size_t eventIdArraySizeBytes,
                                               CUpti_EventID *eventIdArray,
                                               CUpti_EventGroupSets **eventGroupPasses);

/**
 * \brief Destroy a event group sets object.
 *
 * Destroy a CUpti_EventGroupSets object.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroupSets The object to destroy
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_INVALID_OPERATION if any of the event groups
 * contained in the sets is enabled
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroupSets is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupSetsDestroy(CUpti_EventGroupSets *eventGroupSets);


/**
 * \brief Enable an event group set.
 *
 * Enable a set of event groups. Enabling a set of event groups zeros the value of
 * all the events in all the groups and then starts collection of those events.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param eventGroupSet The pointer to the event group set
 *
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_NOT_READY if \p eventGroup does not contain any events
 * \retval CUPTI_ERROR_NOT_COMPATIBLE if \p eventGroup cannot be
 * enabled due to other already enabled event groups
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroupSet is NULL
 * \retval CUPTI_ERROR_HARDWARE_BUSY if other client is profiling and hardware is
 * busy
 */
CUptiResult CUPTIAPI cuptiEventGroupSetEnable(CUpti_EventGroupSet *eventGroupSet);

/**
 * \brief Disable an event group set.
 *
 * Disable a set of event groups. Disabling a set of event groups
 * stops collection of events contained in the groups.
 * \note \b Thread-safety: this function is thread safe.
 * \note \b If this call fails, some of the event groups in the set may be disabled
 * and other event groups may remain enabled.
 *
 * \param eventGroupSet The pointer to the event group set
 * \retval CUPTI_SUCCESS
 * \retval CUPTI_ERROR_NOT_INITIALIZED
 * \retval CUPTI_ERROR_HARDWARE
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p eventGroupSet is NULL
 */
CUptiResult CUPTIAPI cuptiEventGroupSetDisable(CUpti_EventGroupSet *eventGroupSet);

/**
 * \brief Enable kernel replay mode.
 *
 * Set profiling mode for the context to replay mode. In this mode,
 * any number of events can be collected in one run of the kernel. The
 * event collection mode will automatically switch to
 * CUPTI_EVENT_COLLECTION_MODE_KERNEL.  In this mode, \ref
 * cuptiSetEventCollectionMode will return
 * CUPTI_ERROR_INVALID_OPERATION.
 * \note \b Kernels might take longer to run if many events are enabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context
 * \retval CUPTI_SUCCESS
 */
CUptiResult CUPTIAPI cuptiEnableKernelReplayMode(CUcontext context);

/**
 * \brief Disable kernel replay mode.
 *
 * Set profiling mode for the context to non-replay (default)
 * mode. Event collection mode will be set to
 * CUPTI_EVENT_COLLECTION_MODE_KERNEL.  All previously enabled
 * event groups and event group sets will be disabled.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param context The context
 * \retval CUPTI_SUCCESS
 */
CUptiResult CUPTIAPI cuptiDisableKernelReplayMode(CUcontext context);

/**
 * \brief Function type for getting updates on kernel replay.
 *
 * \param kernelName The mangled kernel name
 * \param numReplaysDone Number of replays done so far
 * \param customData Pointer of any custom data passed in when subscribing
 */
typedef void (CUPTIAPI *CUpti_KernelReplayUpdateFunc)(
    const char *kernelName,
    int numReplaysDone,
    void *customData);

/**
 * \brief Subscribe to kernel replay updates.
 *
 * When subscribed, the function pointer passed in will be called each time a
 * kernel run is finished during kernel replay. Previously subscribed function
 * pointer will be replaced. Pass in NULL as the function pointer unsubscribes
 * the update.
 *
 * \param updateFunc The update function pointer
 * \param customData Pointer to any custom data
 * \retval CUPTI_SUCCESS
 */
CUptiResult CUPTIAPI cuptiKernelReplaySubscribeUpdate(CUpti_KernelReplayUpdateFunc updateFunc, void *customData);

/** @} */ /* END CUPTI_EVENT_API */

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_CUPTI_EVENTS_H_*/
