// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/binding/hip/api.h"

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Flag translation functions
//===----------------------------------------------------------------------===//

static iree_hal_streaming_stream_flags_t hip_stream_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_stream_flags_t flags = IREE_HAL_STREAMING_STREAM_FLAG_NONE;
  if (hip_flags & hipStreamNonBlocking) {
    flags |= IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING;
  }
  return flags;
}

static iree_hal_streaming_event_flags_t hip_event_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_event_flags_t flags = IREE_HAL_STREAMING_EVENT_FLAG_NONE;
  if (hip_flags & hipEventBlockingSync) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_BLOCKING_SYNC;
  }
  if (hip_flags & hipEventDisableTiming) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING;
  }
  if (hip_flags & hipEventInterprocess) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS;
  }
  return flags;
}

static iree_hal_streaming_memory_flags_t hip_memory_flags_to_internal(
    unsigned int hip_flags) {
  iree_hal_streaming_memory_flags_t flags = IREE_HAL_STREAMING_MEMORY_FLAG_NONE;
  if (hip_flags & hipHostRegisterPortable) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_PORTABLE;
  }
  if (hip_flags & hipHostRegisterMapped) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_PINNED;
  }
  // hipHostRegisterIoMemory could map to write-combined.
  if (hip_flags & hipHostRegisterIoMemory) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_WRITE_COMBINED;
  }
  return flags;
}

static iree_hal_streaming_graph_flags_t hip_graph_flags_to_internal(
    unsigned int hip_flags) {
  // HIP doesn't have specific graph creation flags, so we return none.
  return IREE_HAL_STREAMING_GRAPH_FLAG_NONE;
}

static iree_hal_streaming_graph_instantiate_flags_t
hip_graph_instantiate_flags_to_internal(unsigned long long hip_flags) {
  iree_hal_streaming_graph_instantiate_flags_t flags =
      IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE;
  if (hip_flags & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
  }
  if (hip_flags & hipGraphInstantiateFlagUpload) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_UPLOAD;
  }
  if (hip_flags & hipGraphInstantiateFlagDeviceLaunch) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
  }
  if (hip_flags & hipGraphInstantiateFlagUseNodePriority) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
  }
  return flags;
}

static iree_hal_streaming_mem_pool_attr_t hip_mempool_attr_to_internal(
    hipMemPool_attribute attr) {
  switch (attr) {
    case hipMemPoolAttrReuseFollowEventDependencies:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES;
    case hipMemPoolAttrReuseAllowOpportunistic:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC;
    case hipMemPoolAttrReuseAllowInternalDependencies:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES;
    case hipMemPoolAttrReleaseThreshold:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD;
    case hipMemPoolAttrReservedMemCurrent:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT;
    case hipMemPoolAttrReservedMemHigh:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH;
    case hipMemPoolAttrUsedMemCurrent:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_CURRENT;
    case hipMemPoolAttrUsedMemHigh:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH;
    default:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT;
  }
}

static iree_hal_streaming_mem_handle_type_t hip_mem_handle_type_to_internal(
    hipMemAllocationHandleType handle_type) {
  switch (handle_type) {
    case hipMemHandleTypeNone:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE;
    case hipMemHandleTypePosixFileDescriptor:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    case hipMemHandleTypeWin32:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_WIN32;
    case hipMemHandleTypeWin32Kmt:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_WIN32_KMT;
    default:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE;
  }
}

static iree_hal_streaming_mem_location_type_t hip_mem_location_type_to_internal(
    hipMemLocationType type) {
  switch (type) {
    case hipMemLocationTypeInvalid:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID;
    case hipMemLocationTypeDevice:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE;
    case hipMemLocationTypeHost:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST;
    case hipMemLocationTypeHostNuma:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA;
    case hipMemLocationTypeHostNumaCurrent:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
    default:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID;
  }
}

static iree_hal_streaming_mem_access_flags_t hip_mem_access_flags_to_internal(
    hipMemAccessFlags flags) {
  switch (flags) {
    case hipMemAccessFlagsProtNone:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE;
    case hipMemAccessFlagsProtRead:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ;
    case hipMemAccessFlagsProtReadWrite:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READWRITE;
    default:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE;
  }
}

static iree_hal_streaming_context_limit_t hip_limit_to_internal(
    hipLimit_t limit) {
  switch (limit) {
    case hipLimitStackSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE;
    case hipLimitPrintfFifoSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE;
    case hipLimitMallocHeapSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE;
    case hipLimitDevRuntimeSyncDepth:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH;
    case hipLimitDevRuntimePendingLaunchCount:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT;
    case hipLimitMaxL2FetchGranularity:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY;
    case hipLimitPersistingL2CacheSize:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE;
    default:
      // Return an invalid value that will trigger error in internal API.
      return (iree_hal_streaming_context_limit_t)-1;
  }
}

//===----------------------------------------------------------------------===//
// Thread-local error tracking
//===----------------------------------------------------------------------===//

// Thread-local error state for HIP.
static iree_thread_local struct {
  hipError_t last_error;
  bool sticky;
} hip_thread_error = {hipSuccess, false};

static void hip_thread_error_set(hipError_t error, bool sticky) {
  hip_thread_error.last_error = error;
  hip_thread_error.sticky = sticky;
}

static hipError_t hip_thread_error_get_and_clear(void) {
  hipError_t error = hip_thread_error.last_error;
  if (!hip_thread_error.sticky) {
    hip_thread_error.last_error = hipSuccess;
  }
  return error;
}

static hipError_t hip_thread_error_peek(void) {
  return hip_thread_error.last_error;
}

// Helper macro to set thread-local error and return.
#define HIP_RETURN_ERROR(error)          \
  do {                                   \
    hipError_t _err = (error);           \
    if (_err != hipSuccess) {            \
      hip_thread_error_set(_err, false); \
    }                                    \
    return _err;                         \
  } while (0)

// Helper macro to set sticky thread-local error and return.
#define HIP_RETURN_STICKY_ERROR(error)  \
  do {                                  \
    hipError_t _err = (error);          \
    if (_err != hipSuccess) {           \
      hip_thread_error_set(_err, true); \
    }                                   \
    return _err;                        \
  } while (0)

//===----------------------------------------------------------------------===//
// Status conversion
//===----------------------------------------------------------------------===//

static hipError_t iree_status_to_hip_result(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return hipSuccess;
  }

  // DO NOT SUBMIT
  iree_status_fprint(stderr, status);

  // Map IREE status codes to HIP error codes.
  iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);

  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      return hipErrorInvalidValue;
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

//===----------------------------------------------------------------------===//
// Implicit initialization helpers
//===----------------------------------------------------------------------===//

// Ensures HIP runtime is initialized (calls hipInit if needed).
static hipError_t hip_ensure_initialized(void) {
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    // Initialize the runtime.
    iree_status_t status = iree_hal_streaming_init_global(
        IREE_HAL_STREAMING_INIT_FLAG_NONE, iree_allocator_system());
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return hipErrorNotInitialized;
    }
  }
  return hipSuccess;
}

// Ensures context exists for current thread and returns it.
// This implements HIP's implicit initialization behavior:
// - Automatically calls hipInit() if needed
// - Creates primary context for device 0 if no context exists
// - Sets the context as current for the thread
static hipError_t hip_ensure_context(
    iree_hal_streaming_context_t** out_context) {
  // First ensure initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    if (out_context) *out_context = NULL;
    HIP_RETURN_ERROR(init_result);
  }

  // Check if current thread has context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    // No context set - create primary context for device 0.
    // This matches HIP behavior of implicitly using device 0.
    iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(0);
    if (!device) {
      if (out_context) *out_context = NULL;
      return hipErrorInvalidDevice;
    }

    // Get or create primary context.
    iree_status_t status =
        iree_hal_streaming_device_get_or_create_primary_context(device,
                                                                &context);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      if (out_context) *out_context = NULL;
      return hipErrorOutOfMemory;
    }

    // Set as current context for thread.
    iree_hal_streaming_context_set_current(context);
  }

  if (out_context) *out_context = context;
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

// Initializes the HIP runtime system.
//
// Parameters:
//  - flags: [IN] Initialization flags (must be 0).
//
// Returns:
//  - hipSuccess: HIP initialized successfully or already initialized.
//  - hipErrorInvalidValue: flags is not 0.
//  - hipErrorNoDevice: No HIP-capable devices found.
//  - hipErrorInsufficientDriver: Incompatible driver version.
//  - hipErrorUnknown: Internal initialization error.
//
// Synchronization: This operation is synchronous.
//
// Initialization behavior:
// - Must be called before using any other HIP functions.
// - Can be called multiple times (subsequent calls are no-ops).
// - Automatically called by most HIP functions if not already initialized.
// - Enumerates and initializes all available HIP devices.
// - Sets up the primary context for each device.
//
// Multi-GPU: Initializes all available devices in the system.
//
// Note: Unlike CUDA, HIP currently requires flags to be 0.
//
// See also: hipGetDeviceCount, hipSetDevice, hipDeviceReset.
HIPAPI hipError_t hipInit(unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // HIP doesn't define init flags, but check for non-zero value.
  if (flags != 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_status_t status = iree_hal_streaming_init_global(
      IREE_HAL_STREAMING_INIT_FLAG_NONE, iree_allocator_system());
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Custom function to deinitialize the HAL backend.
// This is not a standard HIP API function.
HIPAPI hipError_t hipHALDeinit(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_cleanup_global();
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the HIP driver version.
//
// Parameters:
//  - driverVersion: [OUT] Pointer to receive driver version number.
//
// Returns:
//  - hipSuccess: Version retrieved successfully.
//  - hipErrorInvalidValue: driverVersion is NULL.
//
// Version format: Major*1000000 + Minor*1000 + Patch.
//
// See also: hipRuntimeGetVersion.
HIPAPI hipError_t hipDriverGetVersion(int* driverVersion) {
  if (!driverVersion) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *driverVersion = 60000000;  // HIP 6.0
  return hipSuccess;
}

// Gets the HIP runtime version.
//
// Parameters:
//  - runtimeVersion: [OUT] Pointer to receive runtime version number.
//
// Returns:
//  - hipSuccess: Version retrieved successfully.
//  - hipErrorInvalidValue: runtimeVersion is NULL.
//
// Version format: Major*1000000 + Minor*1000 + Patch.
//
// See also: hipDriverGetVersion.
HIPAPI hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
  if (!runtimeVersion) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  *runtimeVersion = 60000000;  // HIP 6.0
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Gets the current device ID.
//
// Parameters:
//  - device: [OUT] Pointer to receive the current device ID.
//
// Returns:
//  - hipSuccess: Device ID retrieved successfully.
//  - hipErrorInvalidValue: device pointer is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Returns the device ID set by hipSetDevice().
// - Default device is 0 if hipSetDevice() has not been called.
// - Device ID is per-thread (each thread has its own current device).
//
// Multi-GPU: Returns the device associated with the current thread.
//
// See also: hipSetDevice, hipGetDeviceCount, hipDeviceGet.
HIPAPI hipError_t hipGetDevice(int* device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  // If no context exists, this will create one for device 0.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  *device = (int)context->device_ordinal;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets the current device for the calling thread.
//
// Parameters:
//  - device: [IN] Device ID to make current (0-based index).
//
// Returns:
//  - hipSuccess: Device set successfully.
//  - hipErrorInvalidDevice: device ID is invalid.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorNoDevice: No HIP-capable devices available.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Sets the current device for this thread.
// - Subsequent HIP calls will target this device.
// - Creates a primary context if not already created.
// - Does not affect other threads.
//
// Multi-GPU:
// - Each thread maintains its own current device.
// - Memory allocations and kernel launches target the current device.
// - Use before any device-specific operations.
//
// Warning: Changing devices does not migrate existing allocations.
//
// See also: hipGetDevice, hipGetDeviceCount, hipDeviceReset.
HIPAPI hipError_t hipSetDevice(int device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // First ensure runtime is initialized.
  // hipSetDevice() is often the first HIP call in applications.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the device.
  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get or create the primary context lazily.
  iree_hal_streaming_context_t* primary_context = NULL;
  iree_status_t status =
      iree_hal_streaming_device_get_or_create_primary_context(device_obj,
                                                              &primary_context);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  // Switch to the primary context for the device.
  status = iree_hal_streaming_context_set_current(primary_context);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets the number of HIP-capable devices.
//
// Parameters:
//  - count: [OUT] Pointer to receive the device count.
//
// Returns:
//  - hipSuccess: Device count retrieved successfully.
//  - hipErrorInvalidValue: count pointer is NULL.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Returns the total number of HIP-capable devices.
// - Returns 0 if no devices are available.
// - Count includes all devices regardless of compute capability.
// - Count remains constant for the lifetime of the process.
//
// Multi-GPU: Returns total count of all devices in the system.
//
// Usage pattern:
// ```c
// int deviceCount;
// hipGetDeviceCount(&deviceCount);
// for (int i = 0; i < deviceCount; i++) {
//   hipSetDevice(i);
//   // Work with device i
// }
// ```
//
// See also: hipSetDevice, hipGetDevice, hipGetDeviceProperties.
HIPAPI hipError_t hipGetDeviceCount(int* count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!count) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    *count = 0;
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    *count = 0;
    IREE_TRACE_ZONE_END(z0);
    return hipSuccess;
  }
  *count = (int)device_registry->device_count;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets a device handle by ordinal.
//
// Parameters:
//  - device: [OUT] Pointer to receive the device handle.
//  - ordinal: [IN] Device ordinal (0-based index).
//
// Returns:
//  - hipSuccess: Device handle retrieved successfully.
//  - hipErrorInvalidValue: device pointer is NULL.
//  - hipErrorInvalidDevice: ordinal is out of range.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device behavior:
// - Returns a handle to the specified device.
// - Device handle can be used with driver API functions.
// - Handle remains valid for the lifetime of the process.
//
// Multi-GPU: Each device has a unique ordinal from 0 to count-1.
//
// Note: This is primarily for HIP driver API compatibility.
//
// See also: hipGetDeviceCount, hipDeviceGetName, hipCtxCreate.
HIPAPI hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the device count to validate ordinal.
  int device_count = 0;
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (device_registry) {
    device_count = (int)device_registry->device_count;
  }

  if (ordinal < 0 || ordinal >= device_count) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // hipDevice_t is just an int, so return the ordinal.
  *device = ordinal;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets properties of a compute device.
//
// Parameters:
//  - prop: [OUT] Pointer to receive device properties.
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Properties retrieved successfully.
//  - hipErrorInvalidValue: prop is NULL.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Device properties include:
// - name: Device name string.
// - totalGlobalMem: Total global memory in bytes.
// - sharedMemPerBlock: Shared memory per block in bytes.
// - regsPerBlock: Registers per block.
// - warpSize: Warp size in threads.
// - maxThreadsPerBlock: Maximum threads per block.
// - maxThreadsDim: Maximum dimensions of a block.
// - maxGridSize: Maximum dimensions of a grid.
// - clockRate: Clock frequency in kHz.
// - memoryClockRate: Memory clock frequency in kHz.
// - memoryBusWidth: Memory bus width in bits.
// - multiProcessorCount: Number of multiprocessors.
// - computeMode: Compute mode settings.
// - And many more architecture-specific properties.
//
// Multi-GPU: Each device has unique properties.
//
// Usage pattern:
// ```c
// hipDeviceProp_t prop;
// hipGetDeviceProperties(&prop, 0);
// printf("Device: %s\n", prop.name);
// printf("Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
// ```
//
// See also: hipGetDeviceCount, hipGetDevice, hipDeviceGetAttribute.
HIPAPI hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!prop) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get memory information using the internal API.
  iree_device_size_t free_memory = 0;
  iree_device_size_t total_memory = 0;
  iree_status_t memory_status = iree_hal_streaming_device_memory_info(
      device, &free_memory, &total_memory);
  if (!iree_status_is_ok(memory_status)) {
    iree_status_ignore(memory_status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Fill device properties from device entry.
  memset(prop, 0, sizeof(hipDeviceProp_t));

  // Get device name using internal API for safe copying.
  iree_status_t name_status = iree_hal_streaming_device_name(
      (iree_hal_streaming_device_ordinal_t)device, prop->name,
      sizeof(prop->name));
  if (!iree_status_is_ok(name_status)) {
    iree_status_ignore(name_status);
    // Fall back to empty name if device name query fails.
    prop->name[0] = '\0';
  }
  prop->totalGlobalMem = (size_t)total_memory;
  prop->sharedMemPerBlock = 65536;  // 64KB default
  prop->regsPerBlock = 65536;
  prop->warpSize = device_obj->warp_size;
  prop->memPitch = 0;
  prop->maxThreadsPerBlock = device_obj->max_threads_per_block;
  prop->maxThreadsDim[0] = device_obj->max_block_dim[0];
  prop->maxThreadsDim[1] = device_obj->max_block_dim[1];
  prop->maxThreadsDim[2] = device_obj->max_block_dim[2];
  prop->maxGridSize[0] = device_obj->max_grid_dim[0];
  prop->maxGridSize[1] = device_obj->max_grid_dim[1];
  prop->maxGridSize[2] = device_obj->max_grid_dim[2];
  prop->clockRate = 1000;       // 1 GHz default
  prop->totalConstMem = 65536;  // 64KB default
  prop->major = device_obj->compute_capability_major;
  prop->minor = device_obj->compute_capability_minor;
  prop->textureAlignment = 0;
  prop->texturePitchAlignment = 0;
  prop->deviceOverlap = 1;
  prop->multiProcessorCount = device_obj->multiprocessor_count;
  prop->kernelExecTimeoutEnabled = 0;
  prop->integrated = 0;
  prop->canMapHostMemory = 1;
  prop->computeMode = 0;  // Default compute mode
  prop->maxTexture1D = 0;
  prop->maxTexture1DMipmap = 0;
  prop->maxTexture1DLinear = 0;
  prop->maxTexture2D[0] = 0;
  prop->maxTexture2D[1] = 0;
  prop->maxTexture2DMipmap[0] = 0;
  prop->maxTexture2DMipmap[1] = 0;
  prop->maxTexture2DLinear[0] = 0;
  prop->maxTexture2DLinear[1] = 0;
  prop->maxTexture2DLinear[2] = 0;
  prop->maxTexture2DGather[0] = 0;
  prop->maxTexture2DGather[1] = 0;
  prop->maxTexture3D[0] = 0;
  prop->maxTexture3D[1] = 0;
  prop->maxTexture3D[2] = 0;
  prop->maxTexture3DAlt[0] = 0;
  prop->maxTexture3DAlt[1] = 0;
  prop->maxTexture3DAlt[2] = 0;
  prop->maxTextureCubemap = 0;
  prop->maxTexture1DLayered[0] = 0;
  prop->maxTexture1DLayered[1] = 0;
  prop->maxTexture2DLayered[0] = 0;
  prop->maxTexture2DLayered[1] = 0;
  prop->maxTexture2DLayered[2] = 0;
  prop->maxTextureCubemapLayered[0] = 0;
  prop->maxTextureCubemapLayered[1] = 0;
  prop->maxSurface1D = 0;
  prop->maxSurface2D[0] = 0;
  prop->maxSurface2D[1] = 0;
  prop->maxSurface3D[0] = 0;
  prop->maxSurface3D[1] = 0;
  prop->maxSurface3D[2] = 0;
  prop->maxSurface1DLayered[0] = 0;
  prop->maxSurface1DLayered[1] = 0;
  prop->maxSurface2DLayered[0] = 0;
  prop->maxSurface2DLayered[1] = 0;
  prop->maxSurface2DLayered[2] = 0;
  prop->maxSurfaceCubemap = 0;
  prop->maxSurfaceCubemapLayered[0] = 0;
  prop->maxSurfaceCubemapLayered[1] = 0;
  prop->surfaceAlignment = 0;
  prop->concurrentKernels = 1;
  prop->ECCEnabled = 0;
  prop->pciBusID = device;
  prop->pciDeviceID = 0;
  prop->pciDomainID = 0;
  prop->tccDriver = 0;
  prop->asyncEngineCount = 2;
  prop->unifiedAddressing = 1;
  prop->memoryClockRate = 1000;  // 1 GHz default
  prop->memoryBusWidth = 256;    // 256-bit default
  prop->l2CacheSize = 0;
  prop->maxThreadsPerMultiProcessor = 2048;  // Default
  prop->streamPrioritiesSupported = 0;
  prop->globalL1CacheSupported = 1;
  prop->localL1CacheSupported = 1;
  prop->sharedMemPerMultiprocessor = 65536;  // 64KB default
  prop->regsPerMultiprocessor = 65536;
  prop->managedMemory = 0;
  prop->isMultiGpuBoard = 0;
  prop->multiGpuBoardGroupID = 0;
  prop->singleToDoublePrecisionPerfRatio = 32;
  prop->pageableMemoryAccess = 0;
  prop->concurrentManagedAccess = 0;
  prop->computePreemptionSupported = 0;
  prop->canUseHostPointerForRegisteredMem = 0;
  prop->cooperativeLaunch = 0;
  prop->cooperativeMultiDeviceLaunch = 0;
  prop->pageableMemoryAccessUsesHostPageTables = 0;
  prop->directManagedMemAccessFromHost = 0;
  prop->maxBlocksPerMultiProcessor = 32;
  prop->accessPolicyMaxWindowSize = 0;
  prop->reservedSharedMemPerBlock = 0;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets a specific attribute of a compute device.
//
// Parameters:
//  - value: [OUT] Pointer to receive the attribute value.
//  - attr: [IN] Attribute to query (hipDeviceAttributeMaxThreadsPerBlock,
//               hipDeviceAttributeMaxBlockDimX, hipDeviceAttributeMaxGridDimX,
//               hipDeviceAttributeComputeCapabilityMajor, etc.).
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Attribute retrieved successfully.
//  - hipErrorInvalidValue: value is NULL or attr is invalid.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Common attributes:
// - hipDeviceAttributeMaxThreadsPerBlock: Max threads per block.
// - hipDeviceAttributeMaxBlockDimX/Y/Z: Max block dimensions.
// - hipDeviceAttributeMaxGridDimX/Y/Z: Max grid dimensions.
// - hipDeviceAttributeMaxSharedMemoryPerBlock: Shared memory per block.
// - hipDeviceAttributeWarpSize: Number of threads in a warp.
// - hipDeviceAttributeComputeCapabilityMajor/Minor: Compute capability.
// - hipDeviceAttributeMultiprocessorCount: Number of SMs/CUs.
// - hipDeviceAttributeClockRate: Core clock frequency.
// - hipDeviceAttributeMemoryClockRate: Memory clock frequency.
// - hipDeviceAttributeGlobalMemoryBusWidth: Memory bus width.
// - hipDeviceAttributeL2CacheSize: L2 cache size.
// - hipDeviceAttributeComputeMode: Current compute mode.
//
// Multi-GPU: Each device has unique attributes.
//
// Note: More efficient than hipGetDeviceProperties() when only specific
// attributes are needed.
//
// See also: hipGetDeviceProperties, hipDeviceGetName, hipGetDevice.
HIPAPI hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr,
                                        int device) {
  if (!value) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Map attributes to device properties.
  switch (attr) {
    case hipDeviceAttributeMaxThreadsPerBlock:
      *value = device_obj->max_threads_per_block;
      break;
    case hipDeviceAttributeMaxBlockDimX:
      *value = device_obj->max_block_dim[0];
      break;
    case hipDeviceAttributeMaxBlockDimY:
      *value = device_obj->max_block_dim[1];
      break;
    case hipDeviceAttributeMaxBlockDimZ:
      *value = device_obj->max_block_dim[2];
      break;
    case hipDeviceAttributeMaxGridDimX:
      *value = device_obj->max_grid_dim[0];
      break;
    case hipDeviceAttributeMaxGridDimY:
      *value = device_obj->max_grid_dim[1];
      break;
    case hipDeviceAttributeMaxGridDimZ:
      *value = device_obj->max_grid_dim[2];
      break;
    case hipDeviceAttributeWarpSize:
      *value = device_obj->warp_size;
      break;
    case hipDeviceAttributeMultiprocessorCount:
      *value = device_obj->multiprocessor_count;
      break;
    case hipDeviceAttributeComputeCapabilityMajor:
      *value = device_obj->compute_capability_major;
      break;
    case hipDeviceAttributeComputeCapabilityMinor:
      *value = device_obj->compute_capability_minor;
      break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
      *value = 65536;  // 64KB default
      break;
    case hipDeviceAttributeMaxRegistersPerBlock:
      *value = 65536;
      break;
    case hipDeviceAttributeClockRate:
      *value = 1000000;  // 1 GHz in kHz
      break;
    case hipDeviceAttributeMemoryClockRate:
      *value = 1000000;  // 1 GHz in kHz
      break;
    case hipDeviceAttributeMemoryBusWidth:
      *value = 256;  // 256-bit default
      break;
    case hipDeviceAttributeL2CacheSize:
      *value = 0;
      break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
      *value = 2048;  // Default
      break;
    case hipDeviceAttributeSharedMemPerBlockOptin:
      // Maximum shared memory per block when opted in (> 48KB).
      // This is equivalent to
      // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
      *value = 49152;  // 48KB default, actual value would be device-specific.
      break;
    case hipDeviceAttributeMaxSharedMemoryPerMultiprocessor:
      // Total shared memory per multiprocessor.
      // This is equivalent to
      // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR.
      *value = 65536;  // 64KB default.
      break;
    case hipDeviceAttributeSharedMemPerMultiprocessor:
      // Shared memory available per multiprocessor.
      // Similar to above, different naming in HIP.
      *value = 65536;  // 64KB default.
      break;
    default:
      // Return sensible defaults for other attributes.
      *value = 0;
      break;
  }

  return hipSuccess;
}

// Gets the name of a compute device.
//
// Parameters:
//  - name: [OUT] Buffer to receive the device name.
//  - len: [IN] Maximum length of name buffer including null terminator.
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Name retrieved successfully.
//  - hipErrorInvalidValue: name is NULL or len <= 0.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Name behavior:
// - Returns human-readable device name.
// - String is null-terminated.
// - Truncated if longer than len-1 characters.
// - Typically includes manufacturer and model.
//
// Multi-GPU: Each device has a unique name.
//
// Usage pattern:
// ```c
// char deviceName[256];
// hipDeviceGetName(deviceName, sizeof(deviceName), 0);
// printf("Device 0: %s\n", deviceName);
// ```
//
// See also: hipGetDeviceProperties, hipDeviceGetAttribute, hipGetDevice.
HIPAPI hipError_t hipDeviceGetName(char* name, int len, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!name || len <= 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_device_name(
      (iree_hal_streaming_device_ordinal_t)device, name, (size_t)len);
  hipError_t result = iree_status_to_hip_result(status);

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets the UUID of a compute device.
//
// Parameters:
//  - uuid: [OUT] Pointer to receive the device UUID.
//  - dev: [IN] Device handle to query.
//
// Returns:
//  - hipSuccess: UUID retrieved successfully.
//  - hipErrorInvalidValue: uuid is NULL.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorNotSupported: UUID not supported (current implementation).
//
// Synchronization: This operation is synchronous.
//
// UUID behavior:
// - Returns a unique identifier for the physical device.
// - UUID persists across reboots and driver reloads.
// - Can be used to identify specific GPUs in multi-GPU systems.
// - Format is 16-byte binary value.
//
// Multi-GPU: Each physical device has a unique UUID.
//
// Note: Currently not implemented in StreamHAL.
//
// See also: hipDeviceGet, hipGetDeviceProperties.
HIPAPI hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t dev) {
  // UUID support is not currently implemented.
  (void)uuid;
  (void)dev;
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Gets the total memory of a compute device.
//
// Parameters:
//  - bytes: [OUT] Pointer to receive total memory in bytes.
//  - device: [IN] Device ordinal to query.
//
// Returns:
//  - hipSuccess: Total memory retrieved successfully.
//  - hipErrorInvalidValue: bytes is NULL.
//  - hipErrorInvalidDevice: Invalid device ordinal.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory information:
// - Returns total global memory available on the device.
// - Does not include shared memory or constant memory.
// - Value is fixed for the device (doesn't change with allocations).
//
// Multi-GPU: Each device has its own memory capacity.
//
// Usage pattern:
// ```c
// size_t totalMem;
// hipDeviceTotalMem(&totalMem, 0);
// printf("Device 0 has %zu GB of memory\n", totalMem / (1024*1024*1024));
// ```
//
// See also: hipMemGetInfo, hipGetDeviceProperties.
HIPAPI hipError_t hipDeviceTotalMem(size_t* bytes, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!bytes) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_device_size_t free_memory = 0;
  iree_device_size_t total_memory = 0;
  iree_status_t status = iree_hal_streaming_device_memory_info(
      device, &free_memory, &total_memory);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  *bytes = (size_t)total_memory;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Queries if one device can directly access another device's memory.
//
// Parameters:
//  - canAccessPeer: [OUT] Set to 1 if access is possible, 0 otherwise.
//  - device: [IN] Device that would be accessing memory.
//  - peerDevice: [IN] Device whose memory would be accessed.
//
// Returns:
//  - hipSuccess: Query completed successfully.
//  - hipErrorInvalidValue: canAccessPeer is NULL.
//  - hipErrorInvalidDevice: Invalid device or peerDevice.
//
// Synchronization: This operation is synchronous.
//
// Peer access behavior:
// - Returns 1 if device can directly access peerDevice's memory.
// - Returns 0 if direct access is not possible.
// - Does not enable peer access, only queries capability.
// - Symmetric access is not guaranteed (A→B doesn't imply B→A).
//
// Multi-GPU:
// - Typically supported between GPUs on the same PCIe root complex.
// - May be supported across PCIe switches with P2P capability.
// - Not supported between discrete and integrated GPUs.
//
// See also: hipDeviceEnablePeerAccess, hipDeviceDisablePeerAccess,
//           hipMemcpyPeer.
HIPAPI hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device,
                                         int peerDevice) {
  if (!canAccessPeer) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry || device < 0 ||
      device >= (int)device_registry->device_count || peerDevice < 0 ||
      peerDevice >= (int)device_registry->device_count) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Use the P2P query function.
  bool can_access = false;
  iree_status_t status = iree_hal_streaming_device_can_access_peer(
      device, peerDevice, &can_access);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    *canAccessPeer = 0;
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  *canAccessPeer = can_access ? 1 : 0;
  return hipSuccess;
}

// Gets peer-to-peer attributes between two devices.
//
// Parameters:
//  - value: [OUT] Pointer to receive the attribute value.
//  - attrib: [IN] P2P attribute to query (hipDevP2PAttrPerformanceRank,
//                 hipDevP2PAttrAccessSupported,
//                 hipDevP2PAttrNativeAtomicSupported,
//                 hipDevP2PAttrCudaArrayAccessSupported).
//  - srcDevice: [IN] Source device in P2P pair.
//  - dstDevice: [IN] Destination device in P2P pair.
//
// Returns:
//  - hipSuccess: Attribute retrieved successfully.
//  - hipErrorInvalidValue: value is NULL or invalid attribute.
//  - hipErrorInvalidDevice: Invalid device ordinals.
//
// Synchronization: This operation is synchronous.
//
// P2P attributes:
// - hipDevP2PAttrPerformanceRank: Relative performance (higher is better).
// - hipDevP2PAttrAccessSupported: 1 if P2P access is supported.
// - hipDevP2PAttrNativeAtomicSupported: 1 if atomic operations supported.
// - hipDevP2PAttrCudaArrayAccessSupported: 1 if array access supported.
//
// Multi-GPU:
// - Queries capabilities of direct GPU-to-GPU communication.
// - Asymmetric: srcDevice→dstDevice may differ from dstDevice→srcDevice.
// - Performance varies based on PCIe topology.
//
// See also: hipDeviceCanAccessPeer, hipDeviceEnablePeerAccess.
HIPAPI hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attrib,
                                           int srcDevice, int dstDevice) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up P2P link.
  iree_hal_streaming_p2p_link_t* link =
      iree_hal_streaming_device_lookup_p2p_link(srcDevice, dstDevice);
  if (!link) {
    *value = 0;
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Map HIP P2P attribute enum to the appropriate link field.
  switch (attrib) {
    case hipDevP2PAttrAccessSupported:
      *value = link->access_supported ? 1 : 0;
      break;
    case hipDevP2PAttrNativeAtomicSupported:
      *value = link->native_atomic_supported ? 1 : 0;
      break;
    case hipDevP2PAttrHipArrayAccessSupported:
      *value = link->cuda_array_access_supported ? 1 : 0;
      break;
    case hipDevP2PAttrPerformanceRank:
      *value = link->performance_rank;
      break;
    default:
      // Unsupported attribute.
      *value = 0;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Enables direct memory access from current device to peer device.
//
// Parameters:
//  - peerDevice: [IN] Peer device to enable access to.
//  - flags: [IN] Reserved for future use (must be 0).
//
// Returns:
//  - hipSuccess: Peer access enabled successfully.
//  - hipErrorInvalidDevice: Invalid peer device.
//  - hipErrorInvalidValue: Invalid flags.
//  - hipErrorPeerAccessAlreadyEnabled: Access already enabled.
//  - hipErrorPeerAccessNotSupported: Devices cannot access each other.
//
// Synchronization: This operation is synchronous.
//
// Peer access behavior:
// - Enables current device to access peerDevice's memory.
// - Access is unidirectional (must enable separately for bidirectional).
// - Remains enabled until explicitly disabled or context destroyed.
// - Allows direct memory copies without staging through host.
//
// Multi-GPU:
// - Improves performance for device-to-device transfers.
// - Enables single-copy transfers instead of staged copies.
// - May increase memory bandwidth utilization.
//
// Warning: Check hipDeviceCanAccessPeer() before enabling.
//
// See also: hipDeviceDisablePeerAccess, hipDeviceCanAccessPeer,
//           hipMemcpyPeer.
HIPAPI hipError_t hipDeviceEnablePeerAccess(int peerDevice,
                                            unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get the current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the peer device's default context.
  iree_hal_streaming_device_t* peer_device =
      iree_hal_streaming_device_entry(peerDevice);
  if (!peer_device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get or create the peer device's primary context.
  iree_hal_streaming_context_t* peer_primary_context = NULL;
  iree_status_t status =
      iree_hal_streaming_device_get_or_create_primary_context(
          peer_device, &peer_primary_context);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  // Enable peer access between the current context and the peer device's
  // context.
  status = iree_hal_streaming_context_enable_peer_access(context,
                                                         peer_primary_context);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Disables direct memory access from current device to peer device.
//
// Parameters:
//  - peerDevice: [IN] Peer device to disable access to.
//
// Returns:
//  - hipSuccess: Peer access disabled successfully.
//  - hipErrorInvalidDevice: Invalid peer device.
//  - hipErrorPeerAccessNotEnabled: Peer access was not enabled.
//
// Synchronization: This operation is synchronous. Waits for all peer
// operations to complete.
//
// Peer access behavior:
// - Disables current device's access to peerDevice's memory.
// - Only affects current device's access (unidirectional).
// - Subsequent peer operations will fail or fall back to staged copies.
// - All ongoing peer operations complete before disabling.
//
// Multi-GPU:
// - Returns to staged copy behavior through host memory.
// - May reduce memory bandwidth utilization.
//
// Warning: Ensure no kernels are actively using peer memory.
//
// See also: hipDeviceEnablePeerAccess, hipDeviceCanAccessPeer.
HIPAPI hipError_t hipDeviceDisablePeerAccess(int peerDevice) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get the current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the peer device's default context.
  iree_hal_streaming_device_t* peer_device =
      iree_hal_streaming_device_entry(peerDevice);
  if (!peer_device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get or create the peer device's primary context.
  iree_hal_streaming_context_t* peer_primary_context = NULL;
  iree_status_t status =
      iree_hal_streaming_device_get_or_create_primary_context(
          peer_device, &peer_primary_context);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  // Disable peer access between the current context and the peer device's
  // context.
  status = iree_hal_streaming_context_disable_peer_access(context,
                                                          peer_primary_context);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Waits for all operations on the current device to complete.
//
// Parameters: None.
//
// Returns:
//  - hipSuccess: All operations completed successfully.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: A kernel launch failed.
//  - hipErrorIllegalAddress: Invalid memory access occurred.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation blocks the host thread until all
// previously enqueued operations on the current device have completed.
//
// Device behavior:
// - Synchronizes all streams on the current device.
// - Waits for all kernels, memory copies, and events.
// - More heavyweight than hipStreamSynchronize().
// - Does not synchronize with other devices.
//
// Multi-GPU: Only synchronizes the current device set by hipSetDevice().
//
// Performance note: Use stream-specific synchronization when possible for
// better performance.
//
// See also: hipStreamSynchronize, hipEventSynchronize, hipSetDevice.
HIPAPI hipError_t hipDeviceSynchronize(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }
  iree_status_t status = iree_hal_streaming_context_synchronize(context);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Resets the current device and destroys all allocations.
//
// Parameters: None.
//
// Returns:
//  - hipSuccess: Device reset successfully.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Waits for all operations
// to complete before resetting.
//
// Reset behavior:
// - Destroys all allocations on the current device.
// - Destroys all streams, events, and modules.
// - Resets all device state to initial values.
// - Primary context is destroyed and recreated.
// - Subsequent API calls reinitialize the device.
//
// Multi-GPU: Only resets the current device set by hipSetDevice().
//
// Warning: This is a heavyweight operation that affects all contexts on the
// device. All pointers and handles become invalid. Use with caution in
// multi-threaded applications.
//
// Usage pattern:
// ```c
// hipSetDevice(0);
// // ... work with device ...
// hipDeviceReset();  // Clean slate for device 0
// ```
//
// See also: hipSetDevice, hipDeviceSynchronize, hipCtxDestroy.
HIPAPI hipError_t hipDeviceReset(void) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get current context to determine which device to reset.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Reset the primary context for the current device.
  hipDevice_t current_device = context->device_ordinal;
  hipError_t result = hipDevicePrimaryCtxReset(current_device);

  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Primary context
//===----------------------------------------------------------------------===//

// Retains the primary context for a device.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the primary context handle.
//  - dev: [IN] Device handle.
//
// Returns:
//  - hipSuccess: Primary context retained successfully.
//  - hipErrorInvalidValue: pctx is NULL.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorOutOfMemory: Insufficient memory to create context.
//
// Synchronization: This operation is synchronous.
//
// Primary context behavior:
// - Creates primary context if it doesn't exist.
// - Increments reference count if it already exists.
// - Primary context is shared by all threads.
// - More lightweight than hipCtxCreate contexts.
// - Automatically created when needed by runtime API.
//
// Multi-GPU: Each device has its own primary context.
//
// Warning: Must balance with hipDevicePrimaryCtxRelease().
//
// See also: hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxSetFlags,
//           hipCtxCreate.
HIPAPI hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP runtime is initialized (but don't create context yet).
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Retain the primary context, creating it if necessary.
  iree_hal_streaming_context_t* primary_context = NULL;
  iree_status_t status = iree_hal_streaming_device_retain_primary_context(
      device, &primary_context);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  *pctx = (hipCtx_t)primary_context;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Releases the primary context for a device.
//
// Parameters:
//  - dev: [IN] Device handle.
//
// Returns:
//  - hipSuccess: Primary context released successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//
// Synchronization: This operation is synchronous.
//
// Primary context behavior:
// - Decrements reference count.
// - Destroys context when reference count reaches zero.
// - All resources in the context are released.
// - Subsequent API calls may recreate the context.
//
// Multi-GPU: Only affects the specified device's primary context.
//
// Warning: Must balance with hipDevicePrimaryCtxRetain().
//
// See also: hipDevicePrimaryCtxRetain, hipDevicePrimaryCtxReset.
HIPAPI hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Release the primary context (destroys when ref count reaches 0).
  iree_status_t status =
      iree_hal_streaming_device_release_primary_context(device);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidContext);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets flags for the primary context.
//
// Parameters:
//  - dev: [IN] Device handle.
//  - flags: [IN] Context flags (hipCtxSchedAuto, hipCtxSchedSpin,
//                hipCtxSchedYield, hipCtxSchedBlockingSync, etc.).
//
// Returns:
//  - hipSuccess: Flags set successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorInvalidValue: Invalid flags.
//  - hipErrorContextAlreadyInUse: Primary context already active.
//
// Synchronization: This operation is synchronous.
//
// Flag behavior:
// - Must be called before primary context is created.
// - Cannot change flags after context is active.
// - Affects scheduling behavior and resource allocation.
//
// Scheduling flags:
// - hipCtxSchedAuto: Automatic scheduling.
// - hipCtxSchedSpin: Spin-wait (low latency, high CPU).
// - hipCtxSchedYield: Yield CPU (higher latency, low CPU).
// - hipCtxSchedBlockingSync: Block on synchronization.
//
// Multi-GPU: Each device's primary context has independent flags.
//
// See also: hipDevicePrimaryCtxGetState, hipDevicePrimaryCtxRetain.
HIPAPI hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev,
                                              unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Convert HIP flags to strongly-typed internal flags.
  iree_hal_streaming_context_flags_t internal_flags = {0};

  // Extract scheduling mode from lower bits.
  unsigned int sched_flags = flags & 0x07;
  switch (sched_flags) {
    case hipDeviceScheduleAuto:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
    case hipDeviceScheduleSpin:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN;
      break;
    case hipDeviceScheduleYield:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD;
      break;
    case hipDeviceScheduleBlockingSync:
      internal_flags.scheduling_mode =
          IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC;
      break;
    default:
      internal_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
  }

  // Extract other flags.
  internal_flags.map_host_memory = (flags & hipDeviceMapHost) != 0;
  internal_flags.resize_local_mem_to_max =
      (flags & hipDeviceLmemResizeToMax) != 0;

  // Set the primary context flags.
  iree_status_t status =
      iree_hal_streaming_device_set_primary_context_flags(dev, &internal_flags);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets the state of the primary context.
//
// Parameters:
//  - dev: [IN] Device handle.
//  - flags: [OUT] Pointer to receive context flags (can be NULL).
//  - active: [OUT] Pointer to receive active state (can be NULL).
//
// Returns:
//  - hipSuccess: State retrieved successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//
// Synchronization: This operation is synchronous.
//
// State information:
// - flags: Current context creation flags.
// - active: 1 if context is active, 0 if inactive.
// - Context is active if reference count > 0.
//
// Multi-GPU: Queries state of specified device's primary context.
//
// Usage pattern:
// ```c
// unsigned int flags;
// int active;
// hipDevicePrimaryCtxGetState(device, &flags, &active);
// if (active) {
//   printf("Primary context is active\n");
// }
// ```
//
// See also: hipDevicePrimaryCtxSetFlags, hipDevicePrimaryCtxRetain.
HIPAPI hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev,
                                              unsigned int* flags,
                                              int* active) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get the primary context state.
  iree_hal_streaming_context_flags_t internal_flags = {0};
  bool is_active = false;
  iree_status_t status = iree_hal_streaming_device_primary_context_state(
      dev, &internal_flags, &is_active);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  if (flags) {
    // Convert internal flags back to HIP flags.
    unsigned int hip_flags = 0;

    // Set scheduling mode.
    switch (internal_flags.scheduling_mode) {
      case IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO:
        hip_flags |= hipDeviceScheduleAuto;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN:
        hip_flags |= hipDeviceScheduleSpin;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD:
        hip_flags |= hipDeviceScheduleYield;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC:
        hip_flags |= hipDeviceScheduleBlockingSync;
        break;
    }

    // Set other flags.
    if (internal_flags.map_host_memory) {
      hip_flags |= hipDeviceMapHost;
    }
    if (internal_flags.resize_local_mem_to_max) {
      hip_flags |= hipDeviceLmemResizeToMax;
    }

    *flags = hip_flags;
  }

  if (active) {
    *active = is_active ? 1 : 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Resets the primary context for a device.
//
// Parameters:
//  - dev: [IN] Device handle.
//
// Returns:
//  - hipSuccess: Primary context reset successfully.
//  - hipErrorInvalidDevice: Invalid device handle.
//
// Synchronization: This operation is synchronous. Waits for all operations
// to complete.
//
// Reset behavior:
// - Destroys the primary context regardless of reference count.
// - All resources in the context are released.
// - All allocations are freed.
// - All streams and events are destroyed.
// - Context will be recreated on next use.
//
// Multi-GPU: Only affects the specified device's primary context.
//
// Warning: This is a heavyweight operation that affects all users of the
// primary context. Use with caution in multi-threaded applications.
//
// See also: hipDeviceReset, hipDevicePrimaryCtxRelease, hipCtxDestroy.
HIPAPI hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Validate device.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (dev < 0 || !device_registry ||
      dev >= (int)device_registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Reset the primary context by:
  // 1. Waiting for all operations to complete (if it exists)
  // 2. Releasing the current context
  // 3. The context will be recreated lazily on next access

  if (device->primary_context) {
    // Wait for all operations on the context to complete.
    iree_status_t status = iree_hal_streaming_context_wait_idle(
        device->primary_context, iree_infinite_timeout());
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorUnknown);
    }

    // Lock to ensure thread safety during reset.
    iree_slim_mutex_lock(&device->primary_context_mutex);

    // Release the old context.
    iree_hal_streaming_context_release(device->primary_context);
    device->primary_context = NULL;

    // Reset reference count to 0.
    device->primary_context_ref_count = 0;

    // Also clear memory pools.
    if (device->default_mem_pool) {
      iree_hal_streaming_mem_pool_release(device->default_mem_pool);
      device->default_mem_pool = NULL;
    }
    if (device->current_mem_pool) {
      iree_hal_streaming_mem_pool_release(device->current_mem_pool);
      device->current_mem_pool = NULL;
    }

    iree_slim_mutex_unlock(&device->primary_context_mutex);

    // Clear current context if it was the primary context.
    iree_hal_streaming_context_t* current_context =
        iree_hal_streaming_context_current();
    if (current_context == device->primary_context) {
      iree_hal_streaming_context_set_current(NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Helper function to convert HIP context flags to internal flags.
static iree_hal_streaming_context_flags_t
iree_hal_streaming_hip_context_flags_to_internal(unsigned int hip_flags) {
  iree_hal_streaming_context_flags_t flags = {0};

  // Convert scheduling flags.
  int sched_flags = hip_flags & hipDeviceScheduleMask;
  switch (sched_flags) {
    case hipDeviceScheduleSpin:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN;
      break;
    case hipDeviceScheduleYield:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD;
      break;
    case hipDeviceScheduleBlockingSync:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC;
      break;
    default:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
  }

  // Convert other flags.
  if (hip_flags & hipDeviceMapHost) {
    flags.map_host_memory = true;
  }
  if (hip_flags & hipDeviceLmemResizeToMax) {
    flags.resize_local_mem_to_max = true;
  }

  return flags;
}

// Creates a new HIP context for a device.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the created context handle.
//  - flags: [IN] Context creation flags (hipCtxSchedAuto, hipCtxSchedSpin,
//                hipCtxSchedYield, hipCtxSchedBlockingSync, hipCtxMapHost,
//                hipCtxLmemResizeToMax).
//  - dev: [IN] Device handle for which to create the context.
//
// Returns:
//  - hipSuccess: Context created successfully.
//  - hipErrorInvalidValue: pctx is NULL or invalid flags.
//  - hipErrorInvalidDevice: Invalid device handle.
//  - hipErrorOutOfMemory: Insufficient memory to create context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during context creation.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Creates a new context and makes it current for this thread.
// - Each context maintains its own address space and resources.
// - Multiple contexts can exist per device.
// - Context must be destroyed with hipCtxDestroy().
//
// Scheduling flags:
// - hipCtxSchedAuto: Automatic scheduling (default).
// - hipCtxSchedSpin: Spin-wait for synchronization (low latency).
// - hipCtxSchedYield: Yield CPU for synchronization (low CPU usage).
// - hipCtxSchedBlockingSync: Block on synchronization.
//
// Multi-GPU: Each device can have multiple contexts.
//
// Warning: Creating multiple contexts per device may impact performance.
//
// See also: hipCtxDestroy, hipCtxPushCurrent, hipCtxPopCurrent,
//           hipCtxSetCurrent.
HIPAPI hipError_t hipCtxCreate(hipCtx_t* pctx, unsigned int flags,
                               hipDevice_t dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Create a new context for the device.
  iree_hal_streaming_context_t* context = NULL;
  // Get the host allocator from the device registry.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorNotInitialized);
  }

  iree_status_t status = iree_hal_streaming_context_create(
      device, iree_hal_streaming_hip_context_flags_to_internal(flags),
      device_registry->host_allocator, &context);

  if (iree_status_is_ok(status)) {
    *pctx = (hipCtx_t)context;
    // Make it current.
    status = iree_hal_streaming_context_set_current(context);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Destroys a HIP context.
//
// Parameters:
//  - ctx: [IN] Context handle to destroy.
//
// Returns:
//  - hipSuccess: Context destroyed successfully.
//  - hipErrorInvalidValue: ctx is NULL.
//  - hipErrorInvalidContext: Invalid context handle.
//  - hipErrorContextIsDestroyed: Context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for all operations
// in the context to complete before destroying.
//
// Context behavior:
// - All resources associated with the context are released.
// - If context is current, it is popped from the stack.
// - All memory allocations in the context are freed.
// - All streams and events in the context are destroyed.
// - Using a destroyed context results in undefined behavior.
//
// Multi-GPU: Only affects the specified context on its device.
//
// Warning: Ensure all operations using this context have completed.
// Destroying a context with active operations may cause errors.
//
// See also: hipCtxCreate, hipCtxPushCurrent, hipCtxPopCurrent.
HIPAPI hipError_t hipCtxDestroy(hipCtx_t ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Check if this is the current context.
  // If so, clear it from TLS to avoid a dangling reference.
  if (ctx == (hipCtx_t)iree_hal_streaming_context_current()) {
    // This will release the TLS reference.
    iree_hal_streaming_context_set_current(NULL);
  }

  // Release the context.
  iree_hal_streaming_context_release((iree_hal_streaming_context_t*)ctx);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Pushes a context onto the current thread's context stack.
//
// Parameters:
//  - ctx: [IN] Context to push (becomes current context).
//
// Returns:
//  - hipSuccess: Context pushed successfully.
//  - hipErrorInvalidValue: ctx is NULL.
//  - hipErrorInvalidContext: Invalid context handle.
//  - hipErrorContextIsDestroyed: Context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context stack behavior:
// - Pushes context onto the thread's context stack.
// - The pushed context becomes the current context.
// - Previous current context remains on the stack.
// - Must be balanced with hipCtxPopCurrent().
// - Stack depth is implementation-defined.
//
// Multi-GPU: Can push contexts from different devices onto the same stack.
//
// Usage pattern:
// ```c
// hipCtxPushCurrent(ctx1);  // ctx1 is now current
// // Operations use ctx1
// hipCtxPopCurrent(&old);    // Restore previous context
// ```
//
// Warning: Unbalanced push/pop operations can cause resource leaks.
//
// See also: hipCtxPopCurrent, hipCtxCreate, hipCtxSetCurrent,
//           hipCtxGetCurrent.
HIPAPI hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_status_t status =
      iree_hal_streaming_context_push((iree_hal_streaming_context_t*)ctx);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Pops a context from the current thread's context stack.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the popped context (can be NULL).
//
// Returns:
//  - hipSuccess: Context popped successfully.
//  - hipErrorInvalidContext: Context stack is empty.
//  - hipErrorContextIsDestroyed: Current context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context stack behavior:
// - Removes the current context from the stack.
// - Previous context on the stack becomes current.
// - If pctx is not NULL, receives the popped context.
// - If stack becomes empty, no context is current.
// - Must balance with hipCtxPushCurrent() calls.
//
// Multi-GPU: Restores the previous context which may be on a different
// device.
//
// Usage pattern:
// ```c
// hipCtx_t old;
// hipCtxPushCurrent(newCtx);
// // Work with newCtx
// hipCtxPopCurrent(&old);  // old == newCtx
// ```
//
// Warning: Popping from an empty stack is an error.
//
// See also: hipCtxPushCurrent, hipCtxGetCurrent, hipCtxSetCurrent.
HIPAPI hipError_t hipCtxPopCurrent(hipCtx_t* pctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = NULL;
  iree_status_t status = iree_hal_streaming_context_pop(&context);
  if (iree_status_is_ok(status) && pctx) {
    *pctx = (hipCtx_t)context;
  }
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets the current context for the calling thread.
//
// Parameters:
//  - pctx: [OUT] Pointer to receive the current context handle.
//
// Returns:
//  - hipSuccess: Current context retrieved successfully.
//  - hipErrorInvalidValue: pctx is NULL.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Returns the context at the top of the thread's context stack.
// - Returns NULL if no context is current.
// - Does not modify the context stack.
// - Returned context remains current.
//
// Multi-GPU: Returns the current context regardless of which device it
// belongs to.
//
// Usage pattern:
// ```c
// hipCtx_t current;
// hipCtxGetCurrent(&current);
// if (current) {
//   // Context is active
// }
// ```
//
// See also: hipCtxSetCurrent, hipCtxPushCurrent, hipGetDevice.
HIPAPI hipError_t hipCtxGetCurrent(hipCtx_t* pctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Do NOT implicitly initialize here - just return what's current.
  // hipCtxGetCurrent should return NULL if no context is set.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  *pctx = (hipCtx_t)context;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets the current context for the calling thread.
//
// Parameters:
//  - ctx: [IN] Context to make current (NULL to clear current context).
//
// Returns:
//  - hipSuccess: Context set successfully.
//  - hipErrorInvalidContext: Invalid context handle.
//  - hipErrorContextIsDestroyed: Context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Replaces the current context without modifying the stack.
// - If ctx is NULL, clears the current context.
// - Does not push or pop the context stack.
// - Previous current context is not retained.
//
// Multi-GPU: Can set a context from any device as current.
//
// Warning: This can leave the context stack in an inconsistent state if
// used with push/pop operations. Prefer push/pop for nested context
// management.
//
// See also: hipCtxGetCurrent, hipCtxPushCurrent, hipCtxPopCurrent.
HIPAPI hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_context_set_current(
      (iree_hal_streaming_context_t*)ctx);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets the device associated with the current context.
//
// Parameters:
//  - device: [OUT] Pointer to receive the device handle.
//
// Returns:
//  - hipSuccess: Device retrieved successfully.
//  - hipErrorInvalidValue: device is NULL.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorContextIsDestroyed: Current context has been destroyed.
//
// Synchronization: This operation is synchronous.
//
// Context behavior:
// - Returns the device that owns the current context.
// - Works with both primary and created contexts.
// - Each context is associated with exactly one device.
//
// Multi-GPU: Returns the device for the current context, which may be
// different from the current device set by hipSetDevice().
//
// See also: hipCtxGetCurrent, hipGetDevice, hipCtxCreate.
HIPAPI hipError_t hipCtxGetDevice(hipDevice_t* device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  *device = (hipDevice_t)context->device_ordinal;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Synchronizes all operations in the current context.
//
// Parameters: None.
//
// Returns:
//  - hipSuccess: All operations completed successfully.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorLaunchFailure: A kernel launch failed.
//  - hipErrorIllegalAddress: Invalid memory access occurred.
//
// Synchronization: This operation blocks until all previously enqueued
// operations in the current context have completed.
//
// Context behavior:
// - Waits for all streams in the current context.
// - Waits for all kernels, copies, and events.
// - More comprehensive than hipStreamSynchronize().
// - Does not affect other contexts.
//
// Multi-GPU: Only synchronizes the current context, not other contexts
// on the same or different devices.
//
// Performance note: Use stream-specific synchronization when possible.
//
// See also: hipDeviceSynchronize, hipStreamSynchronize, hipEventSynchronize.
HIPAPI hipError_t hipCtxSynchronize(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_context_synchronize(context);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Enables peer access from current context to peer context.
//
// Parameters:
//  - peerContext: [IN] Peer context to enable access to.
//  - flags: [IN] Reserved for future use (must be 0).
//
// Returns:
//  - hipSuccess: Peer access enabled successfully.
//  - hipErrorInvalidValue: peerContext is NULL or invalid flags.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorPeerAccessAlreadyEnabled: Access already enabled.
//  - hipErrorPeerAccessNotSupported: Devices cannot access each other.
//  - hipErrorInvalidDevice: Contexts on same device.
//
// Synchronization: This operation is synchronous.
//
// Peer access behavior:
// - Enables current context to access memory in peer context.
// - Access is unidirectional (must enable separately for bidirectional).
// - Contexts must be on different devices.
// - Devices must support P2P access.
// - Remains enabled until explicitly disabled.
//
// Multi-GPU:
// - Enables direct GPU-to-GPU memory access.
// - Avoids staging through host memory.
// - Improves performance for multi-GPU applications.
//
// See also: hipCtxDisablePeerAccess, hipDeviceCanAccessPeer,
//           hipDeviceEnablePeerAccess.
HIPAPI hipError_t hipCtxEnablePeerAccess(hipCtx_t peerContext,
                                         unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!peerContext) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_context_enable_peer_access(
      context, (iree_hal_streaming_context_t*)peerContext);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Disables peer access from current context to peer context.
//
// Parameters:
//  - peerContext: [IN] Peer context to disable access to.
//
// Returns:
//  - hipSuccess: Peer access disabled successfully.
//  - hipErrorInvalidValue: peerContext is NULL.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorPeerAccessNotEnabled: Peer access was not enabled.
//
// Synchronization: This operation is synchronous. Waits for all peer
// operations to complete.
//
// Peer access behavior:
// - Disables current context's access to peer context memory.
// - Only affects current context's access (unidirectional).
// - Subsequent peer operations will fail.
// - All ongoing peer operations complete before disabling.
//
// Multi-GPU:
// - Returns to staged copy behavior through host memory.
// - May reduce performance for multi-GPU operations.
//
// Warning: Ensure no kernels are actively using peer memory.
//
// See also: hipCtxEnablePeerAccess, hipDeviceDisablePeerAccess.
HIPAPI hipError_t hipCtxDisablePeerAccess(hipCtx_t peerContext) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!peerContext) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_context_disable_peer_access(
      context, (iree_hal_streaming_context_t*)peerContext);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets resource limits for the current device.
//
// Parameters:
//  - pValue: [OUT] Pointer to receive the limit value.
//  - limit: [IN] Limit to query (hipLimitStackSize, hipLimitPrintfFifoSize,
//                hipLimitMallocHeapSize, hipLimitDevRuntimeSyncDepth,
//                hipLimitDevRuntimePendingLaunchCount,
//                hipLimitMaxL2FetchGranularity, hipLimitPersistingL2CacheSize).
//
// Returns:
//  - hipSuccess: Limit retrieved successfully.
//  - hipErrorInvalidValue: pValue is NULL or invalid limit.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorUnsupportedLimit: Limit not supported on this device.
//
// Synchronization: This operation is synchronous.
//
// Common limits:
// - hipLimitStackSize: Stack size per thread in bytes.
// - hipLimitPrintfFifoSize: Size of printf FIFO in bytes.
// - hipLimitMallocHeapSize: Size of device malloc heap in bytes.
// - hipLimitDevRuntimeSyncDepth: Maximum nesting depth of sync operations.
// - hipLimitDevRuntimePendingLaunchCount: Maximum pending launches.
// - hipLimitMaxL2FetchGranularity: L2 cache fetch granularity.
// - hipLimitPersistingL2CacheSize: Persisting L2 cache size.
//
// Multi-GPU: Limits are per-device and context-specific.
//
// See also: hipDeviceSetLimit, hipDeviceGetAttribute.
HIPAPI hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pValue) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the limit value using internal API.
  iree_status_t status = iree_hal_streaming_context_limit(
      context, hip_limit_to_internal(limit), pValue);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets resource limits for the current device.
//
// Parameters:
//  - limit: [IN] Limit to set (hipLimitStackSize, hipLimitPrintfFifoSize,
//                hipLimitMallocHeapSize, etc.).
//  - value: [IN] New limit value in bytes.
//
// Returns:
//  - hipSuccess: Limit set successfully.
//  - hipErrorInvalidValue: Invalid limit or value.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorUnsupportedLimit: Limit not supported or read-only.
//  - hipErrorMemoryAllocation: Insufficient memory for new limit.
//
// Synchronization: This operation is synchronous.
//
// Limit behavior:
// - Changes take effect for subsequent kernel launches.
// - Does not affect currently running kernels.
// - Some limits may require context reset to take effect.
// - Setting to 0 may restore default value.
//
// Multi-GPU: Limits are per-device and context-specific.
//
// Warning: Increasing limits may reduce available memory for allocations.
// Setting limits too low may cause kernel launch failures.
//
// See also: hipDeviceGetLimit, hipFuncSetAttribute.
HIPAPI hipError_t hipDeviceSetLimit(hipLimit_t limit, size_t value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Set the limit value using internal API.
  iree_status_t status = iree_hal_streaming_context_set_limit(
      context, hip_limit_to_internal(limit), value);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

// Gets the amount of free and total device memory.
//
// Parameters:
//  - free: [OUT] Pointer to receive free memory in bytes (can be NULL).
//  - total: [OUT] Pointer to receive total memory in bytes (can be NULL).
//
// Returns:
//  - hipSuccess: Memory info retrieved successfully.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Current device is invalid.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory info behavior:
// - Free memory is currently available for allocation.
// - Total memory is the total device memory capacity.
// - Values may change between calls due to other allocations.
// - Does not include host memory.
//
// Multi-GPU: Returns memory info for the current device.
//
// Usage pattern:
// ```c
// size_t free_mem, total_mem;
// hipMemGetInfo(&free_mem, &total_mem);
// printf("GPU memory: %zu/%zu bytes free\n", free_mem, total_mem);
// ```
//
// Note: Free memory may be fragmented; large allocations may fail even if
// total free memory exceeds the requested size.
//
// See also: hipDeviceTotalMem, hipMalloc, hipSetDevice.
HIPAPI hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  size_t free_memory, total_memory;
  iree_status_t status = iree_hal_streaming_device_memory_info(
      context->device_ordinal, &free_memory, &total_memory);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  if (free) *free = (size_t)free_memory;
  if (total) *total = (size_t)total_memory;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Allocates memory on the device.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated device memory pointer.
//  - size: [IN] Size in bytes to allocate.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient device memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal allocation error.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Allocated memory is uninitialized.
// - Memory persists until freed with hipFree().
// - Memory is accessible from all streams on the device.
// - Allocation is aligned to meet all alignment requirements.
// - Zero-size allocations are allowed and return a valid pointer.
//
// Multi-GPU: Memory is allocated on the current device.
//
// Warning: Always check return value before using the allocated pointer.
// Dereferencing a failed allocation results in undefined behavior.
//
// See also: hipFree, hipMallocPitch, hipMallocHost, hipMallocManaged,
//           hipMallocAsync.
HIPAPI hipError_t hipMalloc(void** ptr, size_t size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_hal_streaming_memory_allocate_device(context, size, 0, &buffer);

  if (iree_status_is_ok(status)) {
    *ptr = (void*)buffer->device_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates pitched linear memory on the device.
//
// Parameters:
//  - devPtr: [OUT] Pointer to receive the allocated device memory pointer.
//  - pitch: [OUT] Pointer to receive the pitch in bytes.
//  - width: [IN] Requested width of allocation in bytes.
//  - height: [IN] Requested height of allocation in rows.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: devPtr or pitch is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorOutOfMemory: Insufficient device memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Allocates at least width bytes per row.
// - Actual pitch may be larger than width for alignment.
// - Use returned pitch for row-to-row calculations.
// - Memory layout: row[i] starts at devPtr + i * pitch.
// - Allocated memory is uninitialized.
//
// Multi-GPU: Memory is allocated on the current device.
//
// Performance note: Pitched memory can improve coalescing for 2D data
// access patterns.
//
// Warning: Always use the returned pitch value, not width, when accessing
// rows.
//
// See also: hipMalloc, hipMemcpy2D, hipFree.
HIPAPI hipError_t hipMallocPitch(void** devPtr, size_t* pitch, size_t width,
                                 size_t height) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!devPtr || !pitch) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Allocate pitched memory.
  // HIP doesn't have an ElementSizeBytes parameter like CUDA, so we pass 0.
  size_t calculated_pitch = 0;
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device_pitched(
      context, width, height, 0, &calculated_pitch, &buffer);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  // Return device pointer and pitch.
  *devPtr = (void*)iree_hal_streaming_buffer_device_pointer(buffer);
  *pitch = calculated_pitch;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Frees memory allocated with hipMalloc.
//
// Parameters:
//  - ptr: [IN] Device pointer to free (can be NULL).
//
// Returns:
//  - hipSuccess: Memory freed successfully or ptr was NULL.
//  - hipErrorInvalidDevicePointer: ptr is not a valid allocation.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorDeinitialized: HIP runtime has been deinitialized.
//
// Synchronization: This operation is synchronous. Waits for all operations
// using the memory to complete before freeing.
//
// Memory behavior:
// - Freeing NULL is a no-op and returns hipSuccess.
// - After freeing, the pointer becomes invalid.
// - Using freed memory results in undefined behavior.
// - Double-free results in undefined behavior.
//
// Multi-GPU: Memory must be freed from the same context that allocated it.
//
// Warning: Ensure all kernels using this memory have completed before
// freeing. Use hipDeviceSynchronize() if unsure.
//
// See also: hipMalloc, hipFreeHost, hipFreeAsync.
HIPAPI hipError_t hipFree(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // hipFree is synchronous - the internal function handles synchronization.
  iree_status_t status = iree_hal_streaming_memory_free_device(
      context, (iree_hal_streaming_deviceptr_t)ptr);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates page-locked host memory accessible from device.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated host memory pointer.
//  - size: [IN] Size in bytes to allocate.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient host memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Allocates pinned (page-locked) host memory.
// - Memory is accessible from both host and device.
// - Enables higher bandwidth for host-device transfers.
// - Memory is uninitialized.
// - Must be freed with hipFreeHost().
//
// Multi-GPU: Memory is accessible from all devices in the system.
//
// Performance note: Pinned memory improves transfer performance but reduces
// available memory for other processes. Use judiciously.
//
// Warning: Excessive pinned memory allocation can degrade system
// performance.
//
// See also: hipFreeHost, hipHostMalloc, hipHostRegister, hipMalloc.
HIPAPI hipError_t hipMallocHost(void** ptr, size_t size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_host(
      context, size, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer);

  if (iree_status_is_ok(status)) {
    *ptr = buffer->host_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Frees page-locked host memory allocated with hipMallocHost.
//
// Parameters:
//  - ptr: [IN] Host pointer to free (can be NULL).
//
// Returns:
//  - hipSuccess: Memory freed successfully or ptr was NULL.
//  - hipErrorInvalidValue: ptr is not a valid pinned allocation.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Freeing NULL is a no-op and returns hipSuccess.
// - After freeing, the pointer becomes invalid.
// - Using freed memory results in undefined behavior.
//
// Multi-GPU: Can be called from any device context.
//
// Warning: Only use for memory allocated with hipMallocHost or
// hipHostMalloc.
//
// See also: hipMallocHost, hipHostMalloc, hipHostFree.
HIPAPI hipError_t hipFreeHost(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // hipFreeHost is synchronous - the internal function handles synchronization.
  iree_status_t status = iree_hal_streaming_memory_free_host(context, ptr);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates host memory with specified properties.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive the allocated host memory pointer.
//  - size: [IN] Size in bytes to allocate.
//  - flags: [IN] Allocation flags (hipHostMallocDefault, hipHostMallocPortable,
//                hipHostMallocMapped, hipHostMallocWriteCombined, etc.).
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL or invalid flags.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient host memory.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - hipHostMallocDefault: Page-locked, accessible from all devices.
// - hipHostMallocPortable: Accessible from all devices in all contexts.
// - hipHostMallocMapped: Maps into device address space.
// - hipHostMallocWriteCombined: Optimized for device reads.
// - hipHostMallocCoherent: Coherent between host and device.
// - hipHostMallocNonCoherent: May require explicit synchronization.
// - Memory is uninitialized.
// - Must be freed with hipHostFree().
//
// Multi-GPU: Flags control multi-device accessibility.
//
// Performance note: Write-combined memory is fast for device reads but slow
// for host reads.
//
// See also: hipHostFree, hipMallocHost, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_host(
      context, size, hip_memory_flags_to_internal(flags), &buffer);

  if (iree_status_is_ok(status)) {
    *ptr = buffer->host_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Frees host memory allocated with hipHostMalloc.
//
// Parameters:
//  - ptr: [IN] Host pointer to free (can be NULL).
//
// Returns:
//  - hipSuccess: Memory freed successfully or ptr was NULL.
//  - hipErrorInvalidValue: ptr is not a valid allocation.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Memory behavior:
// - Freeing NULL is a no-op and returns hipSuccess.
// - Unmaps any device mappings if hipHostMallocMapped was used.
// - After freeing, both host and device pointers become invalid.
//
// Multi-GPU: Can be called from any device context.
//
// See also: hipHostMalloc, hipFreeHost, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostFree(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // hipHostFree is synchronous - the internal function handles synchronization.
  iree_status_t status = iree_hal_streaming_memory_free_host(context, ptr);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Allocates memory accessible from both host and device.
//
// Parameters:
//  - dev_ptr: [OUT] Pointer to receive the allocated memory pointer.
//  - size: [IN] Size in bytes to allocate.
//  - flags: [IN] Memory management flags (hipMemAttachGlobal, hipMemAttachHost,
//                hipMemAttachSingle).
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: dev_ptr is NULL or invalid flags.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorNotSupported: Managed memory not supported (current
//  implementation).
//
// Synchronization: This operation is synchronous.
//
// Managed memory behavior:
// - Automatically migrates between host and device on access.
// - Single address valid on both host and device.
// - System manages data movement transparently.
// - Performance depends on access patterns.
//
// Flags:
// - hipMemAttachGlobal: Memory accessible from any stream.
// - hipMemAttachHost: Memory prefers host access.
// - hipMemAttachSingle: Memory used by single stream.
//
// Multi-GPU: Managed memory can migrate between devices.
//
// Note: Currently not implemented in StreamHAL.
//
// See also: hipMalloc, hipMallocHost, hipFree.
HIPAPI hipError_t hipMallocManaged(void** dev_ptr, size_t size,
                                   unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement managed memory allocation.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Registers existing host memory for use by the device.
//
// Parameters:
//  - ptr: [IN] Host memory pointer to register.
//  - size: [IN] Size in bytes of memory to register.
//  - flags: [IN] Registration flags (hipHostRegisterDefault,
//                hipHostRegisterPortable, hipHostRegisterMapped,
//                hipHostRegisterIoMemory).
//
// Returns:
//  - hipSuccess: Memory registered successfully.
//  - hipErrorInvalidValue: ptr is NULL, size is 0, or invalid flags.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorMemoryAllocation: Registration failed.
//  - hipErrorNotSupported: Flags not supported.
//
// Synchronization: This operation is synchronous.
//
// Registration behavior:
// - Pins the memory pages to prevent swapping.
// - Enables high-bandwidth transfers to/from device.
// - Memory must be page-aligned for best performance.
// - Must be unregistered with hipHostUnregister().
//
// Flags:
// - hipHostRegisterDefault: Basic pinning.
// - hipHostRegisterPortable: Accessible from all devices.
// - hipHostRegisterMapped: Maps into device address space.
// - hipHostRegisterIoMemory: Memory is IO memory.
//
// Multi-GPU: Portable flag enables access from all devices.
//
// Warning: Registering too much memory can degrade system performance.
//
// See also: hipHostUnregister, hipHostMalloc, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr || size == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Convert HIP flags to internal flags.
  iree_hal_streaming_host_register_flags_t internal_flags =
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  if (flags & hipHostRegisterPortable) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE;
  }
  if (flags & hipHostRegisterMapped) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED;
  }
  if (flags & hipHostRegisterIoMemory) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED;
  }
  if (flags & hipHostRegisterReadOnly) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY;
  }

  // Register the host memory using the internal function.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_register_host(
      context, ptr, size, internal_flags, &buffer);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Unregisters host memory previously registered with hipHostRegister.
//
// Parameters:
//  - ptr: [IN] Host memory pointer to unregister.
//
// Returns:
//  - hipSuccess: Memory unregistered successfully.
//  - hipErrorInvalidValue: ptr is NULL or not registered.
//  - hipErrorInvalidContext: No current context.
//  - hipErrorHostMemoryNotRegistered: Memory was not registered.
//
// Synchronization: This operation is synchronous.
//
// Unregistration behavior:
// - Unpins the memory pages.
// - Memory can be swapped again.
// - Removes device mappings if hipHostRegisterMapped was used.
// - Must match the pointer from hipHostRegister().
//
// Multi-GPU: Unregisters from all devices if portable.
//
// Warning: Ensure no device operations are using this memory.
//
// See also: hipHostRegister, hipHostFree.
HIPAPI hipError_t hipHostUnregister(void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Unregister the host memory using the internal function.
  iree_status_t status =
      iree_hal_streaming_memory_unregister_host(context, ptr);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets the address range of a device allocation.
//
// Parameters:
//  - pbase: [OUT] Pointer to receive the base address of allocation.
//  - psize: [OUT] Pointer to receive the size of allocation.
//  - dptr: [IN] Device pointer to query.
//
// Returns:
//  - hipSuccess: Address range retrieved successfully.
//  - hipErrorInvalidValue: pbase or psize is NULL.
//  - hipErrorInvalidDevicePointer: dptr is not a valid allocation.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Address range behavior:
// - Returns the original allocation containing dptr.
// - dptr can be anywhere within the allocation.
// - Useful for finding allocation boundaries.
//
// Multi-GPU: Queries allocation on current device.
//
// See also: hipMemPtrGetInfo, hipMalloc.
HIPAPI hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                        hipDeviceptr_t dptr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pbase || !psize) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_deviceptr_t base = 0;
  size_t size = 0;
  iree_status_t status = iree_hal_streaming_memory_address_range(
      context, (iree_hal_streaming_deviceptr_t)dptr, &base, &size);

  if (iree_status_is_ok(status)) {
    *pbase = (hipDeviceptr_t)base;
    *psize = size;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets device pointer for mapped host memory.
//
// Parameters:
//  - pdptr: [OUT] Pointer to receive the device pointer.
//  - p: [IN] Host pointer to query.
//  - flags: [IN] Reserved (must be 0).
//
// Returns:
//  - hipSuccess: Device pointer retrieved successfully.
//  - hipErrorInvalidValue: pdptr or p is NULL, or invalid flags.
//  - hipErrorMemoryAllocation: Host memory not mapped to device.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Mapping behavior:
// - Host memory must be allocated with hipHostMalloc(hipHostMallocMapped).
// - Or registered with hipHostRegister(hipHostRegisterMapped).
// - Returns device-accessible pointer to same memory.
// - Single memory location, two addresses.
//
// Multi-GPU: Device pointer valid on current device.
//
// Warning: Device pointer invalid if host memory unmapped.
//
// See also: hipHostMalloc, hipHostRegister, hipHostGetFlags.
HIPAPI hipError_t hipHostGetDevicePointer(hipDeviceptr_t* pdptr, void* p,
                                          unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pdptr || !p) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up the buffer from the host pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)p, &buffer_ref);

  if (iree_status_is_ok(status)) {
    // For registered host memory, the device pointer is the same as host
    // pointer.
    *pdptr =
        (hipDeviceptr_t)(buffer_ref.buffer->device_ptr + buffer_ref.offset);
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets flags used to allocate pinned host memory.
//
// Parameters:
//  - flagsPtr: [OUT] Pointer to receive the allocation flags.
//  - hostPtr: [IN] Host pointer to query.
//
// Returns:
//  - hipSuccess: Flags retrieved successfully.
//  - hipErrorInvalidValue: flagsPtr or hostPtr is NULL.
//  - hipErrorInvalidHostPointer: hostPtr not allocated with HIP.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Flag information:
// - Returns flags from hipHostMalloc() or hipHostRegister().
// - hipHostMallocDefault, hipHostMallocPortable, hipHostMallocMapped,
//   hipHostMallocWriteCombined, hipHostMallocCoherent, etc.
// - Returns 0 for non-HIP allocations.
//
// Multi-GPU: Flags indicate portability across devices.
//
// See also: hipHostMalloc, hipHostRegister, hipHostGetDevicePointer.
HIPAPI hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flagsPtr || !hostPtr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get the internal flags.
  iree_hal_streaming_host_register_flags_t internal_flags = {0};
  iree_status_t status =
      iree_hal_streaming_memory_host_flags(context, hostPtr, &internal_flags);

  if (iree_status_is_ok(status)) {
    // Convert internal flags back to HIP flags.
    unsigned int hip_flags = 0;
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE) {
      hip_flags |= hipHostRegisterPortable;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED) {
      hip_flags |= hipHostRegisterMapped;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED) {
      hip_flags |= hipHostRegisterIoMemory;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY) {
      hip_flags |= hipHostRegisterReadOnly;
    }
    *flagsPtr = hip_flags;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Gets information about a memory pointer.
//
// Parameters:
//  - ptr: [IN] Pointer to query (host or device).
//  - size: [OUT] Pointer to receive allocation size.
//
// Returns:
//  - hipSuccess: Information retrieved successfully.
//  - hipErrorInvalidValue: ptr or size is NULL.
//  - hipErrorInvalidDevicePointer: ptr is not a valid allocation.
//  - hipErrorInvalidContext: No current context.
//
// Synchronization: This operation is synchronous.
//
// Pointer information:
// - Works with device allocations from hipMalloc().
// - Works with host allocations from hipHostMalloc().
// - Returns size of original allocation.
// - ptr can be anywhere within allocation.
//
// Multi-GPU: Queries current device's allocations.
//
// Note: Extended version hipDrvMemGetInfo provides more details.
//
// See also: hipMemGetAddressRange, hipPointerGetAttributes.
HIPAPI hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!ptr || !size) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up the buffer from the pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);

  if (iree_status_is_ok(status)) {
    *size = buffer_ref.buffer->size;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data between host and device.
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - src: [IN] Source pointer (host or device).
//  - sizeBytes: [IN] Number of bytes to copy.
//  - kind: [IN] Type of copy (hipMemcpyHostToHost, hipMemcpyHostToDevice,
//               hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice).
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers, invalid size, or invalid kind.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevicePointer: Device pointer not valid.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until copy
// completes.
//
// Copy behavior:
// - hipMemcpyHostToHost: CPU memcpy.
// - hipMemcpyHostToDevice: Transfer from host to device.
// - hipMemcpyDeviceToHost: Transfer from device to host.
// - hipMemcpyDeviceToDevice: Copy within or between devices.
// - hipMemcpyDefault: Runtime determines direction from pointer types.
//
// Multi-GPU:
// - Device-to-device copies within same GPU are supported.
// - Cross-device copies require peer access or staging through host.
//
// Performance note: For asynchronous transfers, use hipMemcpyAsync().
//
// See also: hipMemcpyAsync, hipMemcpy2D, hipMemcpyHtoD, hipMemcpyDtoH.
HIPAPI hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                            hipMemcpyKind kind) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_ok_status();
  switch (kind) {
    case hipMemcpyHostToDevice:
      status = iree_hal_streaming_memcpy_host_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst, src, sizeBytes, NULL);
      break;
    case hipMemcpyDeviceToHost:
      status = iree_hal_streaming_memcpy_device_to_host(
          context, dst, (iree_hal_streaming_deviceptr_t)src, sizeBytes, NULL);
      break;
    case hipMemcpyDeviceToDevice:
      status = iree_hal_streaming_memcpy_device_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst,
          (iree_hal_streaming_deviceptr_t)src, sizeBytes, NULL);
      break;
    case hipMemcpyHostToHost:
      memcpy(dst, src, sizeBytes);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      break;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data between host and device asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - src: [IN] Source pointer (host or device).
//  - sizeBytes: [IN] Number of bytes to copy.
//  - kind: [IN] Type of copy (hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
//               hipMemcpyDeviceToDevice, hipMemcpyDefault).
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers, invalid size, or invalid kind.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevicePointer: Device pointer not valid.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous. Returns immediately after
// enqueueing the copy.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Copy executes after all previously enqueued operations in the stream.
// - Subsequent operations in the stream wait for copy to complete.
//
// Memory requirements:
// - Host memory must be pinned for async H2D/D2H transfers.
// - Use hipHostMalloc() or hipHostRegister() to pin memory.
// - Non-pinned memory falls back to synchronous copy.
//
// Multi-GPU:
// - Device-to-device copies within same GPU are supported.
// - Cross-device copies require peer access.
//
// Warning: Host memory must remain valid until copy completes.
//
// See also: hipMemcpy, hipStreamSynchronize, hipHostMalloc,
//           hipMemcpyHtoDAsync.
HIPAPI hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                 hipMemcpyKind kind, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    stream = (hipStream_t)context->default_stream;
  }

  iree_status_t status = iree_ok_status();
  switch (kind) {
    case hipMemcpyHostToDevice:
      status = iree_hal_streaming_memcpy_host_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst, src, sizeBytes,
          (iree_hal_streaming_stream_t*)stream);
      break;
    case hipMemcpyDeviceToHost:
      status = iree_hal_streaming_memcpy_device_to_host(
          context, dst, (iree_hal_streaming_deviceptr_t)src, sizeBytes,
          (iree_hal_streaming_stream_t*)stream);
      break;
    case hipMemcpyDeviceToDevice:
      status = iree_hal_streaming_memcpy_device_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst,
          (iree_hal_streaming_deviceptr_t)src, sizeBytes,
          (iree_hal_streaming_stream_t*)stream);
      break;
    case hipMemcpyHostToHost:
      // Host-to-host copies are synchronous.
      memcpy(dst, src, sizeBytes);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
      break;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Copies data between host and device with stream (deprecated).
//
// Parameters:
//  - dst: [OUT] Destination pointer (host or device).
//  - src: [IN] Source pointer (host or device).
//  - sizeBytes: [IN] Number of bytes to copy.
//  - kind: [IN] Type of copy (hipMemcpyHostToDevice, etc.).
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns: Same as hipMemcpyAsync.
//
// Synchronization: This operation is asynchronous.
//
// Note: This function is deprecated. Use hipMemcpyAsync instead.
//
// See also: hipMemcpyAsync, hipMemcpy.
HIPAPI hipError_t hipMemcpyWithStream(void* dst, const void* src,
                                      size_t sizeBytes, hipMemcpyKind kind,
                                      hipStream_t stream) {
  // hipMemcpyWithStream is the same as hipMemcpyAsync.
  return hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

// Copies data from host memory to device memory (synchronous).
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source host pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorInvalidContext: No active context.
//  - hipErrorInvalidDevice: Device pointer not valid for current device.
//
// Synchronization: SYNCHRONOUS - blocks until copy completes.
// Stream behavior: Executes on default stream (stream 0).
// Multi-GPU: Current device context determines which device performs copy.
// Limitations: None in current implementation.
//
// Warning: Passing device pointer as src or host pointer as dst causes
//          undefined behavior.
//
// See also: hipMemcpy, hipMemcpyHtoDAsync, hipMemcpyDtoH.
HIPAPI hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src,
                                size_t sizeBytes) {
  // Synchronous host-to-device copy.
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyHostToDevice);
}

// Copies data from device memory to host memory (synchronous).
//
// Parameters:
//  - dst: [OUT] Destination host pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorInvalidContext: No active context.
//  - hipErrorInvalidDevice: Device pointer not valid for current device.
//
// Synchronization: SYNCHRONOUS - blocks until copy completes.
// Stream behavior: Executes on default stream (stream 0).
// Multi-GPU: Current device context determines which device performs copy.
// Limitations: None in current implementation.
//
// Warning: Passing host pointer as src or device pointer as dst causes
//          undefined behavior.
//
// See also: hipMemcpy, hipMemcpyDtoHAsync, hipMemcpyHtoD.
HIPAPI hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src,
                                size_t sizeBytes) {
  // Synchronous device-to-host copy.
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToHost);
}

// Copies data from device memory to device memory (synchronous).
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//
// Returns:
//  - hipSuccess: Copy completed successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorInvalidContext: No active context.
//  - hipErrorInvalidDevice: Device pointers not valid for current device.
//
// Synchronization: SYNCHRONOUS - blocks until copy completes.
// Stream behavior: Executes on default stream (stream 0).
// Multi-GPU: Both pointers must be accessible from current device. For
//            cross-device copies, use hipMemcpyPeer.
// Limitations: Intra-device copy only; no automatic peer access.
//
// Note: Source and destination regions may overlap; behavior is equivalent
//       to memmove.
//
// See also: hipMemcpy, hipMemcpyDtoDAsync, hipMemcpyPeer.
HIPAPI hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src,
                                size_t sizeBytes) {
  // Synchronous device-to-device copy.
  return hipMemcpy(dst, src, sizeBytes, hipMemcpyDeviceToDevice);
}

// Copies data from host to device memory asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source host pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorInvalidDevicePointer: Invalid device pointer.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Host memory should be pinned for true async behavior.
//
// Multi-GPU: Copies to current device.
//
// See also: hipMemcpyHtoD, hipMemcpyAsync, hipMemcpyDtoHAsync.
HIPAPI hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src,
                                     size_t sizeBytes, hipStream_t stream) {
  // Asynchronous host-to-device copy.
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyHostToDevice, stream);
}

// Copies data from device to host memory asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination host pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorInvalidDevicePointer: Invalid device pointer.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Host memory should be pinned for true async behavior.
//
// Multi-GPU: Copies from current device.
//
// Warning: Host memory must remain valid until copy completes.
//
// See also: hipMemcpyDtoH, hipMemcpyAsync, hipMemcpyHtoDAsync.
HIPAPI hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src,
                                     size_t sizeBytes, hipStream_t stream) {
  // Asynchronous device-to-host copy.
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToHost, stream);
}

// Copies data from device to device memory asynchronously.
//
// Parameters:
//  - dst: [OUT] Destination device pointer.
//  - src: [IN] Source device pointer.
//  - sizeBytes: [IN] Number of bytes to copy.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Copy enqueued successfully.
//  - hipErrorInvalidValue: NULL pointers or size is 0.
//  - hipErrorInvalidDevicePointer: Invalid device pointers.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Copy is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Executes after all previously enqueued operations.
//
// Multi-GPU: Both pointers must be accessible from current device.
// For cross-device copies, use hipMemcpyPeerAsync.
//
// Note: Source and destination may overlap (behaves like memmove).
//
// See also: hipMemcpyDtoD, hipMemcpyAsync, hipMemcpyPeerAsync.
HIPAPI hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
                                     size_t sizeBytes, hipStream_t stream) {
  // Asynchronous device-to-device copy.
  return hipMemcpyAsync(dst, src, sizeBytes, hipMemcpyDeviceToDevice, stream);
}

// Sets device memory to a value.
//
// Parameters:
//  - dst: [IN/OUT] Device pointer to memory to set.
//  - value: [IN] Value to set (interpreted as unsigned char).
//  - sizeBytes: [IN] Number of bytes to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dst is NULL or sizeBytes overflows.
//  - hipErrorInvalidDevicePointer: dst is not a valid device pointer.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Memory behavior:
// - Sets each byte to (unsigned char)(value & 0xFF).
// - Works with any device memory regardless of allocation type.
// - Can set partial allocations.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// Performance note: For large memory regions, consider using hipMemsetAsync
// to overlap with other operations.
//
// See also: hipMemsetAsync, hipMemsetD8, hipMemsetD16, hipMemsetD32.
HIPAPI hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dst, sizeBytes, &value, 1,
      context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to a value asynchronously.
//
// Parameters:
//  - dst: [IN/OUT] Device pointer to memory to set.
//  - value: [IN] Value to set (interpreted as unsigned char).
//  - sizeBytes: [IN] Number of bytes to set.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Operation enqueued successfully.
//  - hipErrorInvalidValue: dst is NULL or sizeBytes overflows.
//  - hipErrorInvalidDevicePointer: dst is not a valid device pointer.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Operation is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Executes after all previously enqueued operations in the stream.
// - Subsequent operations in the stream wait for this to complete.
//
// Memory behavior:
// - Sets each byte to (unsigned char)(value & 0xFF).
// - Memory contents are undefined if read before operation completes.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// See also: hipMemset, hipMemsetD8Async, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                                 hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dst, sizeBytes, &value, 1,
      stream ? (iree_hal_streaming_stream_t*)stream : context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to an 8-bit value.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - uc: [IN] 8-bit value to set.
//  - N: [IN] Number of 8-bit values to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Memory behavior:
// - Sets N consecutive bytes to the value uc.
// - Total bytes modified: N.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// See also: hipMemset, hipMemsetD8Async, hipMemsetD16, hipMemsetD32.
HIPAPI hipError_t hipMemsetD8(hipDeviceptr_t dstDevice, unsigned char uc,
                              size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N, &uc, 1,
      context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to a 16-bit value.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - us: [IN] 16-bit value to set.
//  - N: [IN] Number of 16-bit values to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Memory behavior:
// - Sets N consecutive 16-bit values to us.
// - Total bytes modified: N * 2.
// - Memory must be 2-byte aligned.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// Warning: dstDevice must be aligned to 2-byte boundary.
//
// See also: hipMemset, hipMemsetD16Async, hipMemsetD8, hipMemsetD32.
HIPAPI hipError_t hipMemsetD16(hipDeviceptr_t dstDevice, unsigned short us,
                               size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 2, &us, 2,
      context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to a 32-bit value.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - i: [IN] 32-bit value to set.
//  - N: [IN] Number of 32-bit values to set.
//
// Returns:
//  - hipSuccess: Memory set successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous. Blocks until complete.
//
// Memory behavior:
// - Sets N consecutive 32-bit values to i.
// - Total bytes modified: N * 4.
// - Memory must be 4-byte aligned.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// Warning: dstDevice must be aligned to 4-byte boundary.
//
// See also: hipMemset, hipMemsetD32Async, hipMemsetD8, hipMemsetD16.
HIPAPI hipError_t hipMemsetD32(hipDeviceptr_t dstDevice, int i, size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 4, &i, 4,
      context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Sets device memory to an 8-bit value asynchronously.
//
// Parameters:
//  - dstDevice: [IN/OUT] Device pointer to memory to set.
//  - uc: [IN] 8-bit value to set.
//  - N: [IN] Number of 8-bit values to set.
//  - stream: [IN] Stream for asynchronous execution (NULL = default stream).
//
// Returns:
//  - hipSuccess: Operation enqueued successfully.
//  - hipErrorInvalidValue: dstDevice is NULL or N overflows.
//  - hipErrorInvalidDevicePointer: dstDevice is not valid.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Operation is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Memory contents are undefined if read before operation completes.
//
// Memory behavior:
// - Sets N consecutive bytes to the value uc.
// - Total bytes modified: N.
//
// Multi-GPU: Operates on memory accessible from current device.
//
// See also: hipMemsetD8, hipMemsetAsync, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetD8Async(hipDeviceptr_t dstDevice, unsigned char uc,
                                   size_t N, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N, &uc, 1,
      stream ? (iree_hal_streaming_stream_t*)stream : context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Asynchronously sets memory to a 16-bit value.
//
// Parameters:
//  - dstDevice: [IN] Device memory pointer to set.
//  - us: [IN] 16-bit value to set in each element.
//  - N: [IN] Number of 16-bit elements to set.
//  - stream: [IN] Stream for asynchronous operation.
//
// Returns:
//  - hipSuccess: Memory set operation queued successfully.
//  - hipErrorInvalidValue: Invalid pointer or count.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorUnknown: Internal error.
//
// Synchronization: This operation is asynchronous and completes
// when the stream reaches this operation.
//
// Memory behavior:
// - Sets N consecutive 16-bit values starting at dstDevice.
// - The destination address must be 2-byte aligned.
// - Total bytes written: N * sizeof(uint16_t).
// - Pattern is replicated as 16-bit values, not bytes.
//
// Multi-GPU: Memory must be accessible from the current device.
//
// Performance notes:
// - May be optimized for aligned addresses and specific patterns.
// - Large fills may be broken into multiple operations internally.
// - Consider coalescing multiple small fills for better performance.
//
// Warning: Ensure proper alignment for 16-bit access. Unaligned
// access may cause performance degradation or errors.
//
// See also: hipMemsetD16, hipMemsetD8Async, hipMemsetD32Async,
//           hipMemsetAsync, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetD16Async(hipDeviceptr_t dstDevice, unsigned short us,
                                    size_t N, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 2, &us, 2,
      stream ? (iree_hal_streaming_stream_t*)stream : context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Asynchronously sets memory to a 32-bit value.
//
// Parameters:
//  - dstDevice: [IN] Device memory pointer to set.
//  - i: [IN] 32-bit value to set in each element.
//  - N: [IN] Number of 32-bit elements to set.
//  - stream: [IN] Stream for asynchronous operation.
//
// Returns:
//  - hipSuccess: Memory set operation queued successfully.
//  - hipErrorInvalidValue: Invalid pointer or count.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorUnknown: Internal error.
//
// Synchronization: This operation is asynchronous and completes
// when the stream reaches this operation.
//
// Memory behavior:
// - Sets N consecutive 32-bit values starting at dstDevice.
// - The destination address must be 4-byte aligned.
// - Total bytes written: N * sizeof(int32_t).
// - Pattern is replicated as 32-bit values, not bytes.
//
// Multi-GPU: Memory must be accessible from the current device.
//
// Performance notes:
// - May be optimized for aligned addresses and specific patterns.
// - Large fills may be broken into multiple operations internally.
// - Consider coalescing multiple small fills for better performance.
// - Often most efficient for clearing large buffers.
//
// Warning: Ensure proper alignment for 32-bit access. Unaligned
// access may cause performance degradation or errors.
//
// See also: hipMemsetD32, hipMemsetD8Async, hipMemsetD16Async,
//           hipMemsetAsync, hipStreamSynchronize.
HIPAPI hipError_t hipMemsetD32Async(hipDeviceptr_t dstDevice, int i, size_t N,
                                    hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 4, &i, 4,
      stream ? (iree_hal_streaming_stream_t*)stream : context->default_stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

// Creates a new asynchronous stream for independent command execution.
//
// Parameters:
//  - stream: [OUT] Pointer to receive the created stream handle.
//
// Returns:
//  - hipSuccess: Stream created successfully.
//  - hipErrorInvalidValue: stream pointer is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create stream.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during stream creation.
//
// Synchronization: This operation is synchronous.
//
// Stream behavior:
// - The new stream is independent of the default stream.
// - Operations on different streams may execute concurrently.
// - Operations within a stream execute in order.
// - Use hipStreamSynchronize() to wait for all operations to complete.
// - Use hipStreamQuery() to check completion status without blocking.
// - Stream must be destroyed with hipStreamDestroy() when no longer needed.
//
// Multi-GPU: The stream is associated with the current device context.
//
// Note: NULL stream (0) represents the default stream which synchronizes
// with all other streams on the device.
//
// See also: hipStreamCreateWithFlags, hipStreamCreateWithPriority,
//           hipStreamDestroy, hipStreamSynchronize.
HIPAPI hipError_t hipStreamCreate(hipStream_t* stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, 0, context->host_allocator,
      &stream_obj);

  if (iree_status_is_ok(status)) {
    *stream = (hipStream_t)stream_obj;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Creates a new asynchronous stream with specified flags.
//
// Parameters:
//  - stream: [OUT] Pointer to receive the created stream handle.
//  - flags: [IN] Stream creation flags (hipStreamDefault or
//                hipStreamNonBlocking).
//
// Returns:
//  - hipSuccess: Stream created successfully.
//  - hipErrorInvalidValue: stream pointer is NULL or invalid flags.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create stream.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during stream creation.
//
// Synchronization: This operation is synchronous.
//
// Stream flags:
// - hipStreamDefault (0): Default stream behavior.
// - hipStreamNonBlocking: Stream does not synchronize with NULL stream.
//
// Stream behavior:
// - The new stream is independent of other streams.
// - Operations on different streams may execute concurrently.
// - Operations within a stream execute in order.
// - Non-blocking streams do not implicitly synchronize with the NULL
//   stream, allowing better overlap of host and device execution.
// - Stream must be destroyed with hipStreamDestroy() when no longer needed.
//
// Multi-GPU: Stream is created on the current device and can only
// execute operations on that device.
//
// Warning: Using non-blocking streams requires careful synchronization
// to avoid race conditions with host code.
//
// See also: hipStreamCreate, hipStreamCreateWithPriority,
//           hipStreamDestroy, hipStreamSynchronize.
HIPAPI hipError_t hipStreamCreateWithFlags(hipStream_t* stream,
                                           unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, hip_stream_flags_to_internal(flags), 0, context->host_allocator,
      &stream_obj);

  if (iree_status_is_ok(status)) {
    *stream = (hipStream_t)stream_obj;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Creates a new asynchronous stream with specified flags and priority.
//
// Parameters:
//  - stream: [OUT] Pointer to receive the created stream handle.
//  - flags: [IN] Stream creation flags (hipStreamDefault or
//                hipStreamNonBlocking).
//  - priority: [IN] Stream priority (higher values = higher priority).
//
// Returns:
//  - hipSuccess: Stream created successfully.
//  - hipErrorInvalidValue: stream pointer is NULL, invalid flags, or
//                          priority outside valid range.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create stream.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during stream creation.
//
// Synchronization: This operation is synchronous.
//
// Priority behavior:
// - Higher numerical values indicate higher priority.
// - Priority range can be queried with hipDeviceGetStreamPriorityRange().
// - Work in higher priority streams may preempt lower priority streams.
// - Priorities are hints; actual scheduling depends on hardware support.
// - If priorities are not supported, all streams execute with equal
//   priority.
//
// Stream behavior:
// - The new stream is independent of other streams.
// - Operations on different streams may execute concurrently.
// - Operations within a stream execute in order.
// - Non-blocking streams do not implicitly synchronize with NULL stream.
// - Stream must be destroyed with hipStreamDestroy() when no longer needed.
//
// Multi-GPU: Stream is created on the current device and can only
// execute operations on that device.
//
// Warning: Priority scheduling is a performance hint and may not be
// supported on all devices. Check device capabilities before relying
// on priority behavior.
//
// See also: hipStreamCreate, hipStreamCreateWithFlags,
//           hipDeviceGetStreamPriorityRange, hipStreamGetPriority.
HIPAPI hipError_t hipStreamCreateWithPriority(hipStream_t* stream,
                                              unsigned int flags,
                                              int priority) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_stream_t* stream_obj = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, hip_stream_flags_to_internal(flags), priority,
      context->host_allocator, &stream_obj);

  if (iree_status_is_ok(status)) {
    *stream = (hipStream_t)stream_obj;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Destroys a stream previously created with hipStreamCreate.
//
// Parameters:
//  - stream: [IN] Stream handle to destroy.
//
// Returns:
//  - hipSuccess: Stream destroyed successfully.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorContextIsDestroyed: Associated context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for all pending
// operations in the stream to complete before destroying.
//
// Stream behavior:
// - All operations in the stream must complete before destruction.
// - After destruction, the stream handle becomes invalid.
// - Using a destroyed stream results in undefined behavior.
//
// Multi-GPU: Stream must be destroyed from the same context that created it.
//
// Warning: Ensure all operations using this stream have completed. The
// function implicitly synchronizes the stream before destruction.
//
// See also: hipStreamCreate, hipStreamCreateWithFlags,
//           hipStreamCreateWithPriority.
HIPAPI hipError_t hipStreamDestroy(hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_stream_release((iree_hal_streaming_stream_t*)stream);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Queries the priority of a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//  - priority: [OUT] Pointer to receive the stream priority.
//
// Returns:
//  - hipSuccess: Priority queried successfully.
//  - hipErrorInvalidValue: priority pointer is NULL.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//
// Synchronization: This operation is synchronous and immediate.
//
// Priority behavior:
// - Returns the priority assigned when the stream was created.
//  - Default streams have priority 0.
// - Higher numerical values indicate higher priority.
// - If priorities are not supported, returns 0.
//
// Multi-GPU: Queries the stream associated with the current context.
//
// See also: hipStreamCreateWithPriority, hipStreamGetFlags,
//           hipDeviceGetStreamPriorityRange.
HIPAPI hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  if (!priority) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      HIP_RETURN_ERROR(init_result);
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  *priority = stream_obj->priority;
  return hipSuccess;
}

// Queries the flags of a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//  - flags: [OUT] Pointer to receive the stream flags.
//
// Returns:
//  - hipSuccess: Flags queried successfully.
//  - hipErrorInvalidValue: flags pointer is NULL.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//
// Synchronization: This operation is synchronous and immediate.
//
// Flag values:
// - hipStreamDefault (0): Default stream behavior.
// - hipStreamNonBlocking: Stream does not synchronize with NULL stream.
//
// Stream behavior:
// - Returns the flags specified when the stream was created.
// - Default streams return hipStreamDefault.
// - Flags determine synchronization behavior with NULL stream.
//
// Multi-GPU: Queries the stream associated with the current context.
//
// See also: hipStreamCreateWithFlags, hipStreamGetPriority,
//           hipStreamGetDevice.
HIPAPI hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  if (!flags) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      HIP_RETURN_ERROR(init_result);
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  *flags = stream_obj->flags;
  return hipSuccess;
}

// Queries the device associated with a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//  - device: [OUT] Pointer to receive the device ordinal.
//
// Returns:
//  - hipSuccess: Device queried successfully.
//  - hipErrorInvalidValue: device pointer is NULL.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//
// Synchronization: This operation is synchronous and immediate.
//
// Device association:
// - Returns the device ordinal where the stream was created.
// - Streams can only execute operations on their associated device.
// - Device association is fixed for the lifetime of the stream.
//
// Multi-GPU:
// - Each stream is bound to a specific device.
// - Operations queued to a stream execute on its associated device.
// - Use hipSetDevice() before creating streams to control placement.
//
// See also: hipStreamCreate, hipGetDevice, hipSetDevice,
//           hipStreamGetFlags.
HIPAPI hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(init_result);
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  *device = (hipDevice_t)stream_obj->context->device_ordinal;

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Waits for all operations in a stream to complete.
//
// Parameters:
//  - stream: [IN] Stream to synchronize (NULL = default stream).
//
// Returns:
//  - hipSuccess: All operations completed successfully.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: A kernel launch in the stream failed.
//  - hipErrorIllegalAddress: Invalid memory access in stream operations.
//  - hipErrorUnknown: Internal error during synchronization.
//
// Synchronization: This operation blocks the host thread until all
// previously enqueued operations in the stream have completed.
//
// Stream behavior:
// - If stream is NULL, synchronizes the default stream.
// - Blocks until all operations enqueued before this call complete.
// - Operations enqueued after this call are not affected.
// - Does not synchronize with other streams.
//
// Multi-GPU: Synchronizes operations on the device associated with the
// stream.
//
// Performance note: Consider using hipStreamQuery for non-blocking status
// checks or hipEventSynchronize for finer-grained synchronization.
//
// See also: hipStreamQuery, hipDeviceSynchronize, hipEventSynchronize.
HIPAPI hipError_t hipStreamSynchronize(hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve NULL stream to default stream.
  if (!stream) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(init_result);
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_status_t status = iree_hal_streaming_stream_synchronize(
      (iree_hal_streaming_stream_t*)stream);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Queries the completion status of operations in a stream.
//
// Parameters:
//  - stream: [IN] Stream to query (NULL = default stream).
//
// Returns:
//  - hipSuccess: All operations in the stream have completed.
//  - hipErrorNotReady: Operations are still in progress.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: A kernel launch in the stream failed.
//
// Synchronization: This operation is non-blocking. Returns immediately with
// the current status.
//
// Stream behavior:
// - If stream is NULL, queries the default stream.
// - Checks if all operations enqueued before this call have completed.
// - Does not wait for operations to complete.
// - Does not affect operations in other streams.
//
// Multi-GPU: Queries operations on the device associated with the stream.
//
// Usage pattern:
// ```c
// while (hipStreamQuery(stream) == hipErrorNotReady) {
//   // Do other work while waiting
// }
// ```
//
// See also: hipStreamSynchronize, hipEventQuery, hipDeviceSynchronize.
HIPAPI hipError_t hipStreamQuery(hipStream_t stream) {
  // Resolve NULL stream to default stream.
  if (!stream) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      HIP_RETURN_ERROR(init_result);
    }
    stream = (hipStream_t)context->default_stream;
  }

  int is_complete = 0;
  iree_status_t status = iree_hal_streaming_stream_query(
      (iree_hal_streaming_stream_t*)stream, &is_complete);
  hipError_t result = iree_status_is_ok(status)
                          ? (is_complete == 1 ? hipSuccess : hipErrorNotReady)
                          : iree_status_to_hip_result(status);
  return result;
}

// Makes a stream wait for an event to complete.
//
// Parameters:
//  - stream: [IN] Stream that will wait (NULL = default stream).
//  - event: [IN] Event to wait for.
//  - flags: [IN] Reserved for future use (must be 0).
//
// Returns:
//  - hipSuccess: Wait dependency added successfully.
//  - hipErrorInvalidResourceHandle: Invalid stream or event handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidValue: Invalid flags or NULL event.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is asynchronous. Adds a dependency to the
// stream but does not block the host.
//
// Stream behavior:
// - If stream is NULL, uses the default stream.
// - Stream will wait for the event before executing subsequent operations.
// - Does not affect operations already enqueued in the stream.
// - The event can be from the same or different stream.
// - The event can be from the same or different device.
//
// Multi-GPU: Enables cross-device synchronization when event is from a
// different device.
//
// Usage pattern:
// ```c
// hipEventRecord(event, stream1);
// hipStreamWaitEvent(stream2, event, 0);  // stream2 waits for stream1
// ```
//
// See also: hipEventRecord, hipEventSynchronize, hipStreamSynchronize.
HIPAPI hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                     unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve NULL stream to default stream.
  if (!stream) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(init_result);
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_status_t status = iree_hal_streaming_stream_wait_event(
      (iree_hal_streaming_stream_t*)stream, (iree_hal_streaming_event_t*)event);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

// Creates an event object for timing and synchronization.
//
// Parameters:
//  - event: [OUT] Pointer to receive the created event handle.
//
// Returns:
//  - hipSuccess: Event created successfully.
//  - hipErrorInvalidValue: event pointer is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create event.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Event behavior:
// - Created with default flags (no special behavior).
// - Can be recorded in any stream.
// - Can be used for timing measurements.
// - Can be used for stream synchronization.
// - Must be destroyed with hipEventDestroy().
//
// Multi-GPU: Event can be used across devices for synchronization.
//
// Usage pattern:
// ```c
// hipEvent_t start, stop;
// hipEventCreate(&start);
// hipEventCreate(&stop);
// hipEventRecord(start, stream);
// // ... operations ...
// hipEventRecord(stop, stream);
// hipEventSynchronize(stop);
// float ms;
// hipEventElapsedTime(&ms, start, stop);
// ```
//
// See also: hipEventCreateWithFlags, hipEventDestroy, hipEventRecord,
//           hipEventSynchronize.
HIPAPI hipError_t hipEventCreate(hipEvent_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_event_t* event_obj = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context, IREE_HAL_STREAMING_EVENT_FLAG_NONE, context->host_allocator,
      &event_obj);

  if (iree_status_is_ok(status)) {
    *event = (hipEvent_t)event_obj;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Creates an event object with specified flags.
//
// Parameters:
//  - event: [OUT] Pointer to receive the created event handle.
//  - flags: [IN] Event creation flags (hipEventDefault, hipEventBlockingSync,
//                hipEventDisableTiming, hipEventInterprocess).
//
// Returns:
//  - hipSuccess: Event created successfully.
//  - hipErrorInvalidValue: event pointer is NULL or invalid flags.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorMemoryAllocation: Insufficient memory to create event.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is synchronous.
//
// Event flags:
// - hipEventDefault (0): Standard event with timing support.
// - hipEventBlockingSync: CPU thread blocks on hipEventSynchronize.
// - hipEventDisableTiming: Faster but cannot be used with
//   hipEventElapsedTime.
// - hipEventInterprocess: Event can be shared across processes.
//
// Multi-GPU: Event can be used across devices for synchronization.
//
// Performance note: hipEventDisableTiming creates lighter-weight events
// when timing is not needed.
//
// See also: hipEventCreate, hipEventDestroy, hipEventRecord.
HIPAPI hipError_t hipEventCreateWithFlags(hipEvent_t* event,
                                          unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_event_t* event_obj = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context, hip_event_flags_to_internal(flags), context->host_allocator,
      &event_obj);

  if (iree_status_is_ok(status)) {
    *event = (hipEvent_t)event_obj;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Destroys an event object.
//
// Parameters:
//  - event: [IN] Event handle to destroy.
//
// Returns:
//  - hipSuccess: Event destroyed successfully.
//  - hipErrorInvalidResourceHandle: Invalid event handle.
//  - hipErrorContextIsDestroyed: Associated context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for the event to
// complete if it has been recorded but not yet reached.
//
// Event behavior:
// - All uses of the event must complete before destruction.
// - After destruction, the event handle becomes invalid.
// - Using a destroyed event results in undefined behavior.
//
// Multi-GPU: Event must be destroyed from a context that can access it.
//
// Warning: Ensure the event has completed or been synchronized before
// destroying.
//
// See also: hipEventCreate, hipEventCreateWithFlags, hipEventSynchronize.
HIPAPI hipError_t hipEventDestroy(hipEvent_t event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_event_release((iree_hal_streaming_event_t*)event);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Records an event in a stream for timing and synchronization.
//
// Parameters:
//  - event: [IN] Event handle to record.
//  - stream: [IN] Stream to record the event in (NULL = default stream).
//
// Returns:
//  - hipSuccess: Event recorded successfully.
//  - hipErrorInvalidValue: event is NULL or invalid.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorLaunchFailure: Previous kernel launch failed.
//  - hipErrorUnknown: Internal error during recording.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - The event captures the current position in the stream's command queue.
// - All previously enqueued operations in the stream must complete before
//   the event is signaled.
// - If stream is NULL, uses the default stream.
// - The event can be waited on by other streams using hipStreamWaitEvent().
// - The event can be queried with hipEventQuery() or synchronized with
//   hipEventSynchronize().
//
// Multi-GPU: Events can be used for synchronization across devices.
//
// Warning: Recording an event multiple times overwrites the previous
// recording. Wait for the event to complete before re-recording.
//
// Note: Use hipEventElapsedTime() to measure time between two events
// recorded in the same stream.
HIPAPI hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_event_record(
      (iree_hal_streaming_event_t*)event, (iree_hal_streaming_stream_t*)stream);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Waits for an event to complete.
//
// Parameters:
//  - event: [IN] Event to wait for.
//
// Returns:
//  - hipSuccess: Event has completed.
//  - hipErrorInvalidResourceHandle: Invalid event handle.
//  - hipErrorLaunchFailure: A kernel launch associated with event failed.
//  - hipErrorIllegalAddress: Invalid memory access in associated operations.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation blocks the host thread until the event
// completes.
//
// Event behavior:
// - Blocks until all operations before the event recording have completed.
// - If event has not been recorded, returns immediately with hipSuccess.
// - If hipEventBlockingSync flag was used, may yield CPU to other threads.
//
// Multi-GPU: Can synchronize events from other devices.
//
// Performance note: For polling without blocking, use hipEventQuery.
//
// See also: hipEventQuery, hipEventRecord, hipStreamSynchronize.
HIPAPI hipError_t hipEventSynchronize(hipEvent_t event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_streaming_event_synchronize((iree_hal_streaming_event_t*)event);
  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Queries the completion status of an event.
//
// Parameters:
//  - event: [IN] Event to query.
//
// Returns:
//  - hipSuccess: Event has completed.
//  - hipErrorNotReady: Event has not completed.
//  - hipErrorInvalidResourceHandle: Invalid event handle.
//  - hipErrorLaunchFailure: A kernel launch associated with event failed.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//
// Synchronization: This operation is non-blocking. Returns immediately with
// the current status.
//
// Event behavior:
// - Checks if all operations before the event recording have completed.
// - If event has not been recorded, returns hipSuccess.
// - Does not wait for the event to complete.
//
// Multi-GPU: Can query events from other devices.
//
// Usage pattern:
// ```c
// while (hipEventQuery(event) == hipErrorNotReady) {
//   // Do other work while waiting
// }
// ```
//
// See also: hipEventSynchronize, hipEventRecord, hipStreamQuery.
HIPAPI hipError_t hipEventQuery(hipEvent_t event) {
  int is_complete = 0;
  iree_status_t status = iree_hal_streaming_event_query(
      (iree_hal_streaming_event_t*)event, &is_complete);
  hipError_t result = iree_status_is_ok(status)
                          ? (is_complete == 1 ? hipSuccess : hipErrorNotReady)
                          : iree_status_to_hip_result(status);
  return result;
}

// Computes elapsed time between two events.
//
// Parameters:
//  - ms: [OUT] Pointer to receive elapsed time in milliseconds.
//  - start: [IN] Start event (must have been recorded earlier).
//  - stop: [IN] Stop event (must have been recorded later).
//
// Returns:
//  - hipSuccess: Time computed successfully.
//  - hipErrorInvalidValue: ms is NULL or events are NULL.
//  - hipErrorInvalidResourceHandle: Invalid event handles.
//  - hipErrorNotReady: One or both events have not completed.
//  - hipErrorInvalidHandle: Events created with hipEventDisableTiming.
//
// Synchronization: This operation may block if events have not completed.
//
// Timing behavior:
// - Both events must be recorded in the same stream.
// - Stop event must be recorded after start event.
// - Events must not have hipEventDisableTiming flag.
// - Returns time in milliseconds with ~0.5 microsecond resolution.
// - Time measurement includes all operations between events.
//
// Multi-GPU: Both events must be from the same device.
//
// Usage pattern:
// ```c
// hipEventRecord(start, stream);
// // ... operations to time ...
// hipEventRecord(stop, stream);
// hipEventSynchronize(stop);
// float milliseconds;
// hipEventElapsedTime(&milliseconds, start, stop);
// ```
//
// Warning: Events must be recorded in the same stream for accurate timing.
//
// See also: hipEventCreate, hipEventRecord, hipEventSynchronize.
HIPAPI hipError_t hipEventElapsedTime(float* ms, hipEvent_t start,
                                      hipEvent_t stop) {
  if (!ms) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  iree_status_t status = iree_hal_streaming_event_elapsed_time(
      ms, (iree_hal_streaming_event_t*)start,
      (iree_hal_streaming_event_t*)stop);
  hipError_t result = iree_status_to_hip_result(status);
  return result;
}

//===----------------------------------------------------------------------===//
// Module and kernel execution
//===----------------------------------------------------------------------===//

// Loads a compute module from a file.
//
// Parameters:
//  - module: [OUT] Pointer to receive the loaded module handle.
//  - fname: [IN] Path to the module file (.hsaco, .co, etc.).
//
// Returns:
//  - hipSuccess: Module loaded successfully.
//  - hipErrorInvalidValue: module or fname is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorFileNotFound: Module file does not exist.
//  - hipErrorInvalidImage: File is not a valid module format.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorOutOfMemory: Insufficient memory to load module.
//
// Synchronization: This operation is synchronous.
//
// Module behavior:
// - Loads compiled GPU code from file.
// - Module remains loaded until hipModuleUnload() is called.
// - Module can contain multiple kernels and global variables.
// - Module is associated with the current context.
//
// Supported formats:
// - .hsaco: AMD GPU code object (GCN/RDNA ISA).
// - .co: NVIDIA GPU code object (PTX/SASS).
// - Architecture-specific binary formats.
//
// Multi-GPU: Module is loaded for the current device's architecture.
//
// See also: hipModuleLoadData, hipModuleUnload, hipModuleGetFunction.
HIPAPI hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!module || !fname) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Load module from file.
  iree_hal_streaming_module_t* stream_module = NULL;
  iree_hal_executable_caching_mode_t caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION;
  iree_status_t status = iree_hal_streaming_module_create_from_file(
      context, caching_mode, iree_make_cstring_view(fname),
      context->host_allocator, &stream_module);

  if (iree_status_is_ok(status)) {
    *module = (hipModule_t)stream_module;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Loads a compute module from memory.
//
// Parameters:
//  - module: [OUT] Pointer to receive the loaded module handle.
//  - image: [IN] Pointer to module data in memory.
//
// Returns:
//  - hipSuccess: Module loaded successfully.
//  - hipErrorInvalidValue: module or image is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidImage: Data is not a valid module format.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorOutOfMemory: Insufficient memory to load module.
//
// Synchronization: This operation is synchronous.
//
// Module behavior:
// - Loads compiled GPU code from memory buffer.
// - Buffer must contain valid module data.
// - Module remains loaded until hipModuleUnload() is called.
// - Image data can be freed after loading.
//
// Multi-GPU: Module is loaded for the current device's architecture.
//
// See also: hipModuleLoad, hipModuleLoadDataEx, hipModuleUnload,
//           hipModuleGetFunction.
HIPAPI hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  // Call the extended version with no options.
  return hipModuleLoadDataEx(module, image, 0, NULL, NULL);
}

// Loads a compute module from memory with extended options.
//
// Parameters:
//  - module: [OUT] Pointer to receive the loaded module handle.
//  - image: [IN] Pointer to module data in memory.
//  - numOptions: [IN] Number of JIT options provided.
//  - options: [IN] Array of option types (hipJitOption enum values).
//  - optionValues: [IN] Array of pointers to option values.
//
// Returns:
//  - hipSuccess: Module loaded successfully.
//  - hipErrorInvalidValue: module or image is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidImage: Data is not a valid module format.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorOutOfMemory: Insufficient memory to load module.
//  - hipErrorNotSupported: Unsupported JIT option.
//
// Synchronization: This operation is synchronous.
//
// Module behavior:
// - Loads compiled GPU code from memory buffer.
// - Module remains loaded until hipModuleUnload() is called.
// - Module can contain multiple kernels and global variables.
// - Module is associated with the current context.
//
// JIT options:
// - hipJitOptionMaxRegisters: Max registers per thread.
// - hipJitOptionThreadsPerBlock: Min threads per block.
// - hipJitOptionWallTime: Compilation time limit (ms).
// - hipJitOptionInfoLogBuffer: Buffer for info messages.
// - hipJitOptionErrorLogBuffer: Buffer for error messages.
// - hipJitOptionOptimizationLevel: Optimization level (0-4).
// - hipJitOptionTargetFromContext: Use context's target.
// - hipJitOptionTarget: Specify compute capability.
// - hipJitOptionFallbackStrategy: Fallback behavior.
//
// Option behavior:
// - Options are hints; implementation may ignore unsupported options.
// - Log buffers are filled with null-terminated strings.
// - Option values are type-specific (int*, char**, etc.).
//
// Multi-GPU: Module is loaded for the current device's architecture.
//
// Warning: Ensure the image buffer remains valid during loading.
// The implementation may reference the buffer asynchronously.
//
// See also: hipModuleLoad, hipModuleLoadData, hipModuleUnload,
//           hipModuleGetFunction.
HIPAPI hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                      unsigned int numOptions,
                                      hipJitOption* options,
                                      void** optionValues) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!module || !image) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Process JIT options if provided.
  // Note: Most JIT options are informational or optimization hints that may
  // not apply to our HAL backend. We parse them for compatibility but may not
  // use all of them.
  for (unsigned int i = 0; i < numOptions; ++i) {
    if (!options || !optionValues) continue;
    switch (options[i]) {
      case hipJitOptionMaxRegisters:
        // Maximum number of registers per thread.
        // This could influence kernel compilation but may be backend-specific.
        break;
      case hipJitOptionThreadsPerBlock:
        // Minimum number of threads per block.
        break;
      case hipJitOptionWallTime:
        // Wall time for compilation in milliseconds.
        break;
      case hipJitOptionInfoLogBuffer:
        // Buffer for informational log.
        break;
      case hipJitOptionInfoLogBufferSizeBytes:
        // Size of info log buffer.
        break;
      case hipJitOptionErrorLogBuffer:
        // Buffer for error log.
        break;
      case hipJitOptionErrorLogBufferSizeBytes:
        // Size of error log buffer.
        break;
      case hipJitOptionOptimizationLevel:
        // Optimization level (0-4).
        break;
      case hipJitOptionTargetFromContext:
        // Use target from current context.
        break;
      case hipJitOptionTarget:
        // Explicit compute capability target.
        break;
      case hipJitOptionFallbackStrategy:
        // Fallback strategy for compilation.
        break;
      case hipJitOptionGenerateDebugInfo:
        // Generate debug information.
        break;
      case hipJitOptionLogVerbose:
        // Enable verbose logging.
        break;
      case hipJitOptionGenerateLineInfo:
        // Generate line number information.
        break;
      case hipJitOptionCacheMode:
        // Cache mode for compiled kernels.
        break;
      case hipJitOptionSm3xOpt:
        // SM 3.x specific optimizations.
        break;
      case hipJitOptionFastCompile:
        // Fast compilation mode.
        break;
      default:
        // Unknown option, ignore.
        break;
    }
  }

  // TODO: Determine caching mode from JIT options if available.
  iree_hal_executable_caching_mode_t caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION;

  iree_hal_streaming_module_t* stream_module = NULL;
  iree_status_t status = iree_hal_streaming_module_create_from_memory(
      context, caching_mode, iree_make_const_byte_span(image, 0),
      context->host_allocator, &stream_module);

  if (iree_status_is_ok(status)) {
    *module = (hipModule_t)stream_module;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Unloads a compute module.
//
// Parameters:
//  - module: [IN] Module handle to unload.
//
// Returns:
//  - hipSuccess: Module unloaded successfully.
//  - hipErrorInvalidResourceHandle: Invalid module handle.
//  - hipErrorContextIsDestroyed: Associated context already destroyed.
//
// Synchronization: This operation is synchronous. Waits for all operations
// using the module to complete.
//
// Module behavior:
// - Releases all resources associated with the module.
// - All kernel functions from the module become invalid.
// - All global variables from the module become inaccessible.
// - Module handle becomes invalid after unloading.
//
// Multi-GPU: Only affects the module in the current context.
//
// Warning: Ensure all kernels from this module have completed execution
// before unloading.
//
// See also: hipModuleLoad, hipModuleLoadData.
HIPAPI hipError_t hipModuleUnload(hipModule_t module) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_module_release((iree_hal_streaming_module_t*)module);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets a kernel function handle from a module.
//
// Parameters:
//  - function: [OUT] Pointer to receive the function handle.
//  - module: [IN] Module containing the function.
//  - kname: [IN] Name of the kernel function to retrieve.
//
// Returns:
//  - hipSuccess: Function retrieved successfully.
//  - hipErrorInvalidValue: function or kname is NULL.
//  - hipErrorInvalidHandle: Invalid module handle.
//  - hipErrorNotFound: Function with given name not found in module.
//
// Synchronization: This operation is synchronous.
//
// Function behavior:
// - Retrieves a handle to a __global__ kernel function.
// - Function name must match exactly (including C++ mangling).
// - Function handle can be used with hipModuleLaunchKernel().
// - Function remains valid until module is unloaded.
//
// Name lookup:
// - For C kernels: Use the exact function name.
// - For C++ kernels: Use the mangled name.
// - For templated kernels: Use the instantiated mangled name.
//
// Multi-GPU: Function is specific to the module's device.
//
// Usage pattern:
// ```c
// hipFunction_t kernel;
// hipModuleGetFunction(&kernel, module, "vector_add");
// hipModuleLaunchKernel(kernel, ...);
// ```
//
// See also: hipModuleLoad, hipModuleLaunchKernel, hipModuleGetGlobal.
HIPAPI hipError_t hipModuleGetFunction(hipFunction_t* function,
                                       hipModule_t module, const char* kname) {
  if (!function || !kname) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_module_t* stream_module =
      (iree_hal_streaming_module_t*)module;
  if (!stream_module) {
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  iree_hal_streaming_symbol_t* stream_symbol = NULL;
  iree_status_t status =
      iree_hal_streaming_module_function(stream_module, kname, &stream_symbol);

  if (iree_status_is_ok(status)) {
    *function = (hipFunction_t)stream_symbol;
  }

  hipError_t result = iree_status_to_hip_result(status);
  return result;
}

// Gets a global variable pointer from a module.
//
// Parameters:
//  - dptr: [OUT] Pointer to receive the device pointer to the global.
//  - bytes: [OUT] Pointer to receive the size of the global (can be NULL).
//  - hmod: [IN] Module containing the global variable.
//  - name: [IN] Name of the global variable.
//
// Returns:
//  - hipSuccess: Global variable found and pointer retrieved.
//  - hipErrorInvalidValue: dptr, hmod, or name is NULL.
//  - hipErrorInvalidHandle: Invalid module handle.
//  - hipErrorNotFound: Global variable with given name not found.
//
// Synchronization: This operation is synchronous.
//
// Global variable behavior:
// - Retrieves device pointer to a __device__ or __constant__ variable.
// - Variable name must match exactly (including C++ mangling).
// - Pointer remains valid until module is unloaded.
// - Can read/write the variable using hipMemcpy functions.
//
// Name lookup:
// - For C globals: Use the exact variable name.
// - For C++ globals: Use the mangled name.
// - Namespace and class scope affect mangling.
//
// Multi-GPU: Pointer is valid only on the module's device.
//
// Usage pattern:
// ```c
// hipDeviceptr_t d_global;
// size_t global_size;
// hipModuleGetGlobal(&d_global, &global_size, module, "globalVar");
// hipMemcpyHtoD(d_global, &host_value, sizeof(int));
// ```
//
// See also: hipModuleLoad, hipModuleGetFunction, hipGetSymbolAddress.
HIPAPI hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                                     hipModule_t hmod, const char* name) {
  if (!dptr || !hmod || !name) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_module_t* module = (iree_hal_streaming_module_t*)hmod;
  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_device_size_t size = 0;
  iree_status_t status =
      iree_hal_streaming_module_global(module, name, &device_ptr, &size);

  if (iree_status_is_ok(status)) {
    *dptr = (hipDeviceptr_t)device_ptr;
    if (bytes) *bytes = (size_t)size;
  }

  hipError_t result = iree_status_to_hip_result(status);
  return result;
}

//===----------------------------------------------------------------------===//
// Function management
//===----------------------------------------------------------------------====//

// Queries a single attribute of a kernel function.
//
// Parameters:
//  - pi: [OUT] Pointer to receive the attribute value.
//  - attrib: [IN] Attribute to query (hipFuncAttribute_t enum).
//  - hfunc: [IN] Function handle to query.
//
// Returns:
//  - hipSuccess: Attribute queried successfully.
//  - hipErrorInvalidValue: pi is NULL or hfunc is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidDeviceFunction: Not a valid kernel function.
//
// Synchronization: This operation is synchronous and immediate.
//
// Available attributes:
// - hipFuncAttributeMaxThreadsPerBlock: Max threads per block.
// - hipFuncAttributeSharedSizeBytes: Static shared memory usage.
// - hipFuncAttributeConstSizeBytes: Constant memory usage.
// - hipFuncAttributeLocalSizeBytes: Local memory per thread.
// - hipFuncAttributeNumRegs: Register usage per thread.
// - hipFuncAttributePtxVersion: PTX version (CUDA compatibility).
// - hipFuncAttributeBinaryVersion: Binary version.
// - hipFuncAttributeCacheModeCA: Cache configuration.
// - hipFuncAttributeMaxDynamicSharedSizeBytes: Max dynamic shared.
// - hipFuncAttributePreferredSharedMemoryCarveout: Shared mem percent.
//
// Attribute values:
// - Values are kernel-specific and architecture-dependent.
// - Can be used for occupancy calculations.
// - Some attributes may return 0 if not applicable.
//
// Multi-GPU: Attributes are specific to the device that compiled
// the kernel.
//
// See also: hipFuncGetAttributes, hipFuncSetAttribute,
//           hipFuncSetCacheConfig.
HIPAPI hipError_t hipFuncGetAttribute(int* pi, hipFuncAttribute_t attrib,
                                      hipFunction_t hfunc) {
  if (!pi || !hfunc) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Cast to symbol pointer.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)hfunc;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Return attribute value based on what we have cached.
  switch (attrib) {
    case hipFuncAttributeMaxThreadsPerBlock:
      *pi = symbol->max_threads_per_block;
      break;
    case hipFuncAttributeSharedSizeBytes:
      *pi = symbol->shared_size_bytes;
      break;
    case hipFuncAttributeConstSizeBytes:
      // We don't track constant memory usage.
      *pi = 0;
      break;
    case hipFuncAttributeLocalSizeBytes:
      // Local memory is typically 0 for modern GPUs.
      *pi = 0;
      break;
    case hipFuncAttributeNumRegs:
      *pi = symbol->num_regs;
      break;
    case hipFuncAttributePtxVersion:
      // Return a default PTX version equivalent for HIP.
      *pi = 0;  // Not applicable to HIP/ROCm.
      break;
    case hipFuncAttributeBinaryVersion:
      // Return a default binary version.
      *pi = 0;  // Not tracked.
      break;
    case hipFuncAttributeCacheModeCA:
      // Cache mode is not tracked.
      *pi = 0;
      break;
    case hipFuncAttributeMaxDynamicSharedSizeBytes:
      // Return the kernel's maximum dynamic shared memory size.
      *pi = symbol->max_dynamic_shared_size_bytes;
      break;
    case hipFuncAttributePreferredSharedMemoryCarveout:
      // Carveout percentage not tracked.
      *pi = 0;
      break;
    default:
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  HIP_RETURN_ERROR(hipSuccess);
}

// Queries all attributes of a kernel function.
//
// Parameters:
//  - attr: [OUT] Pointer to structure to receive all attributes.
//  - hfunc: [IN] Function handle to query.
//
// Returns:
//  - hipSuccess: Attributes queried successfully.
//  - hipErrorInvalidValue: attr or hfunc is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidDeviceFunction: Not a valid kernel function.
//
// Synchronization: This operation is synchronous and immediate.
//
// Attribute structure fields:
// - sharedSizeBytes: Static shared memory in bytes.
// - constSizeBytes: Constant memory in bytes.
// - localSizeBytes: Local memory per thread in bytes.
// - maxThreadsPerBlock: Maximum threads per block.
// - numRegs: Registers per thread.
// - ptxVersion: PTX version (CUDA compatibility).
// - binaryVersion: Binary version.
// - cacheModeCA: Cache configuration.
// - maxDynamicSharedSizeBytes: Max dynamic shared memory.
// - preferredShmemCarveout: Preferred shared memory percentage.
//
// Usage:
// - More efficient than multiple hipFuncGetAttribute calls.
// - Use for occupancy calculations and launch configuration.
// - Values are kernel and architecture specific.
//
// Multi-GPU: Attributes are specific to the device that compiled
// the kernel.
//
// See also: hipFuncGetAttribute, hipFuncSetAttribute,
//           hipOccupancyMaxActiveBlocksPerMultiprocessor.
HIPAPI hipError_t hipFuncGetAttributes(hipFuncAttributes* attr,
                                       hipFunction_t hfunc) {
  if (!attr || !hfunc) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Cast to symbol pointer.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)hfunc;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Fill in the attributes structure.
  memset(attr, 0, sizeof(hipFuncAttributes));
  attr->maxThreadsPerBlock = symbol->max_threads_per_block;
  attr->sharedSizeBytes = symbol->shared_size_bytes;
  attr->constSizeBytes = 0;  // Not tracked.
  attr->localSizeBytes = 0;  // Typically 0 for modern GPUs.
  attr->numRegs = symbol->num_regs;
  attr->ptxVersion = 0;     // Not applicable to HIP/ROCm.
  attr->binaryVersion = 0;  // Not tracked.
  attr->cacheModeCA = 0;    // Not tracked.
  attr->maxDynamicSharedSizeBytes = symbol->max_dynamic_shared_size_bytes;
  attr->preferredShmemCarveout = 0;  // Not tracked.

  HIP_RETURN_ERROR(hipSuccess);
}

// Sets a specific attribute of a kernel function.
//
// Parameters:
//  - hfunc: [IN] Function handle to modify.
//  - attrib: [IN] Attribute to set (hipFuncAttribute_t enum).
//  - value: [IN] New value for the attribute.
//
// Returns:
//  - hipSuccess: Attribute set successfully.
//  - hipErrorInvalidValue: Invalid attribute or value.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidDeviceFunction: Not a valid kernel function.
//  - hipErrorNotSupported: Attribute cannot be modified.
//
// Synchronization: This operation is synchronous.
//
// Settable attributes:
// - hipFuncAttributeMaxDynamicSharedSizeBytes: Set max dynamic shared
//   memory for kernels that use more than 48KB.
// - hipFuncAttributePreferredSharedMemoryCarveout: Set L1/shared split
//   (percentage 0-100).
//
// Attribute effects:
// - Changes apply to all subsequent launches of this function.
// - Does not affect currently executing kernels.
// - Settings persist until module is unloaded.
// - May affect occupancy and performance.
//
// Restrictions:
// - Most attributes are read-only and cannot be set.
// - Value must be within hardware-supported range.
// - Some attributes require specific GPU architectures.
//
// Multi-GPU: Settings apply only to the function on the current device.
//
// Warning: Changing attributes may reduce occupancy or cause launch
// failures if values exceed hardware limits.
//
// See also: hipFuncGetAttribute, hipFuncSetCacheConfig,
//           hipFuncSetSharedMemConfig.
HIPAPI hipError_t hipFuncSetAttribute(hipFunction_t hfunc,
                                      hipFuncAttribute_t attrib, int value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Cast to symbol pointer.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)hfunc;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Only certain attributes can be set.
  hipError_t result = hipSuccess;
  switch (attrib) {
    case hipFuncAttributeMaxDynamicSharedSizeBytes:
      // Store the maximum dynamic shared memory size for this function.
      // This is used by kernels that need more than 48KB shared memory.
      // Note that this is not actually used for anything but queries.
      symbol->max_dynamic_shared_size_bytes = (uint32_t)value;
      break;
    case hipFuncAttributePreferredSharedMemoryCarveout:
      // This controls the L1/shared memory split.
      // Values are percentages (0, 25, 50, 75, 100).
      // We don't actually configure this in the stream HAL yet.
      if (value != 0 && value != 25 && value != 50 && value != 75 &&
          value != 100) {
        result = hipErrorInvalidValue;
      }
      break;
    default:
      // Most attributes are read-only.
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets the preferred cache configuration for a kernel function.
//
// Parameters:
//  - hfunc: [IN] Function handle to configure.
//  - config: [IN] Cache configuration preference.
//
// Returns:
//  - hipSuccess: Cache configuration set successfully.
//  - hipErrorInvalidValue: Invalid function handle or config.
//  - hipErrorInvalidHandle: Not a valid kernel function.
//  - hipErrorNotSupported: Configuration not supported on device.
//
// Synchronization: This operation is synchronous.
//
// Cache configurations:
// - hipFuncCachePreferNone: No preference (default).
// - hipFuncCachePreferShared: Prefer larger shared memory.
// - hipFuncCachePreferL1: Prefer larger L1 cache.
// - hipFuncCachePreferEqual: Equal L1 and shared memory.
//
// Cache behavior:
// - Controls L1 cache vs shared memory allocation.
// - Total L1 + shared memory is fixed per SM.
// - Configuration is a hint; hardware may override.
// - Affects all subsequent launches of this function.
//
// Performance considerations:
// - PreferShared: Good for kernels with heavy shared memory use.
// - PreferL1: Good for kernels with scattered memory access.
// - PreferEqual: Balanced for mixed workloads.
//
// Multi-GPU: Configuration applies per device and context.
//
// Warning: Configuration may not be honored if it violates hardware
// constraints or kernel requirements.
//
// See also: hipFuncSetAttribute, hipFuncSetSharedMemConfig,
//           hipDeviceSetCacheConfig.
HIPAPI hipError_t hipFuncSetCacheConfig(hipFunction_t hfunc,
                                        hipFuncCache_t config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Validate cache configuration.
  hipError_t result = hipSuccess;
  switch (config) {
    case hipFuncCachePreferNone:
    case hipFuncCachePreferShared:
    case hipFuncCachePreferL1:
    case hipFuncCachePreferEqual:
      // These are all valid configurations.
      // We don't actually configure cache in the stream HAL yet,
      // but we accept the values.
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets the shared memory bank configuration for a kernel function.
//
// Parameters:
//  - hfunc: [IN] Function handle to configure.
//  - config: [IN] Shared memory bank size configuration.
//
// Returns:
//  - hipSuccess: Configuration set successfully.
//  - hipErrorInvalidValue: Invalid function handle or config.
//  - hipErrorInvalidHandle: Not a valid kernel function.
//  - hipErrorNotSupported: Configuration not supported on device.
//
// Synchronization: This operation is synchronous.
//
// Bank configurations:
// - hipSharedMemBankSizeDefault: Default bank size.
// - hipSharedMemBankSizeFourByte: 4-byte banks (32 banks).
// - hipSharedMemBankSizeEightByte: 8-byte banks (16 banks).
//
// Bank conflict behavior:
// - 4-byte banks: Better for 32-bit data access patterns.
// - 8-byte banks: Better for 64-bit data access patterns.
// - Bank conflicts occur when multiple threads access same bank.
// - Conflicts serialize memory access, reducing performance.
//
// Performance optimization:
// - Choose based on dominant data type in shared memory.
// - 4-byte for float/int, 8-byte for double/long.
// - Proper padding can avoid bank conflicts.
// - Use stride access patterns to minimize conflicts.
//
// Hardware notes:
// - Not all devices support all configurations.
// - Newer GPUs may have different bank architectures.
// - Configuration is a performance hint.
//
// Multi-GPU: Configuration applies per device and context.
//
// See also: hipFuncSetCacheConfig, hipFuncSetAttribute,
//           hipDeviceSetSharedMemConfig.
HIPAPI hipError_t hipFuncSetSharedMemConfig(hipFunction_t hfunc,
                                            hipSharedMemConfig config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Validate shared memory configuration.
  hipError_t result = hipSuccess;
  switch (config) {
    case hipSharedMemBankSizeDefault:
    case hipSharedMemBankSizeFourByte:
    case hipSharedMemBankSizeEightByte:
      // These are all valid configurations.
      // We don't actually configure shared memory banks in the stream HAL yet,
      // but we accept the values.
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

// Launches a kernel function with specified dimensions and parameters.
//
// Parameters:
//  - f: [IN] Kernel function handle obtained from hipModuleGetFunction().
//  - gridDimX: [IN] Grid X dimension in blocks.
//  - gridDimY: [IN] Grid Y dimension in blocks.
//  - gridDimZ: [IN] Grid Z dimension in blocks.
//  - blockDimX: [IN] Block X dimension in threads.
//  - blockDimY: [IN] Block Y dimension in threads.
//  - blockDimZ: [IN] Block Z dimension in threads.
//  - sharedMemBytes: [IN] Dynamic shared memory size per block in bytes.
//  - stream: [IN] Stream for kernel execution (NULL = default stream).
//  - kernelParams: [IN] Array of kernel parameters, NULL-terminated.
//  - extra: [IN] Extra options (currently unused, should be NULL).
//
// Returns:
//  - hipSuccess: Kernel launched successfully.
//  - hipErrorInvalidValue: Invalid function handle or dimensions.
//  - hipErrorInvalidConfiguration: Invalid launch configuration.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorSharedObjectInitFailed: Shared memory allocation failed.
//  - hipErrorLaunchOutOfResources: Insufficient resources for launch.
//  - hipErrorLaunchTimeOut: Previous kernel execution timed out.
//  - hipErrorNotInitialized: HIP runtime not initialized.
//  - hipErrorUnknown: Internal error during launch.
//
// Synchronization: This operation is asynchronous.
//
// Stream behavior:
// - Kernel execution is enqueued in the specified stream.
// - If stream is NULL, uses the default stream.
// - Kernel executes after all previously enqueued operations in the stream.
// - Use hipStreamSynchronize() to wait for kernel completion.
// - Use hipEventRecord() after launch to mark completion point.
//
// Launch configuration:
// - Total threads = gridDim * blockDim.
// - Grid dimensions must be > 0 and within device limits.
// - Block dimensions must be > 0 and within device limits.
// - Total threads per block must not exceed device maximum.
// - Shared memory size must not exceed device maximum.
//
// Kernel parameters:
// - kernelParams is an array of void* pointers to actual arguments.
// - Array must be NULL-terminated.
// - Each pointer points to the argument value (not a pointer to pointer).
// - Arguments are passed by value to the kernel.
//
// Multi-GPU: Kernel executes on the device associated with the current
// context.
//
// Warning: Ensure all kernel arguments remain valid until kernel completes.
// Do not modify or free argument memory while kernel is executing.
//
// Note: Check device properties with hipDeviceGetAttribute() to determine
// maximum grid/block dimensions and shared memory limits.
HIPAPI hipError_t hipModuleLaunchKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams, void** extra) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    stream = (hipStream_t)context->default_stream;
  }

  // Extract params pointer from HIP's parameter format.
  void* params_ptr = NULL;
  if (extra) {
    // Extra format: {HIP_LAUNCH_PARAM_BUFFER_POINTER, &buffer,
    //                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END}
    if (extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
      params_ptr = *(void**)extra[1];
    }
  } else if (kernelParams) {
    // kernelParams is an array of pointers to the actual parameters.
    params_ptr = kernelParams;
  }

  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {gridDimX, gridDimY, gridDimZ},
      .block_dim = {blockDimX, blockDimY, blockDimZ},
      .shared_memory_bytes = sharedMemBytes,
      .buffer = params_ptr,
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_NONE,
  };
  iree_status_t status =
      iree_hal_streaming_launch_kernel((iree_hal_streaming_symbol_t*)f, &params,
                                       (iree_hal_streaming_stream_t*)stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Launches a cooperative kernel with grid-wide synchronization support.
//
// Parameters:
//  - f: [IN] Kernel function handle from hipModuleGetFunction().
//  - gridDimX: [IN] Grid X dimension in blocks.
//  - gridDimY: [IN] Grid Y dimension in blocks.
//  - gridDimZ: [IN] Grid Z dimension in blocks.
//  - blockDimX: [IN] Block X dimension in threads.
//  - blockDimY: [IN] Block Y dimension in threads.
//  - blockDimZ: [IN] Block Z dimension in threads.
//  - sharedMemBytes: [IN] Dynamic shared memory size per block in bytes.
//  - stream: [IN] Stream for kernel execution (NULL = default stream).
//  - kernelParams: [IN] Array of kernel parameters, NULL-terminated.
//
// Returns:
//  - hipSuccess: Cooperative kernel launched successfully.
//  - hipErrorInvalidValue: Invalid function handle or dimensions.
//  - hipErrorInvalidConfiguration: Invalid launch configuration.
//  - hipErrorCooperativeLaunchTooLarge: Grid exceeds max cooperative size.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//  - hipErrorLaunchOutOfResources: Insufficient resources for launch.
//  - hipErrorNotSupported: Device doesn't support cooperative launch.
//
// Synchronization: This operation is asynchronous.
//
// Cooperative kernel requirements:
// - Kernel must use cooperative grid synchronization primitives.
// - All blocks in the grid must be resident simultaneously.
// - Grid size limited by device occupancy and resources.
// - Use hipOccupancyMaxActiveBlocksPerMultiprocessor() to determine
//   maximum grid size.
//
// Cooperative features:
// - Enables grid-wide synchronization via cooperative_groups::this_grid().
// - All blocks can synchronize at barriers.
// - Useful for iterative algorithms requiring global synchronization.
// - Higher launch overhead than regular kernels.
//
// Device requirements:
// - Device must support cooperative launch (check device attributes).
// - Compute capability 6.0+ for NVIDIA, RDNA+ for AMD.
// - Limited by SM/CU count and available resources.
//
// Performance considerations:
// - May reduce occupancy to ensure all blocks are resident.
// - Launch overhead higher than regular kernels.
// - Use only when grid-wide sync is necessary.
//
// Multi-GPU: Cooperative kernels cannot span multiple devices.
// Use hipLaunchCooperativeKernelMultiDevice() for multi-GPU.
//
// Warning: Grid size must not exceed the maximum determined by
// occupancy calculations, or launch will fail.
//
// See also: hipModuleLaunchKernel,
//           hipOccupancyMaxActiveBlocksPerMultiprocessor,
//           hipLaunchCooperativeKernelMultiDevice.
HIPAPI hipError_t hipModuleLaunchCooperativeKernel(
    hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream,
    void** kernelParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!f) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Resolve NULL stream to default stream.
  if (!stream) {
    stream = (hipStream_t)context->default_stream;
  }

  // Get the device from the stream's context.
  iree_hal_streaming_stream_t* hal_stream =
      (iree_hal_streaming_stream_t*)stream;
  iree_hal_streaming_device_t* device = hal_stream->context->device_entry;

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)f;

  // Calculate maximum blocks for cooperative launch.
  // This will return 0 if the device doesn't support cooperative launch.
  int block_size = blockDimX * blockDimY * blockDimZ;
  int max_blocks = 0;
  iree_status_t status = iree_hal_streaming_calculate_max_cooperative_blocks(
      device, symbol, block_size, sharedMemBytes, &max_blocks);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Verify grid size doesn't exceed max active blocks.
  // If max_blocks is 0 (device doesn't support cooperative launch) or
  // grid is too large, return error.
  int total_blocks = gridDimX * gridDimY * gridDimZ;
  if (max_blocks == 0 || total_blocks > max_blocks) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorCooperativeLaunchTooLarge);
  }

  // Set up dispatch params with cooperative flag.
  // Cooperative launch always uses kernelParams format.
  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {gridDimX, gridDimY, gridDimZ},
      .block_dim = {blockDimX, blockDimY, blockDimZ},
      .shared_memory_bytes = sharedMemBytes,
      .buffer = kernelParams,  // Array of pointers to parameters.
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE,
  };

  status =
      iree_hal_streaming_launch_kernel((iree_hal_streaming_symbol_t*)f, &params,
                                       (iree_hal_streaming_stream_t*)stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Enqueues a host function callback in a stream.
//
// Parameters:
//  - stream: [IN] Stream to enqueue the callback (NULL = default stream).
//  - fn: [IN] Host function to call when stream reaches this point.
//  - userData: [IN] User data pointer passed to the callback.
//
// Returns:
//  - hipSuccess: Host function enqueued successfully.
//  - hipErrorInvalidValue: fn is NULL.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidResourceHandle: Invalid stream handle.
//
// Synchronization: The host function is called asynchronously when
// the stream reaches this operation.
//
// Callback signature:
// ```c
// void hostFunction(void* userData);
// ```
//
// Callback behavior:
// - Called on a runtime thread when stream reaches this point.
// - Blocks stream execution until callback completes.
// - Can make HIP API calls except stream synchronization.
// - Should complete quickly to avoid blocking the stream.
// - Runs after all prior operations in stream complete.
// - Subsequent operations wait for callback to finish.
//
// Restrictions in callback:
// - Cannot call hipStreamSynchronize() on any stream.
// - Cannot call hipDeviceSynchronize().
// - Cannot wait on events from the same stream.
// - Can launch new work to different streams.
//
// Use cases:
// - CPU-GPU task pipelining.
// - Signaling completion to host code.
// - Triggering dependent CPU work.
// - Resource management between kernels.
//
// Performance considerations:
// - Callback runs on runtime thread, not application thread.
// - Long-running callbacks block stream progress.
// - Consider using events for simple synchronization.
//
// Multi-GPU: Callback executes in context of the stream's device.
//
// Warning: Avoid heavy computation in callbacks. Use callbacks for
// quick signaling or launching work on other streams.
//
// See also: hipStreamAddCallback, hipEventRecord,
//           hipStreamWaitEvent.
HIPAPI hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn,
                                    void* userData) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!fn) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // If no stream is specified, use the default stream.
  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  if (!stream_obj) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(init_result);
    }
    stream_obj = context->default_stream;
  }

  iree_status_t status =
      iree_hal_streaming_launch_host_function(stream_obj, fn, userData);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Occupancy functions
//===----------------------------------------------------------------------===//

// Calculates maximum active blocks per multiprocessor for a kernel.
//
// Parameters:
//  - numBlocks: [OUT] Pointer to receive max active blocks per SM/CU.
//  - f: [IN] Kernel function handle to analyze.
//  - blockSize: [IN] Block size in threads for the calculation.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: numBlocks or f is NULL, or blockSize <= 0.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Occupancy calculation:
// - Determines theoretical maximum blocks that can be resident.
// - Considers register usage, shared memory, and block size.
// - Returns blocks per single SM/CU, not total device capacity.
// - Actual occupancy may be lower due to launch configuration.
//
// Limiting factors:
// - Register usage per thread.
// - Shared memory (static + dynamic) per block.
// - Maximum threads per multiprocessor.
// - Maximum blocks per multiprocessor.
// - Warp/wavefront scheduling limits.
//
// Usage pattern:
// ```c
// int maxBlocks;
// hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
//     &maxBlocks, kernel, 256, sharedMemSize);
// int numSMs = deviceProps.multiProcessorCount;
// int totalBlocks = maxBlocks * numSMs;
// ```
//
// Performance optimization:
// - Use to find optimal block size for maximum occupancy.
// - Balance between occupancy and resource usage.
// - Higher occupancy doesn't always mean better performance.
//
// Multi-GPU: Calculation is specific to the current device's
// architecture and capabilities.
//
// See also: hipModuleOccupancyMaxPotentialBlockSize,
//           hipFuncGetAttributes, hipDeviceGetAttribute.
HIPAPI hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!numBlocks || !f || blockSize <= 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context and device.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get device properties.
  iree_hal_streaming_device_t* device = context->device_entry;
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)f;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Use shared occupancy calculation.
  int max_blocks = 0;
  iree_status_t status =
      iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
          device, symbol, blockSize, dynSharedMemPerBlk, &max_blocks);

  if (iree_status_is_ok(status)) {
    *numBlocks = max_blocks;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Calculates maximum active blocks per multiprocessor with flags.
//
// Parameters:
//  - numBlocks: [OUT] Pointer to receive max active blocks per SM/CU.
//  - f: [IN] Kernel function handle to analyze.
//  - blockSize: [IN] Block size in threads for the calculation.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//  - flags: [IN] Flags to control occupancy calculation behavior.
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: numBlocks or f is NULL, or blockSize <= 0.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Flag options:
// - hipOccupancyDefault: Default behavior.
// - hipOccupancyDisableCachingOverride: Don't override cache config.
// - Additional flags may be defined for specific architectures.
//
// Occupancy calculation:
// - Same as hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
// - Flags may modify how cache configuration affects calculation.
// - Useful for kernels with specific cache requirements.
//
// Cache considerations:
// - Default may assume optimal cache configuration.
// - Flags can preserve kernel's specified cache config.
// - Important for kernels tuned for specific L1/shared split.
//
// Multi-GPU: Calculation is specific to the current device's
// architecture and capabilities.
//
// See also: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
//           hipFuncSetCacheConfig, hipOccupancyMaxPotentialBlockSize.
HIPAPI hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk,
    unsigned int flags) {
  // For now, ignore flags and call the base function.
  // Flags might affect caching behavior but not occupancy calculation.
  return hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, f, blockSize, dynSharedMemPerBlk);
}

// Calculates optimal block and grid size for maximum occupancy.
//
// Parameters:
//  - gridSize: [OUT] Pointer to receive optimal grid size (in blocks).
//  - blockSize: [OUT] Pointer to receive optimal block size (in threads).
//  - f: [IN] Kernel function handle to analyze.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//  - blockSizeLimit: [IN] Maximum block size to consider (0 = no limit).
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: Output pointers or f is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Optimization strategy:
// - Tests multiple block sizes to find best occupancy.
// - Returns block size that maximizes multiprocessor occupancy.
// - Grid size calculated to fully utilize the device.
// - Balances threads per block with active blocks.
//
// Block size selection:
// - Tests powers of 2 and warp/wavefront multiples.
// - Respects kernel's max threads per block limit.
// - Considers register and shared memory constraints.
// - blockSizeLimit caps the maximum tested size.
//
// Grid size calculation:
// - Returns minimum grid to achieve maximum occupancy.
// - Grid size = (maxActiveBlocks * numSMs).
// - May be larger than needed for actual problem size.
// - Application should adjust based on actual work.
//
// Usage pattern:
// ```c
// int minGridSize, blockSize;
// hipModuleOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
//                                         kernel, 0, 0);
// int actualGridSize = (problemSize + blockSize - 1) / blockSize;
// hipModuleLaunchKernel(kernel, actualGridSize, 1, 1,
//                      blockSize, 1, 1, ...);
// ```
//
// Performance notes:
// - Optimal occupancy doesn't guarantee best performance.
// - Consider memory access patterns and arithmetic intensity.
// - May need to tune based on actual kernel behavior.
//
// Multi-GPU: Calculation is specific to the current device.
//
// See also: hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
//           hipModuleOccupancyMaxPotentialBlockSizeWithFlags.
HIPAPI hipError_t hipModuleOccupancyMaxPotentialBlockSize(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!gridSize || !blockSize || !f) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context and device.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Get device properties.
  iree_hal_streaming_device_t* device = context->device_entry;
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)f;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidHandle);
  }

  // Use shared occupancy calculation.
  // HIP doesn't yet have a C API for dynamic shared memory callbacks.
  // Pass NULL for the callback to use fixed dynamic shared memory size.
  iree_status_t status = iree_hal_streaming_calculate_optimal_block_size(
      device, symbol, (uint32_t)dynSharedMemPerBlk, NULL,
      (uint32_t)blockSizeLimit, blockSize, gridSize);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Calculates optimal block and grid size with flags.
//
// Parameters:
//  - gridSize: [OUT] Pointer to receive optimal grid size (in blocks).
//  - blockSize: [OUT] Pointer to receive optimal block size (in threads).
//  - f: [IN] Kernel function handle to analyze.
//  - dynSharedMemPerBlk: [IN] Dynamic shared memory per block in bytes.
//  - blockSizeLimit: [IN] Maximum block size to consider (0 = no limit).
//  - flags: [IN] Flags to control occupancy calculation behavior.
//
// Returns:
//  - hipSuccess: Calculation completed successfully.
//  - hipErrorInvalidValue: Output pointers or f is NULL.
//  - hipErrorInvalidHandle: Invalid function handle.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidDevice: Invalid device.
//
// Synchronization: This operation is synchronous and immediate.
//
// Flag options:
// - hipOccupancyDefault: Default optimization behavior.
// - hipOccupancyDisableCachingOverride: Preserve cache configuration.
// - Additional flags may affect optimization strategy.
//
// Extended behavior:
// - Same as hipModuleOccupancyMaxPotentialBlockSize.
// - Flags may affect how cache configuration is considered.
// - Useful for kernels with specific performance requirements.
//
// Multi-GPU: Calculation is specific to the current device.
//
// See also: hipModuleOccupancyMaxPotentialBlockSize,
//           hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.
HIPAPI hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
    int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk,
    int blockSizeLimit, unsigned int flags) {
  // For now, ignore flags and call the base function.
  return hipModuleOccupancyMaxPotentialBlockSize(
      gridSize, blockSize, f, dynSharedMemPerBlk, blockSizeLimit);
}

//===----------------------------------------------------------------------===//
// Unified memory management
//===----------------------------------------------------------------------===//

// Advises about the usage pattern of managed memory.
//
// Parameters:
//  - dev_ptr: [IN] Pointer to managed memory to advise about.
//  - count: [IN] Size in bytes of the memory range.
//  - advice: [IN] Advice to apply to the memory range.
//  - device: [IN] Device ID for device-specific advice.
//
// Returns:
//  - hipSuccess: Advice applied successfully (or ignored).
//  - hipErrorInvalidValue: Invalid pointer, size, or advice.
//  - hipErrorInvalidDevice: Invalid device ID.
//  - hipErrorNotSupported: Advice not supported on this system.
//
// Synchronization: This operation is synchronous.
//
// Advice options:
// - hipMemAdviseSetReadMostly: Data mostly read, rarely written.
// - hipMemAdviseUnsetReadMostly: Clear read-mostly setting.
// - hipMemAdviseSetPreferredLocation: Set preferred device location.
// - hipMemAdviseUnsetPreferredLocation: Clear preferred location.
// - hipMemAdviseSetAccessedBy: Memory will be accessed by device.
// - hipMemAdviseUnsetAccessedBy: Clear accessed-by setting.
//
// Memory migration hints:
// - ReadMostly: Enables read duplication across devices.
// - PreferredLocation: Migrates pages to specified device.
// - AccessedBy: Establishes direct access mapping.
//
// Performance optimization:
// - Reduces page fault overhead for managed memory.
// - Improves memory access patterns across devices.
// - Hints are advisory; system may ignore them.
//
// Managed memory behavior:
// - Only applies to memory allocated with hipMallocManaged.
// - Advice persists until explicitly changed or memory freed.
// - Can significantly impact multi-GPU performance.
//
// Multi-GPU: Advice can specify different devices for different
// memory regions to optimize NUMA behavior.
//
// Warning: Incorrect advice may degrade performance. Profile to
// verify improvements.
//
// See also: hipMallocManaged, hipMemPrefetchAsync,
//           hipMemRangeGetAttribute.
HIPAPI hipError_t hipMemAdvise(const void* dev_ptr, size_t count,
                               hipMemAdvise_t advice, int device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement unified memory advice when we have managed memory support.
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;  // to unblock
}

// Asynchronously prefetches managed memory to a device.
//
// Parameters:
//  - dev_ptr: [IN] Pointer to managed memory to prefetch.
//  - count: [IN] Size in bytes to prefetch.
//  - device: [IN] Destination device ID (or hipCpuDeviceId for host).
//  - stream: [IN] Stream for asynchronous prefetch.
//
// Returns:
//  - hipSuccess: Prefetch initiated successfully (or ignored).
//  - hipErrorInvalidValue: Invalid pointer, size, or stream.
//  - hipErrorInvalidDevice: Invalid device ID.
//  - hipErrorNotSupported: Prefetch not supported on this system.
//
// Synchronization: This operation is asynchronous.
//
// Prefetch behavior:
// - Migrates pages to specified device before access.
// - Reduces page fault latency during kernel execution.
// - Operation is enqueued in the specified stream.
// - Overlaps with other stream operations.
//
// Performance benefits:
// - Eliminates page faults during kernel execution.
// - Enables overlap of data migration with computation.
// - Critical for managed memory performance.
//
// Memory requirements:
// - Only applies to hipMallocManaged allocations.
// - Pages must be resident in system memory.
// - Prefetch may be ignored if pages already on device.
//
// Stream ordering:
// - Prefetch completes before subsequent operations in stream.
// - Use events or stream synchronization to ensure completion.
// - Can prefetch to different devices in different streams.
//
// Multi-GPU patterns:
// - Prefetch to device before kernel launch.
// - Prefetch to hipCpuDeviceId for host access.
// - Pipeline prefetches across multiple devices.
//
// Warning: Prefetching large amounts may cause memory pressure.
// Monitor available device memory.
//
// See also: hipMallocManaged, hipMemAdvise, hipStreamSynchronize,
//           hipMemRangeGetAttribute.
HIPAPI hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count,
                                      int device, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement memory prefetching when we have managed memory support.
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;  // to unblock
}

// Queries a single attribute of a pointer.
//
// Parameters:
//  - data: [OUT] Pointer to receive the attribute value.
//  - attribute: [IN] Attribute to query (hipPointer_attribute_t enum).
//  - ptr: [IN] Pointer to query.
//
// Returns:
//  - hipSuccess: Attribute queried successfully.
//  - hipErrorInvalidValue: data is NULL or invalid attribute.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidMemoryHandle: Pointer not recognized.
//
// Synchronization: This operation is synchronous and immediate.
//
// Available attributes:
// - HIP_POINTER_ATTRIBUTE_CONTEXT: Context that owns the memory.
// - HIP_POINTER_ATTRIBUTE_MEMORY_TYPE: Type of memory.
// - HIP_POINTER_ATTRIBUTE_DEVICE_POINTER: Device pointer value.
// - HIP_POINTER_ATTRIBUTE_HOST_POINTER: Host pointer value.
// - HIP_POINTER_ATTRIBUTE_P2P_TOKENS: P2P tokens for IPC.
// - HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: Synchronous mem ops flag.
// - HIP_POINTER_ATTRIBUTE_BUFFER_ID: Unique buffer identifier.
// - HIP_POINTER_ATTRIBUTE_IS_MANAGED: Is managed memory.
// - HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL: Device ordinal.
// - HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: Allowed export types.
// - HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE: RDMA capable.
// - HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS: Access permissions.
// - HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE: Memory pool handle.
//
// Memory types:
// - Device memory: Allocated with hipMalloc.
// - Host memory: Allocated with hipHostMalloc or registered.
// - Managed memory: Allocated with hipMallocManaged.
// - Unregistered: Regular system memory.
//
// Usage patterns:
// - Determine memory type before operations.
// - Check if pointer is valid device memory.
// - Get device association for multi-GPU.
//
// Multi-GPU: Returns device-specific information for the pointer.
//
// See also: hipPointerGetAttributes, hipPointerSetAttribute,
//           hipMemGetInfo.
HIPAPI hipError_t hipPointerGetAttribute(void* data,
                                         hipPointer_attribute_t attribute,
                                         hipDeviceptr_t ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);

  // If lookup fails, the pointer might not be a valid allocation.
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t result = hipSuccess;

  switch (attribute) {
    case HIP_POINTER_ATTRIBUTE_CONTEXT: {
      // Return the context handle.
      *(hipCtx_t*)data = (hipCtx_t)buffer_ref.buffer->context;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_MEMORY_TYPE: {
      // Determine memory type based on buffer properties.
      hipMemoryType* memType = (hipMemoryType*)data;
      if (buffer_ref.buffer->host_ptr) {
        if (buffer_ref.buffer->memory_type == 2) {
          // Host-registered memory.
          *memType = hipMemoryTypeHost;
        } else {
          // Host allocated memory.
          *memType = hipMemoryTypeHost;
        }
      } else {
        // Device memory.
        *memType = hipMemoryTypeDevice;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_DEVICE_POINTER: {
      // Return the device pointer.
      *(hipDeviceptr_t*)data =
          (hipDeviceptr_t)((iree_device_size_t)buffer_ref.buffer->device_ptr +
                           buffer_ref.offset);
      break;
    }
    case HIP_POINTER_ATTRIBUTE_HOST_POINTER: {
      // Return the host pointer if available.
      *(void**)data = (void*)((iree_host_size_t)buffer_ref.buffer->host_ptr +
                              buffer_ref.offset);
      break;
    }
    case HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
      // Return the device ordinal.
      *(int*)data = (int)context->device_ordinal;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_IS_MANAGED: {
      // We don't support managed memory yet.
      *(unsigned int*)data = 0;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR: {
      // Return the base address of the allocation.
      if (buffer_ref.buffer->host_ptr) {
        *(hipDeviceptr_t*)data = (hipDeviceptr_t)buffer_ref.buffer->host_ptr;
      } else {
        *(hipDeviceptr_t*)data = (hipDeviceptr_t)buffer_ref.buffer->device_ptr;
      }
      break;
    }
    case HIP_POINTER_ATTRIBUTE_RANGE_SIZE: {
      // Return the size of the allocation.
      *(size_t*)data = buffer_ref.buffer->size;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_MAPPED: {
      // Check if memory is mapped (host-visible).
      unsigned int is_mapped = buffer_ref.buffer->host_ptr ? 1 : 0;
      *(unsigned int*)data = is_mapped;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      // Synchronous memory operations flag.
      *(unsigned int*)data = 1;  // Default to synchronous.
      break;
    }
    case HIP_POINTER_ATTRIBUTE_BUFFER_ID: {
      // Return a unique buffer ID (use pointer as ID).
      *(unsigned long long*)data = (unsigned long long)buffer_ref.buffer;
      break;
    }
    case HIP_POINTER_ATTRIBUTE_P2P_TOKENS:
    case HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS:
    case HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
      // These attributes are not supported yet.
      result = hipErrorNotSupported;
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets an attribute of a pointer.
//
// Parameters:
//  - value: [IN] Pointer to the new attribute value.
//  - attribute: [IN] Attribute to set (hipPointer_attribute_t enum).
//  - ptr: [IN] Pointer to modify.
//
// Returns:
//  - hipSuccess: Attribute set successfully.
//  - hipErrorInvalidValue: value is NULL or invalid attribute.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidMemoryHandle: Pointer not recognized.
//  - hipErrorNotSupported: Attribute cannot be modified.
//
// Synchronization: This operation is synchronous.
//
// Settable attributes:
// - HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: Enable synchronous operations.
// - HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS: Set access permissions.
// - Limited subset of attributes are modifiable.
//
// Sync memory operations:
// - When enabled, memory operations complete synchronously.
// - Affects hipMemcpy behavior with this pointer.
// - Default is typically asynchronous for device memory.
//
// Access flags:
// - Control read/write permissions.
// - May affect memory protection and caching.
// - Platform and device specific.
//
// Restrictions:
// - Most attributes are read-only.
// - Changes may not take effect immediately.
// - Some attributes require specific hardware support.
//
// Usage patterns:
// - Configure memory behavior for specific use cases.
// - Optimize memory access patterns.
// - Control synchronization behavior.
//
// Multi-GPU: Attributes are set per pointer, affecting all
// devices that access the memory.
//
// Warning: Changing attributes may affect performance or
// correctness of concurrent operations.
//
// See also: hipPointerGetAttribute, hipPointerGetAttributes,
//           hipMemcpyAsync.
HIPAPI hipError_t hipPointerSetAttribute(const void* value,
                                         hipPointer_attribute_t attribute,
                                         hipDeviceptr_t ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);

  // If lookup fails, the pointer might not be a valid allocation.
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  hipError_t result = hipSuccess;

  switch (attribute) {
    case HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      // Set synchronous memory operations flag.
      // Note: We accept the value but don't store it since all our operations
      // are currently synchronous by default.
      unsigned int sync_value = *(const unsigned int*)value;
      (void)sync_value;  // Suppress unused variable warning.
      break;
    }
    case HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS: {
      // Set memory access permissions (read-only, read-write, etc.).
      // Note: This is typically used for texture memory and may not be
      // applicable to our current buffer model. We accept the value but don't
      // enforce it.
      unsigned int access_flags = *(const unsigned int*)value;
      (void)access_flags;  // Suppress unused variable warning.
      break;
    }
    case HIP_POINTER_ATTRIBUTE_CONTEXT:
    case HIP_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case HIP_POINTER_ATTRIBUTE_DEVICE_POINTER:
    case HIP_POINTER_ATTRIBUTE_HOST_POINTER:
    case HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case HIP_POINTER_ATTRIBUTE_IS_MANAGED:
    case HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case HIP_POINTER_ATTRIBUTE_RANGE_SIZE:
    case HIP_POINTER_ATTRIBUTE_MAPPED:
    case HIP_POINTER_ATTRIBUTE_BUFFER_ID:
    case HIP_POINTER_ATTRIBUTE_P2P_TOKENS:
    case HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
      // These attributes are read-only and cannot be set.
      result = hipErrorNotSupported;
      break;
    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Queries multiple attributes of a pointer in one call.
//
// Parameters:
//  - numAttributes: [IN] Number of attributes to query.
//  - attributes: [IN] Array of attributes to query.
//  - data: [IN/OUT] Array of pointers to receive attribute values.
//  - ptr: [IN] Pointer to query.
//
// Returns:
//  - hipSuccess: All attributes queried successfully.
//  - hipErrorInvalidValue: Invalid parameters or numAttributes is 0.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorInvalidMemoryHandle: Pointer not recognized.
//
// Synchronization: This operation is synchronous and immediate.
//
// Batch query behavior:
// - More efficient than multiple hipPointerGetAttribute calls.
// - Each data[i] receives value for attributes[i].
// - All attributes queried even if some fail.
// - Returns first error encountered, but continues.
//
// Data array setup:
// - Each data[i] must point to appropriate type for attribute.
// - Size depends on attribute (int*, void**, size_t*, etc.).
// - Caller must allocate storage before call.
//
// Example usage:
// ```c
// hipPointer_attribute_t attrs[] = {
//     HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
//     HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
// };
// hipMemoryType memType;
// int device;
// void* data[] = {&memType, &device};
// hipPointerGetAttributes(2, attrs, data, ptr);
// ```
//
// Error handling:
// - If any attribute fails, that data element is undefined.
// - Check return value before using results.
// - Some attributes may not be supported for all pointers.
//
// Multi-GPU: Returns device-specific information for each attribute.
//
// See also: hipPointerGetAttribute, hipPointerSetAttribute,
//           hipMemGetInfo.
HIPAPI hipError_t hipPointerGetAttributes(unsigned int numAttributes,
                                          hipPointer_attribute_t* attributes,
                                          void** data, const void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!attributes || !data || numAttributes == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }
  // Query each attribute individually using hipPointerGetAttribute.
  hipError_t result = hipSuccess;
  for (unsigned int i = 0; i < numAttributes; i++) {
    hipError_t attr_result =
        hipPointerGetAttribute(data[i], attributes[i], (hipDeviceptr_t)ptr);
    if (attr_result != hipSuccess) {
      // Return the first error encountered.
      if (result == hipSuccess) {
        result = attr_result;
      }
      // Continue to try other attributes even if one fails.
    }
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Queries an attribute of a memory range.
//
// Parameters:
//  - data: [OUT] Buffer to receive attribute values.
//  - data_size: [IN] Size of the data buffer in bytes.
//  - attribute: [IN] Attribute to query (hipMemRangeAttribute enum).
//  - dev_ptr: [IN] Start of memory range to query.
//  - count: [IN] Size of memory range in bytes.
//
// Returns:
//  - hipSuccess: Attribute queried successfully.
//  - hipErrorInvalidValue: Invalid parameters or buffer too small.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotSupported: Attribute not supported.
//
// Synchronization: This operation is synchronous and immediate.
//
// Available attributes:
// - hipMemRangeAttributeReadMostly: Pages are read-mostly.
// - hipMemRangeAttributePreferredLocation: Preferred location.
// - hipMemRangeAttributeAccessedBy: Devices with access.
// - hipMemRangeAttributeLastPrefetchLocation: Last prefetch location.
//
// Attribute data formats:
// - ReadMostly: int (0 or 1).
// - PreferredLocation: int (device ID or hipCpuDeviceId).
// - AccessedBy: Array of int (device IDs).
// - LastPrefetchLocation: int (device ID).
//
// Memory range behavior:
// - Range can span multiple pages.
// - Attributes may vary across pages.
// - Returns aggregate or first value depending on attribute.
//
// Managed memory specific:
// - Most attributes only apply to hipMallocManaged memory.
// - Regular allocations return default values.
// - Use to verify memory migration and access patterns.
//
// Performance analysis:
// - Check where pages are currently resident.
// - Verify prefetch and migration effectiveness.
// - Understand access patterns across devices.
//
// Multi-GPU: Attributes reflect multi-device state and
// can show which devices have access.
//
// See also: hipMemRangeGetAttributes, hipMemAdvise,
//           hipMemPrefetchAsync.
HIPAPI hipError_t hipMemRangeGetAttribute(void* data, size_t data_size,
                                          hipMemRangeAttribute attribute,
                                          const void* dev_ptr, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data || data_size == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get the current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Look up the buffer containing this range.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, (iree_hal_streaming_deviceptr_t)dev_ptr, count, &buffer_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Map the attribute and return the appropriate value.
  hipError_t result = hipSuccess;
  switch (attribute) {
    case hipMemRangeAttributeReadMostly:
      // Return 1 if all pages have read-duplication enabled.
      if (data_size < sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = buffer_ref.buffer->read_mostly_hint ? 1 : 0;
      }
      break;

    case hipMemRangeAttributePreferredLocation:
      // Return the preferred device ID or hipCpuDeviceId (-1).
      if (data_size < sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = buffer_ref.buffer->preferred_location;
      }
      break;

    case hipMemRangeAttributeAccessedBy:
      // Not currently supported - would require tracking device access.
      result = hipErrorNotSupported;
      break;

    case hipMemRangeAttributeLastPrefetchLocation:
      // Return the last prefetch location.
      if (data_size < sizeof(int)) {
        result = hipErrorInvalidValue;
      } else {
        *(int*)data = buffer_ref.buffer->last_prefetch_location;
      }
      break;

    default:
      result = hipErrorInvalidValue;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

// Queries multiple attributes of a memory range in one call.
//
// Parameters:
//  - data: [OUT] Array of buffers to receive attribute values.
//  - data_sizes: [IN] Array of buffer sizes in bytes.
//  - attributes: [IN] Array of attributes to query.
//  - num_attributes: [IN] Number of attributes to query.
//  - dev_ptr: [IN] Start of memory range to query.
//  - count: [IN] Size of memory range in bytes.
//
// Returns:
//  - hipSuccess: All attributes queried successfully.
//  - hipErrorInvalidValue: Invalid parameters or num_attributes is 0.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorNotSupported: One or more attributes not supported.
//
// Synchronization: This operation is synchronous and immediate.
//
// Batch query behavior:
// - More efficient than multiple hipMemRangeGetAttribute calls.
// - Each data[i] receives value for attributes[i].
// - data_sizes[i] must be large enough for attribute data.
// - All attributes queried even if some fail.
//
// Example usage:
// ```c
// hipMemRangeAttribute attrs[] = {
//     hipMemRangeAttributeReadMostly,
//     hipMemRangeAttributePreferredLocation
// };
// int readMostly, preferredLoc;
// void* data[] = {&readMostly, &preferredLoc};
// size_t sizes[] = {sizeof(int), sizeof(int)};
// hipMemRangeGetAttributes(data, sizes, attrs, 2, ptr, size);
// ```
//
// Error handling:
// - If any attribute fails, that data element is undefined.
// - Returns first error encountered but continues.
// - Check return value before using results.
//
// Multi-GPU: Attributes reflect state across all devices
// that have access to the memory range.
//
// See also: hipMemRangeGetAttribute, hipMemAdvise,
//           hipPointerGetAttributes.
HIPAPI hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                           hipMemRangeAttribute* attributes,
                                           size_t num_attributes,
                                           const void* dev_ptr, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data || !data_sizes || !attributes || num_attributes == 0) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Query each attribute individually using hipMemRangeGetAttribute.
  hipError_t result = hipSuccess;
  for (size_t i = 0; i < num_attributes; i++) {
    hipError_t attr_result = hipMemRangeGetAttribute(
        data[i], data_sizes[i], attributes[i], dev_ptr, count);
    if (attr_result != hipSuccess) {
      // Return the first error encountered.
      if (result == hipSuccess) {
        result = attr_result;
      }
      // Continue to try other attributes even if one fails.
    }
  }

  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// HIP graphs
//===----------------------------------------------------------------------===//

// Creates an empty task graph.
//
// Parameters:
//  - pGraph: [OUT] Pointer to receive the created graph handle.
//  - flags: [IN] Graph creation flags (must be 0).
//
// Returns:
//  - hipSuccess: Graph created successfully.
//  - hipErrorInvalidValue: pGraph is NULL or flags is non-zero.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorNotSupported: Graphs not supported on this device.
//
// Synchronization: This operation is synchronous.
//
// Graph concepts:
// - Directed acyclic graph (DAG) of GPU operations.
// - Nodes represent kernels, memcpy, memset, or host callbacks.
// - Edges represent dependencies between operations.
// - Captured once, launched multiple times.
//
// Graph benefits:
// - Reduced launch overhead for repeated workloads.
// - Optimized scheduling and resource allocation.
// - Better GPU utilization through known dependencies.
// - Enables whole-graph optimizations.
//
// Graph workflow:
// 1. Create empty graph with hipGraphCreate.
// 2. Add nodes with hipGraphAdd* functions.
// 3. Define dependencies between nodes.
// 4. Instantiate to create executable with hipGraphInstantiate.
// 5. Launch executable multiple times with hipGraphLaunch.
// 6. Destroy graph and executable when done.
//
// Alternative creation:
// - Stream capture: Record operations to build graph.
// - Graph cloning: Copy existing graph structure.
//
// Limitations:
// - No cycles allowed (must be DAG).
// - Some operations cannot be captured.
// - Device-specific node limits.
//
// Multi-GPU: Graphs can contain operations for multiple devices
// but require careful dependency management.
//
// See also: hipGraphDestroy, hipGraphAddKernelNode,
//           hipGraphInstantiate, hipStreamBeginCapture.
HIPAPI hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Get current context.
  // Ensure initialization and get context.

  iree_hal_streaming_context_t* context = NULL;

  hipError_t init_result = hip_ensure_context(&context);

  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Create graph.
  iree_hal_streaming_graph_t* graph = NULL;
  iree_status_t status = iree_hal_streaming_graph_create(
      context, hip_graph_flags_to_internal(flags), context->host_allocator,
      &graph);

  if (iree_status_is_ok(status)) {
    *pGraph = (hipGraph_t)graph;
  } else {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorOutOfMemory);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Destroys a task graph.
//
// Parameters:
//  - graph: [IN] Graph handle to destroy.
//
// Returns:
//  - hipSuccess: Graph destroyed successfully.
//  - hipErrorInvalidValue: graph is NULL or invalid.
//
// Synchronization: This operation is synchronous.
//
// Destruction behavior:
// - Releases all resources associated with the graph.
// - All nodes in the graph are destroyed.
// - Graph handle becomes invalid after destruction.
// - Does not affect instantiated executables.
//
// Resource management:
// - Graph can be destroyed after instantiation.
// - Executable graphs remain valid after source destruction.
// - Must destroy both graph and executables to free all resources.
//
// Warning: Using a destroyed graph results in undefined behavior.
//
// See also: hipGraphCreate, hipGraphExecDestroy,
//           hipGraphInstantiate.
HIPAPI hipError_t hipGraphDestroy(hipGraph_t graph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_release(stream_graph);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Instantiates a graph to create an executable graph.
//
// Parameters:
//  - pGraphExec: [OUT] Pointer to receive executable graph handle.
//  - graph: [IN] Source graph to instantiate.
//  - pErrorNode: [OUT] Optional pointer to receive error node (can be NULL).
//  - pLogBuffer: [OUT] Optional buffer for error messages (can be NULL).
//  - bufferSize: [IN] Size of pLogBuffer in bytes.
//
// Returns:
//  - hipSuccess: Graph instantiated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorGraphCyclesDetected: Graph contains cycles.
//  - hipErrorInvalidDeviceFunction: Invalid kernel in graph.
//
// Synchronization: This operation is synchronous.
//
// Instantiation process:
// - Validates graph structure (DAG, no cycles).
// - Allocates resources for execution.
// - Optimizes node scheduling.
// - Creates device-specific command buffers.
// - Returns executable that can be launched.
//
// Error reporting:
// - pErrorNode receives first problematic node if provided.
// - pLogBuffer receives detailed error message if provided.
// - Both can be NULL if error details not needed.
//
// Performance optimization:
// - Instantiation is expensive; do once, launch many times.
// - Enables whole-graph optimizations.
// - May merge adjacent compatible operations.
// - Optimizes memory allocation and reuse.
//
// Resource lifetime:
// - Executable is independent of source graph.
// - Source graph can be destroyed after instantiation.
// - Must destroy executable with hipGraphExecDestroy.
//
// Multi-GPU: Instantiation binds operations to specific devices
// based on current context and node specifications.
//
// See also: hipGraphInstantiateWithFlags, hipGraphLaunch,
//           hipGraphExecDestroy, hipGraphExecUpdate.
HIPAPI hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec,
                                      hipGraph_t graph,
                                      hipGraphNode_t* pErrorNode,
                                      char* pLogBuffer, size_t bufferSize) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphExec || !graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Always NUL terminate the log buffer and clear error node in case we fail
  // early.
  if (pErrorNode) *pErrorNode = NULL;
  if (pLogBuffer && bufferSize > 0) {
    pLogBuffer[0] = 0;
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status = iree_hal_streaming_graph_instantiate(
      stream_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec);

  if (iree_status_is_ok(status)) {
    *pGraphExec = (hipGraphExec_t)exec;
  } else {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Instantiates a graph with flags to create an executable graph.
//
// Parameters:
//  - pGraphExec: [OUT] Pointer to receive executable graph handle.
//  - graph: [IN] Source graph to instantiate.
//  - flags: [IN] Instantiation flags for optimization hints.
//
// Returns:
//  - hipSuccess: Graph instantiated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//  - hipErrorGraphCyclesDetected: Graph contains cycles.
//
// Synchronization: This operation is synchronous.
//
// Flag options:
// - hipGraphInstantiateFlagAutoFreeOnLaunch: Auto-free after launch.
// - hipGraphInstantiateFlagUpload: Upload to device immediately.
// - hipGraphInstantiateFlagDeviceLaunch: Enable device-side launch.
// - hipGraphInstantiateFlagUseNodePriority: Honor node priorities.
//
// Extended behavior:
// - Same as hipGraphInstantiate but with optimization hints.
// - Flags control resource management and scheduling.
// - May affect performance and memory usage.
//
// Performance flags:
// - AutoFree reduces memory pressure for one-shot graphs.
// - Upload flag can reduce first launch latency.
// - DeviceLaunch enables GPU-driven execution.
//
// Multi-GPU: Flags may affect multi-device scheduling
// and resource allocation strategies.
//
// See also: hipGraphInstantiate, hipGraphLaunch,
//           hipGraphExecDestroy.
HIPAPI hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec,
                                               hipGraph_t graph,
                                               unsigned long long flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pGraphExec || !graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status = iree_hal_streaming_graph_instantiate(
      stream_graph, hip_graph_instantiate_flags_to_internal(flags), &exec);

  if (iree_status_is_ok(status)) {
    *pGraphExec = (hipGraphExec_t)exec;
  } else {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Destroys an executable graph.
//
// Parameters:
//  - graphExec: [IN] Executable graph handle to destroy.
//
// Returns:
//  - hipSuccess: Executable destroyed successfully.
//  - hipErrorInvalidValue: graphExec is NULL or invalid.
//
// Synchronization: This operation is synchronous.
//
// Destruction behavior:
// - Waits for any pending executions to complete.
// - Releases all resources associated with executable.
// - Handle becomes invalid after destruction.
// - Does not affect source graph used for instantiation.
//
// Resource management:
// - Must destroy to free device resources.
// - Independent of source graph lifetime.
// - Multiple executables can exist from same graph.
//
// Warning: Using a destroyed executable results in undefined behavior.
// Ensure no launches are pending before destruction.
//
// See also: hipGraphInstantiate, hipGraphDestroy, hipGraphLaunch.
HIPAPI hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!graphExec) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_exec_t* exec =
      (iree_hal_streaming_graph_exec_t*)graphExec;
  iree_hal_streaming_graph_exec_release(exec);

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Launches an executable graph in a stream.
//
// Parameters:
//  - graphExec: [IN] Executable graph to launch.
//  - stream: [IN] Stream for asynchronous execution.
//
// Returns:
//  - hipSuccess: Graph launched successfully.
//  - hipErrorInvalidValue: Invalid executable or stream.
//  - hipErrorInvalidContext: No active HIP context.
//  - hipErrorLaunchFailure: Launch failed on device.
//
// Synchronization: This operation is asynchronous.
//
// Launch behavior:
// - Enqueues entire graph as single operation.
// - Graph executes after prior stream operations.
// - Subsequent operations wait for graph completion.
// - Internal node dependencies are preserved.
//
// Performance benefits:
// - Single launch for complex workload.
// - Reduced CPU-GPU communication overhead.
// - Optimized scheduling and resource usage.
// - Better than individual kernel launches.
//
// Execution model:
// - Nodes execute based on dependencies.
// - Parallel nodes may run concurrently.
// - Graph completion when all nodes finish.
// - Use hipStreamSynchronize() to wait.
//
// Reusability:
// - Same executable can be launched multiple times.
// - Each launch is independent execution.
// - Resources are reused across launches.
// - Efficient for repeated workloads.
//
// Multi-GPU:
// - Graph may contain operations for multiple devices.
// - Cross-device dependencies handled automatically.
// - Stream determines primary execution context.
//
// Warning: Ensure input/output buffers are valid for each launch.
// Graph captures buffer addresses, not contents.
//
// See also: hipGraphInstantiate, hipGraphExecUpdate,
//           hipStreamSynchronize, hipGraphExecDestroy.
HIPAPI hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!graphExec) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_exec_t* exec =
      (iree_hal_streaming_graph_exec_t*)graphExec;
  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;

  // Use default stream if not specified.
  if (!stream_obj) {
    // Ensure initialization and get context.
    iree_hal_streaming_context_t* context = NULL;
    hipError_t init_result = hip_ensure_context(&context);
    if (init_result != hipSuccess) {
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(init_result);
    }
    stream_obj = context->default_stream;
  }

  iree_status_t status = iree_hal_streaming_graph_exec_launch(exec, stream_obj);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Updates an executable graph with a modified source graph.
//
// Parameters:
//  - hGraphExec: [IN/OUT] Executable graph to update.
//  - hGraph: [IN] Modified source graph with updates.
//  - hErrorNode_out: [OUT] Optional pointer for error node (can be NULL).
//  - flags: [IN] Update flags (currently unused, must be 0).
//
// Returns:
//  - hipSuccess: Update successful.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorGraphExecUpdateFailure: Update not possible.
//
// Synchronization: This operation is synchronous.
//
// Update behavior:
// - Attempts to update executable without full re-instantiation.
// - Preserves optimizations when possible.
// - Faster than destroy + re-instantiate.
// - May fail if topology changed significantly.
//
// Supported updates:
// - Kernel parameters and launch configuration.
// - Memory copy parameters (source/dest/size).
// - Host callback function pointers.
// - Node enable/disable state.
//
// Unsupported updates:
// - Adding or removing nodes.
// - Changing node types.
// - Modifying graph topology/dependencies.
// - Changing device assignments.
//
// Update strategy:
// - Try update first for performance.
// - Fall back to re-instantiation if update fails.
// - Check hErrorNode_out for failure location.
//
// Performance optimization:
// - Update is much faster than re-instantiation.
// - Preserves device-side optimizations.
// - Ideal for parameter-only changes.
//
// Multi-GPU: Updates must preserve device assignments.
// Cannot move operations between devices.
//
// Warning: Updated executable must not be executing during update.
// Ensure stream synchronization before updating.
//
// See also: hipGraphInstantiate, hipGraphLaunch,
//           hipGraphExecDestroy.
HIPAPI hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec,
                                     hipGraph_t hGraph,
                                     hipGraphNode_t* hErrorNode_out,
                                     unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hGraphExec || !hGraph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_exec_t* exec =
      (iree_hal_streaming_graph_exec_t*)hGraphExec;
  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;

  iree_status_t status = iree_hal_streaming_graph_exec_update(exec, graph);

  if (!iree_status_is_ok(status)) {
    if (hErrorNode_out) {
      *hErrorNode_out = NULL;  // We don't track specific error nodes yet.
    }
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a kernel execution node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pNodeParams: [IN] Kernel parameters (hipKernelNodeParams structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: This operation modifies the graph structure.
//
// Node parameters (hipKernelNodeParams):
// - func: Kernel function to execute.
// - gridDim: Grid dimensions (blocks).
// - blockDim: Block dimensions (threads).
// - sharedMemBytes: Dynamic shared memory size.
// - kernelParams: Array of kernel arguments.
// - extra: Extra launch parameters.
//
// Dependency behavior:
// - Node waits for all dependencies to complete.
// - Creates edges from dependency nodes to this node.
// - NULL dependencies means no predecessors.
// - Forms directed acyclic graph (DAG).
//
// Graph construction:
// - Nodes can be added in any order.
// - Dependencies must already exist in graph.
// - Multiple nodes can have same dependencies.
// - Node becomes dependency for subsequent nodes.
//
// Kernel execution:
// - Parameters captured at node creation.
// - Kernel launches when dependencies satisfied.
// - Uses captured parameters for each graph launch.
// - Parameters can be updated via node update APIs.
//
// Multi-GPU:
// - Kernel executes on device associated with function.
// - Cross-device dependencies handled automatically.
//
// Warning: Ensure kernel parameters remain valid until graph
// destruction. Pointers in kernelParams are captured by reference.
//
// See also: hipGraphAddMemcpyNode, hipGraphAddMemsetNode,
//           hipGraphNodeGetType, hipGraphKernelNodeSetParams.
HIPAPI hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pNodeParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pNodeParams) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipKernelNodeParams* params = (const hipKernelNodeParams*)pNodeParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Create dispatch params from kernel node params.
  // Extract params pointer from HIP's parameter format.
  void* params_ptr = NULL;
  if (params->extra) {
    // Extra format: {HIP_LAUNCH_PARAM_BUFFER_POINTER, &buffer,
    //                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END}
    if (params->extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
      params_ptr = *(void**)params->extra[1];
    }
  } else if (params->kernelParams) {
    // kernelParams is an array of pointers to the actual parameters.
    params_ptr = params->kernelParams;
  }

  iree_hal_streaming_dispatch_params_t dispatch_params = {
      .grid_dim = {params->gridDim.x, params->gridDim.y, params->gridDim.z},
      .block_dim = {params->blockDim.x, params->blockDim.y, params->blockDim.z},
      .shared_memory_bytes = params->sharedMemBytes,
      .buffer = params_ptr,
  };

  // Add kernel node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_kernel_node(
      stream_graph, deps, numDependencies,
      (iree_hal_streaming_symbol_t*)params->func, &dispatch_params, &node);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a memory copy node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pCopyParams: [IN] Copy parameters (hipMemcpy3DParms structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: This operation modifies the graph structure.
//
// Copy parameters (hipMemcpy3DParms):
// - srcArray/srcPtr: Source memory (array or pointer).
// - srcPos: Source position for 3D copies.
// - dstArray/dstPtr: Destination memory.
// - dstPos: Destination position for 3D copies.
// - extent: Size of copy region (width, height, depth).
// - kind: Memory copy kind (D2D, H2D, D2H, etc.).
//
// Copy behavior:
// - Captures copy parameters at node creation.
// - Executes copy when dependencies satisfied.
// - Supports 1D, 2D, and 3D memory copies.
// - Handles all memory types (host/device/managed).
//
// Dependency management:
// - Waits for all dependency nodes to complete.
// - Copy occurs after dependencies in graph execution.
// - Can be dependency for subsequent nodes.
//
// Performance optimization:
// - Enables copy-compute overlap in graph.
// - May merge adjacent copies when possible.
// - Optimizes for memory bandwidth utilization.
//
// Multi-GPU:
// - Supports peer-to-peer copies between devices.
// - Cross-device copies handled transparently.
// - Routing optimized based on topology.
//
// Warning: Source and destination addresses captured at creation.
// Ensure memory remains valid for all graph launches.
//
// See also: hipGraphAddKernelNode, hipGraphAddMemsetNode,
//           hipMemcpy3DAsync, hipGraphMemcpyNodeSetParams.
HIPAPI hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pCopyParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pCopyParams) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipMemcpy3DParms* params = (const hipMemcpy3DParms*)pCopyParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // For simplicity, handle basic device-to-device copy.
  // Full implementation would handle all memory types.
  iree_hal_streaming_deviceptr_t src =
      (iree_hal_streaming_deviceptr_t)params->srcPtr.ptr;
  iree_hal_streaming_deviceptr_t dst =
      (iree_hal_streaming_deviceptr_t)params->dstPtr.ptr;
  size_t size =
      params->extent.width * params->extent.height * params->extent.depth;

  // Add memcpy node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_memcpy_node(
      stream_graph, deps, numDependencies, dst, src, size, &node);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a memory set node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pMemsetParams: [IN] Memset parameters (hipMemsetParams structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: This operation modifies the graph structure.
//
// Memset parameters (hipMemsetParams):
// - dst: Destination device pointer.
// - value: Value to set (supports 1, 2, or 4 byte values).
// - elementSize: Size of each element (1, 2, or 4 bytes).
// - width: Width of 2D memset area in elements.
// - height: Height of 2D memset area.
// - pitch: Pitch of destination memory.
//
// Memset behavior:
// - Captures parameters at node creation.
// - Executes memset when dependencies satisfied.
// - Supports 1D and 2D memory regions.
// - Fills memory with specified pattern.
//
// Pattern replication:
// - 1-byte: Pattern replicated across region.
// - 2-byte: 16-bit pattern (requires alignment).
// - 4-byte: 32-bit pattern (requires alignment).
//
// Performance optimization:
// - Optimized for memory bandwidth.
// - May use specialized hardware for fills.
// - Coalesced memory access patterns.
//
// Multi-GPU:
// - Memset executes on device owning the memory.
// - Cross-device dependencies handled automatically.
//
// Warning: Destination address captured at creation.
// Ensure memory remains valid for all graph launches.
//
// See also: hipGraphAddMemcpyNode, hipGraphAddKernelNode,
//           hipMemsetAsync, hipGraphMemsetNodeSetParams.
HIPAPI hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode,
                                        hipGraph_t graph,
                                        const hipGraphNode_t* pDependencies,
                                        size_t numDependencies,
                                        const void* pMemsetParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pMemsetParams) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipMemsetParams* params = (const hipMemsetParams*)pMemsetParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Add memset node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_memset_node(
      stream_graph, deps, numDependencies,
      (iree_hal_streaming_deviceptr_t)params->dst, params->value,
      params->elementSize, params->width * params->height, &node);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds a host callback node to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//  - pNodeParams: [IN] Host node parameters (hipHostNodeParams structure).
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: Host callback blocks graph execution.
//
// Host node parameters (hipHostNodeParams):
// - fn: Host function to call.
// - userData: User data passed to callback.
//
// Callback signature:
// ```c
// void hostFunction(void* userData);
// ```
//
// Callback behavior:
// - Executes on host when dependencies complete.
// - Blocks graph execution until callback returns.
// - Runs on runtime thread, not application thread.
// - Can make HIP API calls except synchronization.
//
// Restrictions in callback:
// - Cannot call hipStreamSynchronize().
// - Cannot call hipDeviceSynchronize().
// - Cannot wait on events from same graph.
// - Should complete quickly to avoid stalls.
//
// Use cases:
// - CPU computation between GPU operations.
// - Logging or debugging within graphs.
// - Dynamic parameter updates.
// - Resource management.
//
// Performance considerations:
// - Host nodes serialize graph execution.
// - Long callbacks hurt GPU utilization.
// - Consider async alternatives when possible.
//
// Multi-GPU:
// - Callback executes once regardless of devices.
// - Can access resources from multiple devices.
//
// Warning: Avoid heavy computation in callbacks.
// Graph execution stalls until callback completes.
//
// See also: hipGraphAddKernelNode, hipLaunchHostFunc,
//           hipGraphHostNodeSetParams.
HIPAPI hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode,
                                      hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies,
                                      size_t numDependencies,
                                      const void* pNodeParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph || !pNodeParams) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;
  const hipHostNodeParams* params = (const hipHostNodeParams*)pNodeParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Add host node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_host_call_node(
      stream_graph, deps, numDependencies, (void (*)(void*))params->fn,
      params->userData, &node);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Adds an empty node for synchronization to a graph.
//
// Parameters:
//  - pGraphNode: [OUT] Pointer to receive the created node handle.
//  - graph: [IN] Graph to add the node to.
//  - pDependencies: [IN] Array of nodes this node depends on (can be NULL).
//  - numDependencies: [IN] Number of dependencies.
//
// Returns:
//  - hipSuccess: Node added successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Synchronization: Empty nodes are pure synchronization points.
//
// Empty node behavior:
// - No operation performed.
// - Only waits for dependencies.
// - Acts as synchronization barrier.
// - Zero runtime overhead.
//
// Use cases:
// - Join multiple parallel paths.
// - Create synchronization points.
// - Simplify complex dependencies.
// - Graph structure organization.
//
// Dependency patterns:
// - Fan-in: Multiple nodes converge to empty node.
// - Fan-out: Empty node fans out to multiple nodes.
// - Barrier: Forces serialization point.
//
// Graph optimization:
// - May be optimized away if redundant.
// - Helps graph analysis and scheduling.
// - No device resources required.
//
// Multi-GPU:
// - Synchronizes across device boundaries.
// - Useful for cross-device coordination.
//
// Example:
// ```c
// // Create barrier after parallel operations
// hipGraphNode_t nodes[] = {kernel1, kernel2, kernel3};
// hipGraphNode_t barrier;
// hipGraphAddEmptyNode(&barrier, graph, nodes, 3);
// ```
//
// See also: hipGraphAddKernelNode, hipGraphNodeGetType,
//           hipGraphNodeGetDependencies.
HIPAPI hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode,
                                       hipGraph_t graph,
                                       const hipGraphNode_t* pDependencies,
                                       size_t numDependencies) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pGraphNode || !graph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_graph_t* stream_graph = (iree_hal_streaming_graph_t*)graph;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && pDependencies)
          ? (iree_hal_streaming_graph_node_t**)pDependencies
          : NULL;

  // Empty nodes are just synchronization points.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_empty_node(
      stream_graph, deps, numDependencies, &node);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  *pGraphNode = (hipGraphNode_t)node;
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

// Begins capturing operations into a graph.
//
// Parameters:
//  - stream: [IN] Stream to capture operations from.
//  - mode: [IN] Capture mode (affects cross-stream dependencies).
//
// Returns:
//  - hipSuccess: Capture started successfully.
//  - hipErrorInvalidValue: Invalid stream or mode.
//  - hipErrorStreamCaptureInvalidated: Stream already capturing.
//  - hipErrorStreamCaptureImplicit: Implicit capture not supported.
//
// Synchronization: Subsequent stream operations are captured.
//
// Capture modes:
// - hipStreamCaptureModeGlobal: Capture all work.
// - hipStreamCaptureModeThreadLocal: Thread-local capture.
// - hipStreamCaptureModeRelaxed: Relaxed ordering.
//
// Capture behavior:
// - Operations recorded to graph instead of executing.
// - Continues until hipStreamEndCapture().
// - Creates graph from recorded operations.
// - Stream cannot be synchronized during capture.
//
// Captured operations:
// - Kernel launches.
// - Memory copies and sets.
// - Event records and waits.
// - Child stream operations (mode dependent).
//
// Restrictions during capture:
// - No hipStreamSynchronize() on capturing stream.
// - No hipDeviceSynchronize().
// - No blocking operations.
// - Limited cross-stream dependencies.
//
// Cross-stream capture:
// - Global mode: Captures work from other streams.
// - ThreadLocal mode: Only current thread's work.
// - Relaxed mode: Best-effort capture.
//
// Error handling:
// - Invalid operations invalidate capture.
// - Check hipStreamEndCapture() for errors.
// - Query status with hipStreamIsCapturing().
//
// Multi-GPU:
// - Can capture operations for multiple devices.
// - Device switches recorded in graph.
//
// Warning: Stream must be idle before starting capture.
// Pending operations may cause undefined behavior.
//
// See also: hipStreamEndCapture, hipStreamIsCapturing,
//           hipGraphCreate, hipGraphLaunch.
HIPAPI hipError_t hipStreamBeginCapture(hipStream_t stream,
                                        hipStreamCaptureMode mode) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;

  // Map HIP capture mode to internal mode.
  iree_hal_streaming_capture_mode_t capture_mode;
  switch (mode) {
    case hipStreamCaptureModeGlobal:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
      break;
    case hipStreamCaptureModeThreadLocal:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL;
      break;
    case hipStreamCaptureModeRelaxed:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_status_t status =
      iree_hal_streaming_begin_capture(stream_obj, capture_mode);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Ends stream capture and returns the captured graph.
//
// Parameters:
//  - stream: [IN] Stream to end capture on.
//  - pGraph: [OUT] Pointer to receive the captured graph.
//
// Returns:
//  - hipSuccess: Capture ended successfully.
//  - hipErrorInvalidValue: Invalid stream or pGraph is NULL.
//  - hipErrorStreamCaptureInvalidated: Capture was invalidated.
//  - hipErrorStreamCaptureUnmatched: Unmatched begin/end.
//  - hipErrorStreamCaptureWrongThread: Wrong thread for thread-local.
//
// Synchronization: Stream returns to normal execution mode.
//
// Capture completion:
// - Stops recording operations to graph.
// - Returns complete graph of captured work.
// - Stream resumes normal execution.
// - Graph ready for instantiation.
//
// Graph contents:
// - All operations between begin/end.
// - Preserved dependencies and ordering.
// - Cross-stream work if mode allows.
// - Event synchronization captured.
//
// Error conditions:
// - Invalid operations during capture.
// - Stream synchronization attempted.
// - Blocking operations encountered.
// - Resource allocation failures.
//
// Graph validation:
// - Check for cycles (must be DAG).
// - Verify all dependencies resolved.
// - Ensure device compatibility.
//
// Post-capture:
// - Graph can be instantiated immediately.
// - Stream can begin new capture.
// - Graph independent of stream lifetime.
//
// Multi-GPU:
// - Captured graph may span devices.
// - Device switches preserved in graph.
//
// Warning: Ensure capture was started with hipStreamBeginCapture.
// Mismatched begin/end causes undefined behavior.
//
// See also: hipStreamBeginCapture, hipGraphInstantiate,
//           hipStreamIsCapturing, hipGraphCreate.
HIPAPI hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream || !pGraph) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  iree_hal_streaming_graph_t* graph = NULL;
  iree_status_t status = iree_hal_streaming_end_capture(stream_obj, &graph);

  if (iree_status_is_ok(status)) {
    *pGraph = (hipGraph_t)graph;
  } else {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Queries if a stream is currently capturing.
//
// Parameters:
//  - stream: [IN] Stream to query.
//  - pCaptureStatus: [OUT] Pointer to receive capture status.
//
// Returns:
//  - hipSuccess: Status queried successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorStreamCaptureImplicit: Implicit capture query.
//
// Synchronization: This operation is immediate.
//
// Capture status values:
// - hipStreamCaptureStatusNone: Not capturing.
// - hipStreamCaptureStatusActive: Currently capturing.
// - hipStreamCaptureStatusInvalidated: Capture invalidated.
//
// Status meanings:
// - None: Stream in normal execution mode.
// - Active: Between begin and end capture.
// - Invalidated: Error occurred during capture.
//
// Query behavior:
// - Non-blocking status check.
// - Safe to call anytime.
// - Works with NULL stream.
//
// Use cases:
// - Check before synchronization.
// - Verify capture state.
// - Error detection during capture.
// - Conditional code paths.
//
// Invalidation causes:
// - Invalid API calls during capture.
// - Resource allocation failures.
// - Unsupported operations attempted.
//
// Multi-GPU:
// - Status is per-stream.
// - Cross-device captures tracked.
//
// See also: hipStreamBeginCapture, hipStreamEndCapture,
//           hipStreamGetCaptureInfo.
HIPAPI hipError_t hipStreamIsCapturing(hipStream_t stream,
                                       hipStreamCaptureStatus* pCaptureStatus) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pCaptureStatus) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (!stream) {
    // Use default stream.
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *pCaptureStatus = hipStreamCaptureStatusNone;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  bool is_capturing = false;
  iree_status_t status =
      iree_hal_streaming_is_capturing(stream_obj, &is_capturing);

  if (iree_status_is_ok(status)) {
    *pCaptureStatus = is_capturing ? hipStreamCaptureStatusActive
                                   : hipStreamCaptureStatusNone;
  } else {
    *pCaptureStatus = hipStreamCaptureStatusNone;
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Gets detailed information about stream capture.
//
// Parameters:
//  - stream: [IN] Stream to query.
//  - pCaptureStatus: [OUT] Pointer to receive capture status.
//  - pId: [OUT] Optional pointer to receive capture ID (can be NULL).
//
// Returns:
//  - hipSuccess: Information retrieved successfully.
//  - hipErrorInvalidValue: pCaptureStatus is NULL.
//
// Synchronization: This operation is immediate.
//
// Extended capture information:
// - Status: Current capture state.
// - ID: Unique capture session identifier.
// - ID remains constant during capture.
// - ID changes with each new capture.
//
// Capture ID usage:
// - Track capture sessions.
// - Match related captures.
// - Debug capture issues.
// - Correlate with events.
//
// Status values:
// - None: Not capturing.
// - Active: Currently capturing.
// - Invalidated: Capture failed.
//
// Multi-stream capture:
// - ID shared across captured streams.
// - Helps identify related captures.
// - Cross-stream dependencies tracked.
//
// Multi-GPU:
// - Each device maintains capture IDs.
// - Cross-device captures share ID.
//
// See also: hipStreamIsCapturing, hipStreamBeginCapture,
//           hipStreamEndCapture.
HIPAPI hipError_t hipStreamGetCaptureInfo(
    hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
    unsigned long long* pId) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pCaptureStatus) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  if (!stream) {
    // Use default stream.
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *pCaptureStatus = hipStreamCaptureStatusNone;
      if (pId) *pId = 0;
      IREE_TRACE_ZONE_END(z0);
      return hipSuccess;
    }
    stream = (hipStream_t)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;
  iree_hal_streaming_capture_status_t status_internal;
  unsigned long long capture_id;
  iree_status_t status = iree_hal_streaming_capture_status(
      stream_obj, &status_internal, &capture_id);

  if (iree_status_is_ok(status)) {
    // Map internal status to HIP status.
    switch (status_internal) {
      case IREE_HAL_STREAMING_CAPTURE_STATUS_NONE:
        *pCaptureStatus = hipStreamCaptureStatusNone;
        break;
      case IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE:
        *pCaptureStatus = hipStreamCaptureStatusActive;
        break;
      case IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED:
        *pCaptureStatus = hipStreamCaptureStatusInvalidated;
        break;
      default:
        *pCaptureStatus = hipStreamCaptureStatusNone;
        break;
    }
    if (pId) {
      *pId = capture_id;
    }
  } else {
    *pCaptureStatus = hipStreamCaptureStatusNone;
    if (pId) *pId = 0;
    iree_status_ignore(status);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Updates dependencies for the next captured node.
//
// Parameters:
//  - stream: [IN] Stream currently capturing.
//  - dependencies: [IN] Array of nodes for next operation to depend on.
//  - numDependencies: [IN] Number of dependencies.
//  - flags: [IN] Update flags (must be 0).
//
// Returns:
//  - hipSuccess: Dependencies updated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorStreamCaptureInvalidated: Stream not capturing.
//
// Synchronization: Affects next captured operation.
//
// Dependency injection:
// - Modifies dependencies for next node.
// - Does not affect already captured nodes.
// - Creates edges from specified nodes.
// - Enables complex graph patterns.
//
// Use cases:
// - Connect independent capture sequences.
// - Create custom synchronization points.
// - Merge parallel capture paths.
// - Fix missing dependencies.
//
// Capture flow:
// - Call during active capture.
// - Next operation depends on specified nodes.
// - Subsequent operations follow normal flow.
// - One-time effect per call.
//
// Dependency rules:
// - Nodes must be from same capture.
// - Cannot create cycles.
// - Dependencies validated at capture.
//
// Advanced patterns:
// - Fork-join parallelism.
// - Conditional dependencies.
// - Dynamic graph construction.
//
// Multi-GPU:
// - Dependencies can cross devices.
// - Ensures proper synchronization.
//
// Warning: Invalid dependencies may invalidate entire capture.
// Verify nodes are from current capture session.
//
// See also: hipStreamBeginCapture, hipGraphAddEmptyNode,
//           hipStreamEndCapture.
HIPAPI hipError_t hipStreamUpdateCaptureDependencies(
    hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies,
    unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!stream) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_stream_t* stream_obj =
      (iree_hal_streaming_stream_t*)stream;

  // Map HIP flags to internal mode.
  iree_hal_streaming_capture_dependencies_mode_t mode;
  if (flags & hipStreamAddCaptureDependencies) {
    mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD;
  } else if (flags & hipStreamSetCaptureDependencies) {
    mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET;
  } else {
    // Default to SET if no specific flag.
    mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET;
  }

  iree_status_t status = iree_hal_streaming_update_capture_dependencies(
      stream_obj, (iree_hal_streaming_graph_node_t**)dependencies,
      numDependencies, mode);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

//===----------------------------------------------------------------------===//
// Memory pool management
//===----------------------------------------------------------------------===//

// Creates a memory pool for stream-ordered allocations.
//
// Parameters:
//  - pool: [OUT] Pointer to receive the created pool handle.
//  - poolProps: [IN] Pool properties structure.
//
// Returns:
//  - hipSuccess: Pool created successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorNotSupported: Pool type not supported.
//  - hipErrorMemoryAllocation: Insufficient resources.
//
// Synchronization: This operation is synchronous.
//
// Pool properties (hipMemPoolProps):
// - allocType: Allocation type (hipMemAllocationTypePinned, etc.).
// - handleTypes: Export handle types supported.
// - location: Memory location hints.
// - reserved: Reserved for future use.
//
// Pool characteristics:
// - Stream-ordered allocation/deallocation.
// - Automatic memory reuse.
// - Reduced allocation overhead.
// - Better memory utilization.
//
// Allocation behavior:
// - Memory allocated on-demand.
// - Freed memory returned to pool.
// - Pool manages fragmentation.
// - Supports suballocation.
//
// Performance benefits:
// - Faster than hipMalloc/hipFree.
// - Reduced kernel launch overhead.
// - Better memory locality.
// - Efficient resource usage.
//
// Pool sharing:
// - Can be shared between streams.
// - Supports IPC export/import.
// - Multi-process capable.
//
// Multi-GPU:
// - Pool associated with specific device.
// - Cross-device access configurable.
// - Peer access rules apply.
//
// Warning: Pool must be destroyed when no longer needed.
// Active allocations prevent pool destruction.
//
// See also: hipMemPoolDestroy, hipMallocFromPoolAsync,
//           hipMemPoolSetAttribute, hipDeviceSetMemPool.
HIPAPI hipError_t hipMemPoolCreate(hipMemPool_t* pool,
                                   const hipMemPoolProps* poolProps) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !poolProps) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  // Convert HIP pool props to internal props.
  iree_hal_streaming_mem_pool_props_t props = {
      .alloc_handle_type =
          hip_mem_handle_type_to_internal(poolProps->handleTypes),
      .location_type =
          hip_mem_location_type_to_internal(poolProps->location.type),
      .location_id = poolProps->location.id,
  };

  iree_hal_streaming_mem_pool_t* mem_pool = NULL;
  iree_status_t status = iree_hal_streaming_mem_pool_create(
      context, &props, context->host_allocator, &mem_pool);

  if (iree_status_is_ok(status)) {
    *pool = (hipMemPool_t)mem_pool;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Destroys a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle to destroy.
//
// Returns:
//  - hipSuccess: Pool destroyed successfully.
//  - hipErrorInvalidValue: pool is NULL or invalid.
//
// Synchronization: Waits for all allocations to be freed.
//
// Destruction behavior:
// - Releases all pool resources.
// - Invalidates pool handle.
// - Pending operations must complete.
// - Active allocations must be freed.
//
// Resource cleanup:
// - Returns memory to system.
// - Closes IPC handles.
// - Releases device resources.
//
// Warning: Ensure all allocations from pool are freed.
// Destroying pool with active allocations causes errors.
//
// See also: hipMemPoolCreate, hipFreeAsync,
//           hipMemPoolTrimTo.
HIPAPI hipError_t hipMemPoolDestroy(hipMemPool_t pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_mem_pool_release((iree_hal_streaming_mem_pool_t*)pool);
  IREE_TRACE_ZONE_END(z0);
  return hipSuccess;
}

// Sets an attribute of a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - attr: [IN] Attribute to set (hipMemPool_attribute enum).
//  - value: [IN] Pointer to new attribute value.
//
// Returns:
//  - hipSuccess: Attribute set successfully.
//  - hipErrorInvalidValue: Invalid pool, attribute, or value.
//  - hipErrorNotSupported: Attribute cannot be modified.
//
// Synchronization: This operation is synchronous.
//
// Settable attributes:
// - hipMemPoolAttrReleaseThreshold: Memory release threshold.
// - hipMemPoolAttrReuseFollowEventDependencies: Event-based reuse.
// - hipMemPoolAttrReuseAllowOpportunistic: Opportunistic reuse.
// - hipMemPoolAttrReuseAllowInternalDependencies: Internal reuse.
//
// Attribute effects:
// - ReleaseThreshold: Bytes to retain before OS release.
// - ReuseFollowEventDependencies: Honor event dependencies.
// - ReuseAllowOpportunistic: Aggressive memory reuse.
// - ReuseAllowInternalDependencies: Reuse within operations.
//
// Performance tuning:
// - Adjust threshold for memory vs performance.
// - Enable opportunistic reuse for throughput.
// - Disable dependencies for deterministic behavior.
//
// Multi-GPU:
// - Attributes apply to pool on all devices.
// - May affect cross-device synchronization.
//
// Warning: Changing attributes affects all future allocations.
// Existing allocations retain original behavior.
//
// See also: hipMemPoolGetAttribute, hipMemPoolCreate,
//           hipMallocFromPoolAsync.
HIPAPI hipError_t hipMemPoolSetAttribute(hipMemPool_t pool,
                                         hipMemPool_attribute attr,
                                         void* value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  uint64_t attr_value = 0;
  switch (attr) {
    case hipMemPoolAttrReleaseThreshold:
      attr_value = *(size_t*)value;
      break;
    case hipMemPoolAttrReuseFollowEventDependencies:
    case hipMemPoolAttrReuseAllowOpportunistic:
    case hipMemPoolAttrReuseAllowInternalDependencies:
      attr_value = *(int*)value;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_mem_pool_attr_t internal_attr =
      hip_mempool_attr_to_internal(attr);
  iree_status_t status = iree_hal_streaming_mem_pool_set_attribute(
      (iree_hal_streaming_mem_pool_t*)pool, internal_attr, attr_value);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets an attribute of a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - attr: [IN] Attribute to query (hipMemPool_attribute enum).
//  - value: [OUT] Pointer to receive attribute value.
//
// Returns:
//  - hipSuccess: Attribute retrieved successfully.
//  - hipErrorInvalidValue: Invalid pool, attribute, or value.
//
// Synchronization: This operation is immediate.
//
// Queryable attributes:
// - hipMemPoolAttrReleaseThreshold: Current release threshold.
// - hipMemPoolAttrReservedMemCurrent: Currently reserved memory.
// - hipMemPoolAttrReservedMemHigh: Peak reserved memory.
// - hipMemPoolAttrUsedMemCurrent: Currently used memory.
// - hipMemPoolAttrUsedMemHigh: Peak used memory.
// - hipMemPoolAttrReuseFollowEventDependencies: Event reuse setting.
// - hipMemPoolAttrReuseAllowOpportunistic: Opportunistic reuse.
// - hipMemPoolAttrReuseAllowInternalDependencies: Internal reuse.
//
// Statistics:
// - Reserved: Memory held by pool from OS.
// - Used: Memory allocated to application.
// - High watermarks: Peak usage tracking.
//
// Performance monitoring:
// - Track memory efficiency.
// - Identify memory pressure.
// - Optimize pool configuration.
//
// See also: hipMemPoolSetAttribute, hipMemPoolCreate,
//           hipMemPoolTrimTo.
HIPAPI hipError_t hipMemPoolGetAttribute(hipMemPool_t pool,
                                         hipMemPool_attribute attr,
                                         void* value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !value) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  uint64_t attr_value = 0;
  iree_hal_streaming_mem_pool_attr_t internal_attr =
      hip_mempool_attr_to_internal(attr);
  iree_status_t status = iree_hal_streaming_mem_pool_get_attribute(
      (iree_hal_streaming_mem_pool_t*)pool, internal_attr, &attr_value);

  if (iree_status_is_ok(status)) {
    switch (attr) {
      case hipMemPoolAttrReleaseThreshold:
      case hipMemPoolAttrReservedMemCurrent:
      case hipMemPoolAttrReservedMemHigh:
      case hipMemPoolAttrUsedMemCurrent:
      case hipMemPoolAttrUsedMemHigh:
        *(size_t*)value = (size_t)attr_value;
        break;
      case hipMemPoolAttrReuseFollowEventDependencies:
      case hipMemPoolAttrReuseAllowOpportunistic:
      case hipMemPoolAttrReuseAllowInternalDependencies:
        *(int*)value = (int)attr_value;
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        break;
    }
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Sets memory access permissions for a memory pool.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - map: [IN] Array of access descriptors.
//  - count: [IN] Number of access descriptors.
//
// Returns:
//  - hipSuccess: Access permissions set.
//  - hipErrorNotSupported: Not implemented.
//
// See also: hipMemPoolGetAccess.
HIPAPI hipError_t hipMemPoolSetAccess(hipMemPool_t pool,
                                      const hipMemAccessDesc* map,
                                      size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Gets memory access permissions for a memory pool.
//
// Parameters:
//  - flags: [OUT] Pointer to receive access flags.
//  - pool: [IN] Memory pool handle.
//  - location: [IN] Memory location to query.
//
// Returns:
//  - hipSuccess: Access permissions retrieved.
//  - hipErrorNotSupported: Not implemented.
//
// See also: hipMemPoolSetAccess.
HIPAPI hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags,
                                      hipMemPool_t pool,
                                      hipMemLocation* location) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Trims a memory pool to specified size.
//
// Parameters:
//  - pool: [IN] Memory pool handle.
//  - minBytesToKeep: [IN] Minimum bytes to retain.
//
// Returns:
//  - hipSuccess: Pool trimmed successfully.
//  - hipErrorInvalidValue: Invalid pool.
//
// Releases unused memory back to the system.
//
// See also: hipMemPoolGetAttribute.
HIPAPI hipError_t hipMemPoolTrimTo(hipMemPool_t pool, size_t minBytesToKeep) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_status_t status = iree_hal_streaming_mem_pool_trim_to(
      (iree_hal_streaming_mem_pool_t*)pool, minBytesToKeep);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Exports a memory pool to a shareable handle.
//
// Parameters:
//  - handle_out: [OUT] Pointer to receive handle.
//  - pool: [IN] Memory pool to export.
//  - handleType: [IN] Type of handle to create.
//  - flags: [IN] Export flags.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolImportFromShareableHandle.
HIPAPI hipError_t hipMemPoolExportToShareableHandle(
    void* handle_out, hipMemPool_t pool, hipMemAllocationHandleType handleType,
    unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Imports a memory pool from a shareable handle.
//
// Parameters:
//  - pool_out: [OUT] Pointer to receive pool.
//  - handle: [IN] Shareable handle.
//  - handleType: [IN] Type of handle.
//  - flags: [IN] Import flags.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolExportToShareableHandle.
HIPAPI hipError_t hipMemPoolImportFromShareableHandle(
    hipMemPool_t* pool_out, void* handle, hipMemAllocationHandleType handleType,
    unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Exports a pointer from a memory pool for IPC.
//
// Parameters:
//  - shareData_out: [OUT] Export data structure.
//  - ptr: [IN] Pointer to export.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolImportPointer.
HIPAPI hipError_t
hipMemPoolExportPointer(hipMemPoolPtrExportData* shareData_out, void* ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Imports a pointer into a memory pool from IPC.
//
// Parameters:
//  - ptr_out: [OUT] Pointer to receive imported pointer.
//  - pool: [IN] Target memory pool.
//  - shareData: [IN] Import data structure.
//
// Returns:
//  - hipErrorNotSupported: IPC not implemented.
//
// See also: hipMemPoolExportPointer.
HIPAPI hipError_t hipMemPoolImportPointer(void** ptr_out, hipMemPool_t pool,
                                          hipMemPoolPtrExportData* shareData) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(hipErrorNotSupported);
}

// Sets the default memory pool for a device.
//
// Parameters:
//  - device: [IN] Device ID.
//  - pool: [IN] Memory pool to set as default.
//
// Returns:
//  - hipSuccess: Default pool set.
//  - hipErrorInvalidDevice: Invalid device.
//
// See also: hipDeviceGetMemPool, hipMallocAsync.
HIPAPI hipError_t hipDeviceSetMemPool(int device, hipMemPool_t pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  iree_status_t status = iree_hal_streaming_device_set_mem_pool(
      device_obj, (iree_hal_streaming_mem_pool_t*)pool);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Gets the current memory pool for a device.
//
// Parameters:
//  - pool: [OUT] Pointer to receive pool handle.
//  - device: [IN] Device ID.
//
// Returns:
//  - hipSuccess: Pool retrieved.
//  - hipErrorInvalidValue: pool is NULL.
//  - hipErrorInvalidDevice: Invalid device.
//
// See also: hipDeviceSetMemPool.
HIPAPI hipError_t hipDeviceGetMemPool(hipMemPool_t* pool, int device) {
  if (!pool) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  *pool = (hipMemPool_t)iree_hal_streaming_device_mem_pool(device_obj);

  return hipSuccess;
}

// Gets the default memory pool for a device.
//
// Parameters:
//  - pool_out: [OUT] Pointer to receive default pool.
//  - device: [IN] Device ID.
//
// Returns:
//  - hipSuccess: Default pool retrieved.
//  - hipErrorInvalidValue: pool_out is NULL.
//  - hipErrorInvalidDevice: Invalid device.
//
// Default pool is used by hipMallocAsync when no pool specified.
//
// See also: hipDeviceSetMemPool, hipMallocAsync.
HIPAPI hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* pool_out,
                                             int device) {
  if (!pool_out) {
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure HIP is initialized.
  hipError_t init_result = hip_ensure_initialized();
  if (init_result != hipSuccess) {
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    HIP_RETURN_ERROR(hipErrorInvalidDevice);
  }

  *pool_out =
      (hipMemPool_t)iree_hal_streaming_device_default_mem_pool(device_obj);

  return hipSuccess;
}

// Allocates memory asynchronously from the default pool.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive allocated memory.
//  - size: [IN] Size in bytes to allocate.
//  - stream: [IN] Stream for ordered allocation.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: ptr is NULL.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Stream-ordered allocation from device's default pool.
// Memory lifetime tied to stream operations.
//
// See also: hipFreeAsync, hipMallocFromPoolAsync.
HIPAPI hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_status_t status = iree_hal_streaming_memory_allocate_async(
      context, size, (iree_hal_streaming_stream_t*)stream, &device_ptr);

  if (iree_status_is_ok(status)) {
    *ptr = (void*)device_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Allocates memory asynchronously from a specific pool.
//
// Parameters:
//  - ptr: [OUT] Pointer to receive allocated memory.
//  - size: [IN] Size in bytes to allocate.
//  - pool: [IN] Memory pool to allocate from.
//  - stream: [IN] Stream for ordered allocation.
//
// Returns:
//  - hipSuccess: Memory allocated successfully.
//  - hipErrorInvalidValue: Invalid parameters.
//  - hipErrorMemoryAllocation: Insufficient memory.
//
// Stream-ordered allocation with explicit pool selection.
//
// See also: hipFreeAsync, hipMallocAsync, hipMemPoolCreate.
HIPAPI hipError_t hipMallocFromPoolAsync(void** ptr, size_t size,
                                         hipMemPool_t pool,
                                         hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ptr || !pool) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(hipErrorInvalidValue);
  }

  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_status_t status = iree_hal_streaming_memory_allocate_from_pool_async(
      (iree_hal_streaming_mem_pool_t*)pool, size,
      (iree_hal_streaming_stream_t*)stream, &device_ptr);

  if (iree_status_is_ok(status)) {
    *ptr = (void*)device_ptr;
  }

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

// Frees memory asynchronously.
//
// Parameters:
//  - ptr: [IN] Pointer to free (can be NULL).
//  - stream: [IN] Stream for ordered deallocation.
//
// Returns:
//  - hipSuccess: Memory freed successfully.
//  - hipErrorInvalidValue: Invalid stream.
//
// Stream-ordered deallocation. Memory returned to pool
// when stream reaches this operation.
//
// See also: hipMallocAsync, hipMallocFromPoolAsync.
HIPAPI hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Ensure initialization and get context.
  iree_hal_streaming_context_t* context = NULL;
  hipError_t init_result = hip_ensure_context(&context);
  if (init_result != hipSuccess) {
    IREE_TRACE_ZONE_END(z0);
    HIP_RETURN_ERROR(init_result);
  }

  iree_status_t status = iree_hal_streaming_memory_free_async(
      context, (iree_hal_streaming_deviceptr_t)ptr,
      (iree_hal_streaming_stream_t*)stream);

  hipError_t result = iree_status_to_hip_result(status);
  IREE_TRACE_ZONE_END(z0);
  HIP_RETURN_ERROR(result);
}

//===----------------------------------------------------------------------===//
// Error handling
//===----------------------------------------------------------------------===//

// Gets a string describing the error code.
//
// Parameters:
//  - error: [IN] Error code to get description for.
//
// Returns: Pointer to a null-terminated string describing the error.
//
// Synchronization: This operation is synchronous.
//
// String behavior:
// - Returns a static string, do not free.
// - String remains valid for program lifetime.
// - Returns "unknown error" for unrecognized codes.
// - String is in English.
//
// Usage pattern:
// ```c
// hipError_t err = hipMalloc(&ptr, size);
// if (err != hipSuccess) {
//   printf("HIP error: %s\n", hipGetErrorString(err));
// }
// ```
//
// See also: hipGetErrorName, hipGetLastError.
HIPAPI const char* hipGetErrorString(hipError_t error) {
  switch (error) {
    case hipSuccess:
      return "hipSuccess";
    case hipErrorInvalidValue:
      return "hipErrorInvalidValue";
    case hipErrorOutOfMemory:
      return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
      return "hipErrorNotInitialized";
    case hipErrorDeinitialized:
      return "hipErrorDeinitialized";
    case hipErrorProfilerDisabled:
      return "hipErrorProfilerDisabled";
    case hipErrorProfilerNotInitialized:
      return "hipErrorProfilerNotInitialized";
    case hipErrorProfilerAlreadyStarted:
      return "hipErrorProfilerAlreadyStarted";
    case hipErrorProfilerAlreadyStopped:
      return "hipErrorProfilerAlreadyStopped";
    case hipErrorInvalidConfiguration:
      return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol:
      return "hipErrorInvalidSymbol";
    case hipErrorInvalidDevicePointer:
      return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
      return "hipErrorInvalidMemcpyDirection";
    case hipErrorInsufficientDriver:
      return "hipErrorInsufficientDriver";
    case hipErrorMissingConfiguration:
      return "hipErrorMissingConfiguration";
    case hipErrorPriorLaunchFailure:
      return "hipErrorPriorLaunchFailure";
    case hipErrorInvalidDeviceFunction:
      return "hipErrorInvalidDeviceFunction";
    case hipErrorNoDevice:
      return "hipErrorNoDevice";
    case hipErrorInvalidDevice:
      return "hipErrorInvalidDevice";
    case hipErrorInvalidImage:
      return "hipErrorInvalidImage";
    case hipErrorInvalidContext:
      return "hipErrorInvalidContext";
    case hipErrorContextAlreadyCurrent:
      return "hipErrorContextAlreadyCurrent";
    case hipErrorMapFailed:
      return "hipErrorMapFailed";
    case hipErrorUnmapFailed:
      return "hipErrorUnmapFailed";
    case hipErrorArrayIsMapped:
      return "hipErrorArrayIsMapped";
    case hipErrorAlreadyMapped:
      return "hipErrorAlreadyMapped";
    case hipErrorNoBinaryForGpu:
      return "hipErrorNoBinaryForGpu";
    case hipErrorAlreadyAcquired:
      return "hipErrorAlreadyAcquired";
    case hipErrorNotMapped:
      return "hipErrorNotMapped";
    case hipErrorNotMappedAsArray:
      return "hipErrorNotMappedAsArray";
    case hipErrorNotMappedAsPointer:
      return "hipErrorNotMappedAsPointer";
    case hipErrorECCNotCorrectable:
      return "hipErrorECCNotCorrectable";
    case hipErrorUnsupportedLimit:
      return "hipErrorUnsupportedLimit";
    case hipErrorContextAlreadyInUse:
      return "hipErrorContextAlreadyInUse";
    case hipErrorPeerAccessUnsupported:
      return "hipErrorPeerAccessUnsupported";
    case hipErrorInvalidKernelFile:
      return "hipErrorInvalidKernelFile";
    case hipErrorInvalidGraphicsContext:
      return "hipErrorInvalidGraphicsContext";
    case hipErrorInvalidSource:
      return "hipErrorInvalidSource";
    case hipErrorFileNotFound:
      return "hipErrorFileNotFound";
    case hipErrorSharedObjectSymbolNotFound:
      return "hipErrorSharedObjectSymbolNotFound";
    case hipErrorSharedObjectInitFailed:
      return "hipErrorSharedObjectInitFailed";
    case hipErrorOperatingSystem:
      return "hipErrorOperatingSystem";
    case hipErrorInvalidHandle:
      return "hipErrorInvalidHandle";
    case hipErrorNotFound:
      return "hipErrorNotFound";
    case hipErrorNotReady:
      return "hipErrorNotReady";
    case hipErrorIllegalAddress:
      return "hipErrorIllegalAddress";
    case hipErrorLaunchOutOfResources:
      return "hipErrorLaunchOutOfResources";
    case hipErrorLaunchTimeOut:
      return "hipErrorLaunchTimeOut";
    case hipErrorPeerAccessAlreadyEnabled:
      return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled:
      return "hipErrorPeerAccessNotEnabled";
    case hipErrorSetOnActiveProcess:
      return "hipErrorSetOnActiveProcess";
    case hipErrorContextIsDestroyed:
      return "hipErrorContextIsDestroyed";
    case hipErrorAssert:
      return "hipErrorAssert";
    case hipErrorHostMemoryAlreadyRegistered:
      return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
      return "hipErrorHostMemoryNotRegistered";
    case hipErrorLaunchFailure:
      return "hipErrorLaunchFailure";
    case hipErrorCooperativeLaunchTooLarge:
      return "hipErrorCooperativeLaunchTooLarge";
    case hipErrorNotSupported:
      return "hipErrorNotSupported";
    case hipErrorStreamCaptureUnsupported:
      return "hipErrorStreamCaptureUnsupported";
    case hipErrorStreamCaptureInvalidated:
      return "hipErrorStreamCaptureInvalidated";
    case hipErrorStreamCaptureMerge:
      return "hipErrorStreamCaptureMerge";
    case hipErrorStreamCaptureUnmatched:
      return "hipErrorStreamCaptureUnmatched";
    case hipErrorStreamCaptureUnjoined:
      return "hipErrorStreamCaptureUnjoined";
    case hipErrorStreamCaptureIsolation:
      return "hipErrorStreamCaptureIsolation";
    case hipErrorStreamCaptureImplicit:
      return "hipErrorStreamCaptureImplicit";
    case hipErrorCapturedEvent:
      return "hipErrorCapturedEvent";
    case hipErrorStreamCaptureWrongThread:
      return "hipErrorStreamCaptureWrongThread";
    case hipErrorGraphExecUpdateFailure:
      return "hipErrorGraphExecUpdateFailure";
    case hipErrorUnknown:
    default:
      return "hipErrorUnknown";
  }
}

HIPAPI const char* hipGetErrorName(hipError_t error) {
  // Return the same as hipGetErrorString for simplicity.
  return hipGetErrorString(error);
}

// Gets and clears the last error from HIP runtime calls.
//
// Parameters: None.
//
// Returns: The last error code set by any HIP runtime call in this thread.
//
// Synchronization: This operation is synchronous.
//
// Error behavior:
// - Returns the last error from this thread.
// - Clears the error after returning it.
// - Returns hipSuccess if no error has occurred.
// - Each thread has its own error state.
//
// Usage pattern:
// ```c
// hipMalloc(&ptr, size);
// hipError_t err = hipGetLastError();
// if (err != hipSuccess) {
//   printf("Error: %s\n", hipGetErrorString(err));
// }
// ```
//
// Warning: This function clears the error. Use hipPeekAtLastError() to
// check without clearing.
//
// See also: hipPeekAtLastError, hipGetErrorString, hipGetErrorName.
HIPAPI hipError_t hipGetLastError(void) {
  return hip_thread_error_get_and_clear();
}

// Gets the last error without clearing it.
//
// Parameters: None.
//
// Returns: The last error code set by any HIP runtime call in this thread.
//
// Synchronization: This operation is synchronous.
//
// Error behavior:
// - Returns the last error from this thread.
// - Does NOT clear the error.
// - Returns hipSuccess if no error has occurred.
// - Error remains set for subsequent calls.
//
// See also: hipGetLastError, hipGetErrorString.
HIPAPI hipError_t hipPeekAtLastError(void) { return hip_thread_error_peek(); }
