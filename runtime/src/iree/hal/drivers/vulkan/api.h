// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_API_H_
#define IREE_HAL_DRIVERS_VULKAN_API_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Minimal Vulkan handle declarations
//===----------------------------------------------------------------------===//

// Declares only the opaque Vulkan handle types used by the public HAL API.
// Including full Vulkan headers here would leak loader/header policy into every
// user of the IREE runtime API.

#if !defined(VK_VERSION_1_0)
#if !defined(VK_DEFINE_HANDLE)
#define VK_DEFINE_HANDLE(object) typedef struct object##_T* object;
#endif  // !VK_DEFINE_HANDLE
#if !defined(VK_DEFINE_NON_DISPATCHABLE_HANDLE)
#if defined(__LP64__) || defined(_WIN64) ||                            \
    (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || \
    defined(__ia64) || defined(_M_IA64) || defined(__aarch64__) ||     \
    defined(__powerpc64__)
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) \
  typedef struct object##_T* object;
#else
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef uint64_t object;
#endif  // 64-bit pointer check
#endif  // !VK_DEFINE_NON_DISPATCHABLE_HANDLE

VK_DEFINE_HANDLE(VkInstance);
VK_DEFINE_HANDLE(VkPhysicalDevice);
VK_DEFINE_HANDLE(VkDevice);
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDeviceMemory);
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer);
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSemaphore);

#if defined(_WIN32)
#define IREE_HAL_VULKAN_API_PTR __stdcall
#else
#define IREE_HAL_VULKAN_API_PTR
#endif  // defined(_WIN32)

typedef void(IREE_HAL_VULKAN_API_PTR* PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction(IREE_HAL_VULKAN_API_PTR* PFN_vkGetInstanceProcAddr)(
    VkInstance instance, const char* name);
#undef IREE_HAL_VULKAN_API_PTR
#endif  // !VK_VERSION_1_0

//===----------------------------------------------------------------------===//
// Request, feature, and extension policy
//===----------------------------------------------------------------------===//

// Bitfield that defines non-device-feature behavior requested during Vulkan
// driver and instance creation.
typedef enum iree_hal_vulkan_request_flag_bits_t {
  // No optional behavior requested.
  IREE_HAL_VULKAN_REQUEST_FLAG_NONE = 0u,
  // Requests validation layers during driver-created instance setup.
  IREE_HAL_VULKAN_REQUEST_FLAG_VALIDATION_LAYERS = 1u << 0,
  // Requests VK_EXT_debug_utils object names and command labels.
  IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS = 1u << 1,
  // Requests Vulkan events in IREE HAL profiling streams.
  IREE_HAL_VULKAN_REQUEST_FLAG_TRACING = 1u << 2,
  // Recognized request flags accepted by public Vulkan HAL APIs.
  IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED =
      IREE_HAL_VULKAN_REQUEST_FLAG_VALIDATION_LAYERS |
      IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS |
      IREE_HAL_VULKAN_REQUEST_FLAG_TRACING,
} iree_hal_vulkan_request_flag_bits_t;

typedef uint32_t iree_hal_vulkan_request_flags_t;

// Bitfield that defines Vulkan features requested or enabled on a logical
// device.
typedef enum iree_hal_vulkan_feature_bits_t {
  // No optional features requested.
  IREE_HAL_VULKAN_FEATURE_NONE = 0u,
  // Bit positions 0-2 are reserved for iree_hal_vulkan_request_flags_t.
  // Requests robust buffer access for validation-oriented runs.
  IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS = 1u << 3,
  // Requests sparse binding for large virtual buffers.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING = 1u << 4,
  // Requests sparse residency with aliased sparse buffer mappings.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED =
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING | (1u << 5),
  // Requests buffer device address support for pointer-first executables.
  IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES = 1u << 6,
  // Reports timeline semaphore support enabled on a logical device.
  IREE_HAL_VULKAN_FEATURE_ENABLE_TIMELINE_SEMAPHORES = 1u << 7,
  // Reports synchronization2 support enabled on a logical device.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SYNCHRONIZATION2 = 1u << 8,
  // Reports scalar block layout support enabled on a logical device.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SCALAR_BLOCK_LAYOUT = 1u << 9,
  // Requests and reports subgroup size control on a logical device.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL = 1u << 10,
  // Requests and reports KHR cooperative matrix support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX = 1u << 11,
  // Requests and reports storageBuffer8BitAccess support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS = 1u << 12,
  // Requests and reports shaderFloat16 support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 = 1u << 13,
  // Requests and reports shaderFloat64 support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64 = 1u << 14,
  // Requests and reports shaderInt8 support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8 = 1u << 15,
  // Requests and reports shaderInt16 support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16 = 1u << 16,
  // Requests and reports shaderInt64 support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64 = 1u << 17,
  // Requests and reports shaderIntegerDotProduct support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT = 1u << 18,
  // Requests and reports vulkanMemoryModel support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL = 1u << 19,
  // Requests and reports vulkanMemoryModelDeviceScope support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 1u << 20,
  // Requests and reports storageBuffer16BitAccess support.
  IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS = 1u << 21,
  // Required enabled logical-device feature set for the Vulkan HAL baseline.
  IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE =
      IREE_HAL_VULKAN_FEATURE_ENABLE_TIMELINE_SEMAPHORES |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SYNCHRONIZATION2 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SCALAR_BLOCK_LAYOUT,
  // Recognized feature bits accepted by public Vulkan HAL APIs.
  IREE_HAL_VULKAN_FEATURE_ALL_RECOGNIZED =
      IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED |
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
      IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL |
      IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX |
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS |
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
} iree_hal_vulkan_feature_bits_t;

typedef uint32_t iree_hal_vulkan_features_t;

// Recognized device extension bits cached from device enumeration.
typedef enum iree_hal_vulkan_device_extension_bits_t {
  // No recognized device extensions.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE = 0u,
  // VK_KHR_portability_subset.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET = 1u << 0,
  // VK_KHR_external_memory.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY = 1u << 1,
  // VK_KHR_external_memory_fd.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD = 1u << 2,
  // VK_KHR_external_memory_win32.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32 = 1u << 3,
  // VK_EXT_external_memory_host.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST = 1u << 4,
  // VK_EXT_calibrated_timestamps.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS = 1u << 5,
  // VK_KHR_push_descriptor.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR = 1u << 6,
  // VK_KHR_cooperative_matrix.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX = 1u << 7,
  // Recognized extension bits accepted by public Vulkan HAL APIs.
  IREE_HAL_VULKAN_DEVICE_EXTENSION_ALL_RECOGNIZED =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PORTABILITY_SUBSET |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_FD |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_EXTERNAL_MEMORY_WIN32 |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_EXTERNAL_MEMORY_HOST |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_PUSH_DESCRIPTOR |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX,
} iree_hal_vulkan_device_extension_bits_t;

typedef uint32_t iree_hal_vulkan_device_extensions_t;

// Populates recognized Vulkan device extension bits from an enabled extension
// name list. Unknown extension names are ignored.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_device_extensions_from_names(
    iree_host_size_t extension_count, const char* const* extension_names,
    iree_hal_vulkan_device_extensions_t* out_extensions);

// Identifies a layer or extension name set exposed through the public API.
typedef enum iree_hal_vulkan_extensibility_set_e {
  // Required instance layer names.
  IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED = 0,
  // Optional instance layer names.
  IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
  // Required instance extension names.
  IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
  // Optional instance extension names.
  IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
  // Required device extension names.
  IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
  // Optional device extension names.
  IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
  // Count of defined extensibility sets.
  IREE_HAL_VULKAN_EXTENSIBILITY_SET_COUNT,
} iree_hal_vulkan_extensibility_set_t;

// Queries the Vulkan layer or extension names used by a request flag set.
// Required sets must be enabled by applications wrapping external instances or
// devices. Optional sets should be enabled when the Vulkan implementation
// reports them so IREE can use the corresponding strategy bits.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_query_extensibility_set(
    iree_hal_vulkan_request_flags_t request_flags,
    iree_hal_vulkan_extensibility_set_t set, iree_host_size_t string_capacity,
    iree_host_size_t* out_string_count, const char** out_string_values);

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_syms_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_syms_t iree_hal_vulkan_syms_t;

// Wraps an externally resolved vkGetInstanceProcAddr pointer for future Vulkan
// dispatch table loading.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_syms_create(
    void* vkGetInstanceProcAddr_fn, iree_allocator_t host_allocator,
    iree_hal_vulkan_syms_t** out_syms);

// Loads Vulkan functions from the system loader.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_syms_create_from_system_loader(
    iree_allocator_t host_allocator, iree_hal_vulkan_syms_t** out_syms);

// Retains |syms| for the caller.
IREE_API_EXPORT void iree_hal_vulkan_syms_retain(iree_hal_vulkan_syms_t* syms);

// Releases |syms| from the caller.
IREE_API_EXPORT void iree_hal_vulkan_syms_release(iree_hal_vulkan_syms_t* syms);

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_device_t
//===----------------------------------------------------------------------===//

// A set of queues within a specific queue family on a VkDevice.
typedef struct iree_hal_vulkan_queue_set_t {
  // Queue family index from vkGetPhysicalDeviceQueueFamilyProperties.
  uint32_t queue_family_index;
  // Bitfield of queue indices within the queue family.
  uint64_t queue_indices;
} iree_hal_vulkan_queue_set_t;

// Externally-created VkDevice contract supplied to iree_hal_vulkan_wrap_device.
typedef struct iree_hal_vulkan_external_device_params_t {
  // Feature bits enabled on the VkDevice and usable by IREE. Must include
  // IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE.
  iree_hal_vulkan_features_t enabled_features;

  // Recognized device extension bits enabled on the VkDevice.
  iree_hal_vulkan_device_extensions_t enabled_extensions;

  // Non-device-feature behavior IREE may use with the wrapped VkInstance and
  // VkDevice.
  iree_hal_vulkan_request_flags_t request_flags;

  // Compute-capable queue family and indices available to IREE.
  iree_hal_vulkan_queue_set_t compute_queue_set;

  // Transfer-capable queue family and indices available to IREE. Leave
  // queue_indices zero to reuse the selected compute queue for transfers.
  iree_hal_vulkan_queue_set_t transfer_queue_set;

  // Sparse-binding-capable queue family and indices available to IREE. Leave
  // queue_indices zero to reuse compute or transfer when either supports sparse
  // binding.
  iree_hal_vulkan_queue_set_t sparse_binding_queue_set;
} iree_hal_vulkan_external_device_params_t;

// KHR cooperative matrix property row reported by a Vulkan logical device.
//
// Enum-valued fields preserve the Vulkan wire values from VkComponentTypeKHR
// and VkScopeKHR. This keeps the public HAL API independent of full Vulkan
// headers while allowing compiler targets to compare against their own SPIR-V
// wire enums without string translation.
typedef struct iree_hal_vulkan_cooperative_matrix_property_t {
  // Matrix M dimension.
  uint32_t m_size;

  // Matrix N dimension.
  uint32_t n_size;

  // Matrix K dimension.
  uint32_t k_size;

  // Matrix A component type as VkComponentTypeKHR.
  uint32_t a_type;

  // Matrix B component type as VkComponentTypeKHR.
  uint32_t b_type;

  // Matrix C accumulator component type as VkComponentTypeKHR.
  uint32_t c_type;

  // Result component type as VkComponentTypeKHR.
  uint32_t result_type;

  // Non-zero if saturating accumulation is supported for this row.
  uint32_t saturating_accumulation;

  // Cooperative matrix scope as VkScopeKHR.
  uint32_t scope;
} iree_hal_vulkan_cooperative_matrix_property_t;

// Queries active KHR cooperative matrix property rows for a Vulkan HAL device.
//
// Returns OK with |out_property_count| set to zero when the logical device does
// not have cooperative matrix support enabled. Passing NULL |out_properties|
// with zero |property_capacity| performs a count-only query. If non-NULL
// storage is provided and |property_capacity| is too small, returns
// OUT_OF_RANGE after writing the required count.
IREE_API_EXPORT iree_status_t
iree_hal_vulkan_device_query_cooperative_matrix_properties(
    iree_hal_device_t* device, iree_host_size_t property_capacity,
    iree_host_size_t* out_property_count,
    iree_hal_vulkan_cooperative_matrix_property_t* out_properties);

typedef enum iree_hal_vulkan_device_flag_bits_t {
  // No device flags.
  IREE_HAL_VULKAN_DEVICE_FLAG_NONE = 0u,
  // Prefer a dedicated compute queue without graphics capabilities.
  IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE = 1u << 0,
} iree_hal_vulkan_device_flag_bits_t;

typedef uint32_t iree_hal_vulkan_device_flags_t;

typedef enum iree_hal_vulkan_dispatch_abi_bits_t {
  // No executable dispatch ABIs are enabled.
  IREE_HAL_VULKAN_DISPATCH_ABI_NONE = 0u,
  // Descriptor-set based storage-buffer dispatch ABI.
  IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR = 1u << 0,
  // Buffer-device-address based storage-buffer dispatch ABI.
  IREE_HAL_VULKAN_DISPATCH_ABI_BDA = 1u << 1,
  // Recognized executable dispatch ABI bits accepted by public APIs.
  IREE_HAL_VULKAN_DISPATCH_ABI_ALL_RECOGNIZED =
      IREE_HAL_VULKAN_DISPATCH_ABI_DESCRIPTOR |
      IREE_HAL_VULKAN_DISPATCH_ABI_BDA,
} iree_hal_vulkan_dispatch_abi_bits_t;

typedef uint32_t iree_hal_vulkan_dispatch_abis_t;

// Parses a dispatch ABI option string.
//
// Accepted values are "descriptor", "bda", and "all". The output is a
// non-empty bitmask suitable for iree_hal_vulkan_device_options_t.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_dispatch_abis_parse(
    iree_string_view_t value, iree_hal_vulkan_dispatch_abis_t* out_abis);

// Verifies that |dispatch_abis| contains a non-empty recognized ABI bit set.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_dispatch_abis_verify(
    iree_hal_vulkan_dispatch_abis_t dispatch_abis);

// Parameters configuring an iree_hal_vulkan_device_t.
typedef struct iree_hal_vulkan_device_options_t {
  // Device behavior flags.
  iree_hal_vulkan_device_flags_t flags;

  // Requested executable dispatch ABIs for this logical device.
  //
  // Device creation enables the subset whose required Vulkan features are
  // present. Requesting only an unsupported ABI fails loudly.
  iree_hal_vulkan_dispatch_abis_t dispatch_abis;

  // Maximum cached native BDA replay instances retained per queue lane.
  //
  // Additional concurrent command-buffer executions use the one-shot replay
  // path after this limit is reached. Set to 0 to disable native BDA replay
  // caching.
  uint32_t max_cached_bda_replay_instances;

  // Maximum host-visible BDA publication bytes retained by cached replay
  // instances per queue lane.
  uint64_t max_cached_bda_replay_publication_bytes;

  // Idle cached native BDA replay instances retained per command buffer by
  // trim.
  uint32_t retained_cached_bda_replay_instances;
} iree_hal_vulkan_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_vulkan_device_options_initialize(
    iree_hal_vulkan_device_options_t* out_options);

// Creates a Vulkan HAL device that wraps an existing VkDevice.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_wrap_device(
    iree_string_view_t identifier,
    const iree_hal_vulkan_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    const iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    VkPhysicalDevice physical_device, VkDevice logical_device,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_driver_t
//===----------------------------------------------------------------------===//

// Vulkan driver creation options.
typedef struct iree_hal_vulkan_driver_options_t {
  // Search paths (directories or files) for finding the Vulkan loader shared
  // library. Driver creation clones these strings; callers only need to keep
  // them live until iree_hal_vulkan_driver_create returns.
  iree_string_view_list_t libvulkan_search_paths;

  // Vulkan API version requested by driver-created instances.
  uint32_t api_version;

  // Non-device-feature behavior requested for driver-created instances.
  iree_hal_vulkan_request_flags_t request_flags;

  // Feature bits requested for driver-created logical devices.
  iree_hal_vulkan_features_t requested_features;

  // Cutoff for debug output: 0=none, 1=errors, 2=warnings, 3=info, 4=debug.
  int32_t debug_verbosity;

  // Default options for devices created by this driver.
  iree_hal_vulkan_device_options_t device_options;
} iree_hal_vulkan_driver_options_t;

// Initializes |out_options| with default driver creation options.
IREE_API_EXPORT void iree_hal_vulkan_driver_options_initialize(
    iree_hal_vulkan_driver_options_t* out_options);

// Creates a Vulkan HAL driver that will own its Vulkan instance.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_driver_create(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* syms, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

// Creates a Vulkan HAL driver over an application-provided Vulkan instance.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_driver_create_using_instance(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

// Returns the Vulkan memory and buffer handle backing an allocated HAL buffer.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_allocated_buffer_handle(
    iree_hal_buffer_t* allocated_buffer, VkDeviceMemory* out_memory,
    VkBuffer* out_handle);

// Returns the Vulkan semaphore handle backing a HAL semaphore.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_semaphore_handle(
    iree_hal_semaphore_t* semaphore, VkSemaphore* out_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_API_H_
