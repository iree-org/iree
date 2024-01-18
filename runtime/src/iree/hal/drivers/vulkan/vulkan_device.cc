// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/vulkan_device.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/builtin_executables.h"
#include "iree/hal/drivers/vulkan/command_queue.h"
#include "iree/hal/drivers/vulkan/descriptor_pool_cache.h"
#include "iree/hal/drivers/vulkan/direct_command_buffer.h"
#include "iree/hal/drivers/vulkan/direct_command_queue.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/extensibility_util.h"
#include "iree/hal/drivers/vulkan/handle_util.h"
#include "iree/hal/drivers/vulkan/native_allocator.h"
#include "iree/hal/drivers/vulkan/native_event.h"
#include "iree/hal/drivers/vulkan/native_pipeline_layout.h"
#include "iree/hal/drivers/vulkan/native_semaphore.h"
#include "iree/hal/drivers/vulkan/nop_executable_cache.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/tracing.h"
#include "iree/hal/drivers/vulkan/util/arena.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// RenderDoc integration
//===----------------------------------------------------------------------===//

// Configure cmake with -DIREE_ENABLE_RENDERDOC_PROFILING=ON in order to
// enable profiling support. This should be left off in production builds to
// avoid introducing a backdoor.
#if defined(IREE_HAL_VULKAN_HAVE_RENDERDOC)

#if !defined(IREE_PLATFORM_WINDOWS)
#include <dlfcn.h>
#endif  // IREE_PLATFORM_WINDOWS

// NOTE: C API, see https://renderdoc.org/docs/in_application_api.html.
// When compiled in the API will no-op itself if not running under a RenderDoc
// capture context (renderdoc.dll/so already loaded).
#include "third_party/renderdoc/renderdoc_app.h"

typedef RENDERDOC_API_1_5_0 RENDERDOC_API_LATEST;

// Returns a handle to the RenderDoc API when it is hooking the process.
// Returns NULL when RenderDoc is not present (or valid).
static RENDERDOC_API_LATEST* iree_hal_vulkan_query_renderdoc_api(
    VkInstance instance) {
  pRENDERDOC_GetAPI RENDERDOC_GetAPI = NULL;
#if defined(IREE_PLATFORM_WINDOWS)

  // NOTE: RenderDoc only supports hooking so we can't use LoadLibrary - if
  // we're going to use RenderDoc its library must already be loaded.
  if (HMODULE hook_module = GetModuleHandleA("renderdoc.dll")) {
    RENDERDOC_GetAPI =
        (pRENDERDOC_GetAPI)GetProcAddress(hook_module, "RENDERDOC_GetAPI");
  }

#else

  // dlopen/dlsym on posix-like systems. Note that each platform has its own
  // naming for the injected module. Because RenderDoc only supports hooking
  // (where the hosting process loads the library in magic ways for us) we use
  // RTLD_NOLOAD to ensure we don't accidentally try to load it when not hooked.
  void* hook_module = NULL;
#if defined(IREE_PLATFORM_ANDROID)
  hook_module = dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD);
#elif defined(IREE_PLATFORM_APPLE)
  hook_module = dlopen("librenderdoc.dylib", RTLD_NOW | RTLD_NOLOAD);
#elif defined(IREE_PLATFORM_LINUX)
  hook_module = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
#else
#error "RenderDoc profiling not supported on this platform"
#endif  // IREE_PLATFORM_*
  if (hook_module) {
    RENDERDOC_GetAPI =
        (pRENDERDOC_GetAPI)dlsym(hook_module, "RENDERDOC_GetAPI");
  }

#endif  // IREE_PLATFORM_WINDOWS

  if (!RENDERDOC_GetAPI) return NULL;  // not found, no-op

  RENDERDOC_API_LATEST* api = NULL;
  int query_result =
      RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_5_0, (void**)&api);
  if (query_result != 1) {
    // Failed to initialize API (old version, etc). No-op.
    return NULL;
  }

  return api;
}

// Begins a new RenderDoc capture.
static void iree_hal_vulkan_begin_renderdoc_capture(
    RENDERDOC_API_LATEST* renderdoc_api, VkInstance instance,
    const iree_hal_device_profiling_options_t* options) {
  if (!renderdoc_api) return;
  if (options->file_path) {
    renderdoc_api->SetCaptureFilePathTemplate(options->file_path);
  }
  renderdoc_api->StartFrameCapture(
      RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance), NULL);
}

// Ends the active RenderDoc capture, if any active.
static void iree_hal_vulkan_end_renderdoc_capture(
    RENDERDOC_API_LATEST* renderdoc_api, VkInstance instance) {
  if (!renderdoc_api) return;
  if (renderdoc_api->IsFrameCapturing()) {
    renderdoc_api->EndFrameCapture(
        RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance), NULL);
  }
}

#endif  // IREE_HAL_VULKAN_HAVE_RENDERDOC

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_device_t extensibility util
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_vulkan_query_extensibility_set(
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_extensibility_set_t set, iree_host_size_t string_capacity,
    iree_host_size_t* out_string_count, const char** out_string_values) {
  *out_string_count = 0;

  iree_status_t status = iree_ok_status();
  iree_host_size_t string_count = 0;
#define ADD_EXT(target_set, name_literal)                       \
  if (iree_status_is_ok(status) && set == (target_set)) {       \
    if (string_count >= string_capacity && out_string_values) { \
      status = iree_status_from_code(IREE_STATUS_OUT_OF_RANGE); \
    } else if (out_string_values) {                             \
      out_string_values[string_count] = (name_literal);         \
    }                                                           \
    ++string_count;                                             \
  }

  //===--------------------------------------------------------------------===//
  // Baseline IREE requirements
  //===--------------------------------------------------------------------===//
  // Using IREE at all requires these extensions unconditionally. Adding things
  // here changes our minimum requirements and should be done carefully.
  // Optional extensions here are feature detected by the runtime.

#if defined(IREE_PLATFORM_APPLE)
  // VK_KHR_portability_subset:
  // For Apple platforms, Vulkan is layered on top of Metal via MoltenVK.
  // It exposes this extension to allow a non-conformant Vulkan implementation
  // to be built on top of another non-Vulkan graphics API. This extension must
  // be enabled if exists.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
          VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

  // VK_KHR_portability_enumeration:
  // Further, since devices which support the VK_KHR_portability_subset
  // extension are not fully conformant Vulkan implementations, the Vulkan
  // loader does not report those devices unless the application explicitly
  // asks for them.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
          VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  // VK_KHR_storage_buffer_storage_class:
  // Our generated SPIR-V kernels use storage buffers for all their data access.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
          VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);

  // VK_KHR_get_physical_device_properties2:
  // Multiple extensions depend on VK_KHR_get_physical_device_properties2.
  // This extension was deprecated in Vulkan 1.1 as its functionality was
  // promoted to core so we list it as optional even though we require it.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
          VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  // VK_KHR_push_descriptor:
  // We can avoid a lot of additional Vulkan descriptor set manipulation
  // overhead when this extension is present. Android is a holdout, though, and
  // we have a fallback for when it's not available.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // VK_KHR_timeline_semaphore:
  // Required as IREE's primary synchronization primitive, but the extension
  // was promoted to core in Vulkan 1.2.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);

  // VK_KHR_external_memory:
  // Promoted to core in Vulkan 1.1 and not required but here just in case
  // tooling wants to see the request.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);

  // VK_EXT_external_memory_host:
  // Optional to enable import/export of host pointers.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);

  // VK_KHR_buffer_device_address:
  // Promoted to core in Vulkan 1.2 but still an extension in 1.1.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

  //===--------------------------------------------------------------------===//
  // Vulkan forward-compatibility shims
  //===--------------------------------------------------------------------===//
  // These are shims or extensions that are made core later in the spec and can
  // be removed once we require the core version that contains them.

  // VK_LAYER_KHRONOS_timeline_semaphore:
  // polyfill layer - enable if present. Ignored if timeline semaphores are
  // supported natively (Vulkan 1.2+).
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
          "VK_LAYER_KHRONOS_timeline_semaphore");

  //===--------------------------------------------------------------------===//
  // Optional CodeGen features
  //===--------------------------------------------------------------------===//
  // VK_EXT_subgroup_size_control:
  // This extensions allows us to control the subgroup size used by Vulkan
  // implementations, which can boost performance. It's promoted to core
  // since Vulkan v1.3.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

  // VK_KHR_8bit_storage:
  // This extension allows use of 8-bit types in uniform and storage buffers,
  // and push constant blocks. It's promoted to core since Vulkan 1.2.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_8BIT_STORAGE_EXTENSION_NAME);

  // VK_KHR_shader_float16_int8:
  // This extension allows use of 16-bit floating-point types and 8-bit integer
  // types in shaders for arithmetic operations. It's promoted to core since
  // Vulkan 1.2.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);

  // VK_KHR_cooperative_matrix:
  // This extension exposes SIMD matrix-matrix multiply accumulate operations.
  // It's available in Vulkan 1.3.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

  //===--------------------------------------------------------------------===//
  // Optional debugging features
  //===--------------------------------------------------------------------===//
  // Used only when explicitly requested as they drastically change the
  // performance behavior of Vulkan.

  // VK_LAYER_KHRONOS_validation:
  // only enabled if validation is desired. Since validation in Vulkan is just a
  // API correctness check it can't be used as a security mechanism and is fine
  // to ignore.
  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS)) {
    ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
            "VK_LAYER_KHRONOS_validation");
  }

  // VK_EXT_debug_utils:
  // only enabled if debugging is desired to route Vulkan debug messages through
  // our logging sinks. Note that this adds a non-trivial runtime overhead and
  // we may want to disable it even in debug builds.
  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS)) {
    ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING)) {
    // VK_EXT_host_query_reset:
    // optionally allows for vkResetQueryPool to be used to reset query pools
    // from the host without needing to do an expensive vkCmdResetQueryPool
    // submission.
    ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
            VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);

    // VK_EXT_calibrated_timestamps:
    // optionally provides more accurate timestamps that correspond to the
    // system time. If this is not present then tracy will attempt calibration
    // itself and have some per-run variance in the skew (up to many
    // milliseconds).
    ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  *out_string_count = string_count;
  return status;
}

//===----------------------------------------------------------------------===//
// Queue selection
//===----------------------------------------------------------------------===//

#define IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX (-1)

typedef struct iree_hal_vulkan_queue_family_info_t {
  uint32_t dispatch_index;
  iree_host_size_t dispatch_queue_count;
  uint32_t transfer_index;
  iree_host_size_t transfer_queue_count;
} iree_hal_vulkan_queue_family_info_t;

// Finds the first queue in the listing (which is usually the
// driver-preferred) that has all of the |required_queue_flags| and none of
// the |excluded_queue_flags|.
// Returns IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX if no matching queue is
// found.
static uint32_t iree_hal_vulkan_find_first_queue_family_with_flags(
    uint32_t queue_family_count,
    const VkQueueFamilyProperties* queue_family_properties,
    VkQueueFlags required_queue_flags, VkQueueFlags excluded_queue_flags) {
  for (uint32_t queue_family_index = 0; queue_family_index < queue_family_count;
       ++queue_family_index) {
    const VkQueueFamilyProperties* properties =
        &queue_family_properties[queue_family_index];
    if (iree_all_bits_set(properties->queueFlags, required_queue_flags) &&
        !iree_any_bit_set(properties->queueFlags, excluded_queue_flags)) {
      return queue_family_index;
    }
  }
  return IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX;
}

// Selects queue family indices for compute and transfer queues.
// Note that both queue families may be the same if there is only one family
// available.
static iree_status_t iree_hal_vulkan_select_queue_families(
    const iree_hal_vulkan_device_options_t* options,
    VkPhysicalDevice physical_device, iree::hal::vulkan::DynamicSymbols* syms,
    iree_hal_vulkan_queue_family_info_t* out_family_info) {
  // Enumerate queue families available on the device.
  uint32_t queue_family_count = 0;
  syms->vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count, NULL);
  VkQueueFamilyProperties* queue_family_properties =
      (VkQueueFamilyProperties*)iree_alloca(queue_family_count *
                                            sizeof(VkQueueFamilyProperties));
  syms->vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_family_properties);

  memset(out_family_info, 0, sizeof(*out_family_info));
  out_family_info->dispatch_index = IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX;
  out_family_info->dispatch_queue_count = 0;
  out_family_info->transfer_index = IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX;
  out_family_info->transfer_queue_count = 0;

  // By default we choose graphics+compute as on most current GPUs this is a
  // primary queue and may run at the fastest clock speed.
  // If the user is integrating into applications with existing graphics
  // workloads then they can request that we instead try to find a dedicated
  // compute-only queue such that we can run async with the rest of their
  // existing workload.
  if (iree_all_bits_set(options->flags,
                        IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE)) {
    // Try to find a dedicated compute queue. If this fails then we'll fall back
    // to any queue supporting compute.
    out_family_info->dispatch_index =
        iree_hal_vulkan_find_first_queue_family_with_flags(
            queue_family_count, queue_family_properties, VK_QUEUE_COMPUTE_BIT,
            VK_QUEUE_GRAPHICS_BIT);
  }
  if (out_family_info->dispatch_index ==
      IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX) {
    out_family_info->dispatch_index =
        iree_hal_vulkan_find_first_queue_family_with_flags(
            queue_family_count, queue_family_properties, VK_QUEUE_COMPUTE_BIT,
            0);
  }
  if (out_family_info->dispatch_index ==
      IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "unable to find any queue family support compute operations");
  }
  out_family_info->dispatch_queue_count =
      queue_family_properties[out_family_info->dispatch_index].queueCount;

  // Try to find a dedicated transfer queue (no compute or graphics caps).
  // Not all devices have one, and some have only a queue family for
  // everything and possibly a queue family just for compute/etc. If that
  // fails then fallback to any queue that supports transfer. Finally, if
  // /that/ fails then we just won't create a transfer queue and instead use
  // the compute queue for all operations.
  out_family_info->transfer_index =
      iree_hal_vulkan_find_first_queue_family_with_flags(
          queue_family_count, queue_family_properties, VK_QUEUE_TRANSFER_BIT,
          VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
  if (out_family_info->transfer_index ==
      IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX) {
    out_family_info->transfer_index =
        iree_hal_vulkan_find_first_queue_family_with_flags(
            queue_family_count, queue_family_properties, VK_QUEUE_TRANSFER_BIT,
            VK_QUEUE_GRAPHICS_BIT);
  }
  if (out_family_info->transfer_index ==
      IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX) {
    out_family_info->transfer_index =
        iree_hal_vulkan_find_first_queue_family_with_flags(
            queue_family_count, queue_family_properties, VK_QUEUE_TRANSFER_BIT,
            0);
  }
  if (out_family_info->transfer_index !=
      IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX) {
    out_family_info->transfer_queue_count =
        queue_family_properties[out_family_info->transfer_index].queueCount;
  }

  // Ensure that we don't share the dispatch queues with transfer queues if
  // that would put us over the queue count.
  if (out_family_info->dispatch_index == out_family_info->transfer_index) {
    out_family_info->transfer_queue_count = iree_min(
        queue_family_properties[out_family_info->dispatch_index].queueCount -
            out_family_info->dispatch_queue_count,
        out_family_info->transfer_queue_count);
  }

  // Limit the number of queues we create (for now).
  // We may want to allow this to grow, but each queue adds overhead and we
  // need to measure to make sure we can effectively use them all.
  out_family_info->dispatch_queue_count =
      iree_min(2u, out_family_info->dispatch_queue_count);
  out_family_info->transfer_queue_count =
      iree_min(1u, out_family_info->transfer_queue_count);

  return iree_ok_status();
}

// Builds a set of compute and transfer queues based on the queues available on
// the device and some magic heuristical goo.
static iree_status_t iree_hal_vulkan_build_queue_sets(
    const iree_hal_vulkan_device_options_t* options,
    VkPhysicalDevice physical_device, iree::hal::vulkan::DynamicSymbols* syms,
    iree_hal_vulkan_queue_set_t* out_compute_queue_set,
    iree_hal_vulkan_queue_set_t* out_transfer_queue_set) {
  // Select which queues to use (and fail the implementation can't handle them).
  iree_hal_vulkan_queue_family_info_t queue_family_info;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_queue_families(
      options, physical_device, syms, &queue_family_info));

  // Build queue indices for the selected queue families.
  memset(out_compute_queue_set, 0, sizeof(*out_compute_queue_set));
  out_compute_queue_set->queue_family_index = queue_family_info.dispatch_index;
  for (iree_host_size_t i = 0; i < queue_family_info.dispatch_queue_count;
       ++i) {
    out_compute_queue_set->queue_indices |= 1ull << i;
  }

  memset(out_transfer_queue_set, 0, sizeof(*out_transfer_queue_set));
  out_transfer_queue_set->queue_family_index = queue_family_info.transfer_index;
  uint32_t base_queue_index = 0;
  if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
    // Sharing a family, so transfer queues follow compute queues.
    base_queue_index = queue_family_info.dispatch_index;
  }
  for (iree_host_size_t i = 0; i < queue_family_info.transfer_queue_count;
       ++i) {
    out_transfer_queue_set->queue_indices |= 1ull << (i + base_queue_index);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Optional driver that owns the instance. We retain it for our lifetime to
  // ensure the instance remains valid.
  iree_hal_driver_t* driver;

  // Flags overriding default device behavior.
  iree_hal_vulkan_device_flags_t flags;
  // Which optional extensions are active and available on the device.
  iree_hal_vulkan_device_extensions_t device_extensions;
  // Device properties for various optional features.
  iree_hal_vulkan_device_properties_t device_properties;

  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDeviceHandle* logical_device;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // All queues available on the device; the device owns these.
  iree_host_size_t queue_count;
  CommandQueue** queues;
  // The subset of queues that support dispatch operations. May overlap with
  // transfer_queues.
  iree_host_size_t dispatch_queue_count;
  CommandQueue** dispatch_queues;
  // The subset of queues that support transfer operations. May overlap with
  // dispatch_queues.
  iree_host_size_t transfer_queue_count;
  CommandQueue** transfer_queues;

  // |queue_count| tracing contexts, if tracing is enabled.
  iree_hal_vulkan_tracing_context_t** queue_tracing_contexts;

  DescriptorPoolCache* descriptor_pool_cache;

  VkCommandPoolHandle* dispatch_command_pool;
  VkCommandPoolHandle* transfer_command_pool;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  BuiltinExecutables* builtin_executables;

#if defined(IREE_HAL_VULKAN_HAVE_RENDERDOC)
  RENDERDOC_API_LATEST* renderdoc_api;
#endif  // IREE_HAL_VULKAN_HAVE_RENDERDOC
} iree_hal_vulkan_device_t;

namespace {
extern const iree_hal_device_vtable_t iree_hal_vulkan_device_vtable;
}  // namespace

static iree_hal_vulkan_device_t* iree_hal_vulkan_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_device_vtable);
  return (iree_hal_vulkan_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_vulkan_device_options_initialize(
    iree_hal_vulkan_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->flags = 0;
}

// Creates a transient command pool for the given queue family.
// Command buffers allocated from the pool must only be issued on queues
// belonging to the specified family.
static iree_status_t iree_hal_vulkan_create_transient_command_pool(
    VkDeviceHandle* logical_device, uint32_t queue_family_index,
    VkCommandPoolHandle** out_handle) {
  VkCommandPoolCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.pNext = NULL;
  create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  create_info.queueFamilyIndex = queue_family_index;
  VkCommandPoolHandle* command_pool = new VkCommandPoolHandle(logical_device);
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkCreateCommandPool(
          *logical_device, &create_info, logical_device->allocator(),
          command_pool->mutable_value()),
      "vkCreateCommandPool");
  if (iree_status_is_ok(status)) {
    *out_handle = command_pool;
  } else {
    delete command_pool;
  }
  return status;
}

// Creates a command queue of the given queue family.
static CommandQueue* iree_hal_vulkan_device_create_queue(
    VkDeviceHandle* logical_device,
    iree_hal_command_category_t command_category, uint32_t queue_family_index,
    uint32_t queue_index) {
  VkQueue queue = VK_NULL_HANDLE;
  logical_device->syms()->vkGetDeviceQueue(*logical_device, queue_family_index,
                                           queue_index, &queue);

  return new DirectCommandQueue(logical_device, command_category, queue);
}

// Creates command queues for the given sets of queues and populates the
// device queue lists.
static iree_status_t iree_hal_vulkan_device_initialize_command_queues(
    iree_hal_vulkan_device_t* device,
    iree_hal_vulkan_features_t enabled_features,
    iree_string_view_t queue_prefix,
    const iree_hal_vulkan_queue_set_t* compute_queue_set,
    const iree_hal_vulkan_queue_set_t* transfer_queue_set) {
  device->queue_count = 0;
  device->dispatch_queue_count = 0;
  device->transfer_queue_count = 0;

  // The first available queue supporting dispatch commands that will be used by
  // the tracing subsystem for query and cleanup tasks.
  VkQueue maintenance_dispatch_queue = VK_NULL_HANDLE;

  uint64_t compute_queue_count =
      iree_math_count_ones_u64(compute_queue_set->queue_indices);
  uint64_t transfer_queue_count =
      iree_math_count_ones_u64(transfer_queue_set->queue_indices);
  for (iree_host_size_t i = 0; i < compute_queue_count; ++i) {
    if (!(compute_queue_set->queue_indices & (1ull << i))) continue;

    char queue_name_buffer[32];
    int queue_name_length =
        snprintf(queue_name_buffer, IREE_ARRAYSIZE(queue_name_buffer),
                 "Vulkan[%c:%d]", 'D', (int)device->dispatch_queue_count);
    iree_string_view_t queue_name =
        iree_make_string_view(queue_name_buffer, queue_name_length);

    CommandQueue* queue = iree_hal_vulkan_device_create_queue(
        device->logical_device, IREE_HAL_COMMAND_CATEGORY_ANY,
        compute_queue_set->queue_family_index, i);

    iree_host_size_t queue_index = device->queue_count++;
    device->queues[queue_index] = queue;
    device->dispatch_queues[device->dispatch_queue_count++] = queue;

    if (!transfer_queue_count) {
      // If we don't have any dedicated transfer queues then use all dispatch
      // queues as transfer queues.
      device->transfer_queues[device->transfer_queue_count++] = queue;
    }

    if (maintenance_dispatch_queue == VK_NULL_HANDLE) {
      maintenance_dispatch_queue = queue->handle();
    }

    if (iree_all_bits_set(enabled_features,
                          IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING)) {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_tracing_context_allocate(
          device->physical_device, device->logical_device, queue->handle(),
          queue_name, maintenance_dispatch_queue, device->dispatch_command_pool,
          device->host_allocator,
          &device->queue_tracing_contexts[queue_index]));
      queue->set_tracing_context(device->queue_tracing_contexts[queue_index]);
    }
  }
  for (iree_host_size_t i = 0; i < transfer_queue_count; ++i) {
    if (!(transfer_queue_set->queue_indices & (1ull << i))) continue;

    char queue_name_buffer[32];
    int queue_name_length =
        snprintf(queue_name_buffer, IREE_ARRAYSIZE(queue_name_buffer),
                 "Vulkan[%c:%d]", 'T', (int)device->transfer_queue_count);
    iree_string_view_t queue_name =
        iree_make_string_view(queue_name_buffer, queue_name_length);

    CommandQueue* queue = iree_hal_vulkan_device_create_queue(
        device->logical_device, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        transfer_queue_set->queue_family_index, i);

    iree_host_size_t queue_index = device->queue_count++;
    device->queues[queue_index] = queue;
    device->transfer_queues[device->transfer_queue_count++] = queue;

    if (iree_all_bits_set(enabled_features,
                          IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING)) {
      IREE_RETURN_IF_ERROR(iree_hal_vulkan_tracing_context_allocate(
          device->physical_device, device->logical_device, queue->handle(),
          queue_name, maintenance_dispatch_queue, device->dispatch_command_pool,
          device->host_allocator,
          &device->queue_tracing_contexts[queue_index]));
      queue->set_tracing_context(device->queue_tracing_contexts[queue_index]);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_vulkan_features_t enabled_features,
    const iree_hal_vulkan_device_options_t* options, VkInstance instance,
    VkPhysicalDevice physical_device, VkDeviceHandle* logical_device,
    const iree_hal_vulkan_device_extensions_t* device_extensions,
    const iree_hal_vulkan_device_properties_t* device_properties,
    const iree_hal_vulkan_queue_set_t* compute_queue_set,
    const iree_hal_vulkan_queue_set_t* transfer_queue_set,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_host_size_t compute_queue_count =
      iree_math_count_ones_u64(compute_queue_set->queue_indices);
  iree_host_size_t transfer_queue_count =
      iree_math_count_ones_u64(transfer_queue_set->queue_indices);
  iree_host_size_t total_queue_count =
      compute_queue_count + transfer_queue_count;

  iree_hal_vulkan_device_t* device = NULL;
  iree_host_size_t total_size =
      sizeof(*device) + identifier.size +
      total_queue_count * sizeof(device->queues[0]) +
      total_queue_count * sizeof(device->dispatch_queues[0]) +
      total_queue_count * sizeof(device->transfer_queues[0]) +
      total_queue_count * sizeof(device->queue_tracing_contexts[0]);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_vulkan_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);
  device->flags = options->flags;

  device->device_extensions = *device_extensions;
  device->device_properties = *device_properties;
  device->instance = instance;
  device->physical_device = physical_device;
  device->logical_device = logical_device;
  device->logical_device->AddReference();

#if defined(IREE_HAL_VULKAN_HAVE_RENDERDOC)
  device->renderdoc_api = iree_hal_vulkan_query_renderdoc_api(instance);
#endif  // IREE_HAL_VULKAN_HAVE_RENDERDOC

  iree_arena_block_pool_initialize(32 * 1024, host_allocator,
                                   &device->block_pool);

  // Point the queue storage into the new device allocation. The queues
  // themselves are populated
  device->queues = (CommandQueue**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->queues[0]);
  device->dispatch_queues = (CommandQueue**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->dispatch_queues[0]);
  device->transfer_queues = (CommandQueue**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->transfer_queues[0]);
  device->queue_tracing_contexts =
      (iree_hal_vulkan_tracing_context_t**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->queue_tracing_contexts[0]);

  device->descriptor_pool_cache =
      new DescriptorPoolCache(device->logical_device);

  // Create the device memory allocator that will service all buffer
  // allocation requests.
  iree_status_t status = iree_hal_vulkan_native_allocator_create(
      options, instance, physical_device, logical_device,
      &device->device_allocator);

  // Create command pools for each queue family. If we don't have a transfer
  // queue then we'll ignore that one and just use the dispatch pool.
  // If we wanted to expose the pools through the HAL to allow the VM to more
  // effectively manage them (pool per fiber, etc) we could, however I doubt
  // the overhead of locking the pool will be even a blip.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_transient_command_pool(
        device->logical_device, compute_queue_set->queue_family_index,
        &device->dispatch_command_pool);
  }
  if (transfer_queue_set->queue_indices != 0 && iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_transient_command_pool(
        device->logical_device, transfer_queue_set->queue_family_index,
        &device->transfer_command_pool);
  }

  // Initialize queues now that we've completed the rest of the device
  // initialization; this happens last as the queues require the pools allocated
  // above.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_initialize_command_queues(
        device, enabled_features, identifier, compute_queue_set,
        transfer_queue_set);
  }

  if (iree_status_is_ok(status)) {
    device->builtin_executables =
        new BuiltinExecutables(device->logical_device);
    status = device->builtin_executables->InitializeExecutables();
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_destroy((iree_hal_device_t*)device);
  }
  return status;
}

static void iree_hal_vulkan_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Drop all command queues. These may wait until idle in their destructor.
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    delete device->queues[i];
    iree_hal_vulkan_tracing_context_free(device->queue_tracing_contexts[i]);
  }

  // Drop command pools now that we know there are no more outstanding command
  // buffers.
  delete device->dispatch_command_pool;
  delete device->transfer_command_pool;

  // Now that no commands are outstanding we can release all resources that may
  // have been in use.
  delete device->builtin_executables;
  delete device->descriptor_pool_cache;

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // All arena blocks should have been returned.
  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  device->logical_device->ReleaseReference();
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_device_query_extensibility_set(
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_extensibility_set_t set, iree::Arena* arena,
    iree_hal_vulkan_string_list_t* out_string_list) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_query_extensibility_set(
      requested_features, set, 0, &out_string_list->count, NULL));
  out_string_list->values = (const char**)arena->AllocateBytes(
      out_string_list->count * sizeof(out_string_list->values[0]));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_query_extensibility_set(
      requested_features, set, out_string_list->count, &out_string_list->count,
      out_string_list->values));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_get_device_properties(
    DynamicSymbols* instance_syms, VkPhysicalDevice physical_device,
    iree_hal_vulkan_device_properties_t* device_properties) {
  memset(device_properties, 0, sizeof(*device_properties));

  VkPhysicalDeviceFeatures2 physical_device_features;
  memset(&physical_device_features, 0, sizeof(physical_device_features));
  physical_device_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

  // + Shader float16 and int8 features.
  VkPhysicalDeviceShaderFloat16Int8Features shader_float16_int8_features;
  memset(&shader_float16_int8_features, 0,
         sizeof(shader_float16_int8_features));
  shader_float16_int8_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  shader_float16_int8_features.pNext = physical_device_features.pNext;
  physical_device_features.pNext = &shader_float16_int8_features;

  // + Shader 8 bit storage features.
  VkPhysicalDevice8BitStorageFeatures supported_8bit_storage_features;
  memset(&supported_8bit_storage_features, 0,
         sizeof(supported_8bit_storage_features));
  supported_8bit_storage_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
  supported_8bit_storage_features.pNext = physical_device_features.pNext;
  physical_device_features.pNext = &supported_8bit_storage_features;

  // + Shader 16 bit storage features.
  VkPhysicalDevice16BitStorageFeatures supported_16bit_storage_features;
  memset(&supported_16bit_storage_features, 0,
         sizeof(supported_16bit_storage_features));
  supported_16bit_storage_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  supported_16bit_storage_features.pNext = physical_device_features.pNext;
  physical_device_features.pNext = &supported_16bit_storage_features;

  // + Shader integer dot product features.
  VkPhysicalDeviceShaderIntegerDotProductFeatures dot_product_features;
  memset(&dot_product_features, 0, sizeof(dot_product_features));
  dot_product_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
  dot_product_features.pNext = physical_device_features.pNext;
  physical_device_features.pNext = &dot_product_features;

  // + Cooperative matrix features.
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_matrix_features;
  memset(&coop_matrix_features, 0, sizeof(coop_matrix_features));
  coop_matrix_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  coop_matrix_features.pNext = physical_device_features.pNext;
  physical_device_features.pNext = &coop_matrix_features;

  instance_syms->vkGetPhysicalDeviceFeatures2(physical_device,
                                              &physical_device_features);

  VkPhysicalDeviceProperties2 physical_device_properties;
  memset(&physical_device_properties, 0, sizeof(physical_device_properties));
  physical_device_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  physical_device_properties.pNext = NULL;

  // + Subgroup properties.
  VkPhysicalDeviceSubgroupProperties subgroup_properties;
  memset(&subgroup_properties, 0, sizeof(subgroup_properties));
  subgroup_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  subgroup_properties.pNext = physical_device_properties.pNext;
  physical_device_properties.pNext = &subgroup_properties;

  // + Shader integer dot product properties.
  VkPhysicalDeviceShaderIntegerDotProductProperties dot_product_properties;
  memset(&dot_product_properties, 0, sizeof(dot_product_properties));
  dot_product_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
  dot_product_properties.pNext = physical_device_properties.pNext;
  physical_device_properties.pNext = &dot_product_properties;

  instance_syms->vkGetPhysicalDeviceProperties2(physical_device,
                                                &physical_device_properties);

  if (shader_float16_int8_features.shaderFloat16) {
    device_properties->compute_float |= 0x1u;
  }
  if (physical_device_features.features.shaderFloat64) {
    device_properties->compute_float |= 0x2u;
  }
  if (shader_float16_int8_features.shaderInt8) {
    device_properties->compute_int |= 0x1u;
  }
  if (physical_device_features.features.shaderInt16) {
    device_properties->compute_int |= 0x2u;
  }
  if (physical_device_features.features.shaderInt64) {
    device_properties->compute_int |= 0x4u;
  }
  if (supported_8bit_storage_features.storageBuffer8BitAccess &&
      supported_8bit_storage_features.uniformAndStorageBuffer8BitAccess) {
    device_properties->storage |= 0x1u;
  }
  if (supported_16bit_storage_features.storageBuffer16BitAccess &&
      supported_16bit_storage_features.uniformAndStorageBuffer16BitAccess) {
    device_properties->storage |= 0x2u;
  }

  if (iree_all_bits_set(subgroup_properties.supportedOperations,
                        VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) {
    device_properties->subgroup |= 0x1u;
  }
  if (iree_all_bits_set(subgroup_properties.supportedOperations,
                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)) {
    device_properties->subgroup |= 0x2u;
  }

  if (dot_product_features.shaderIntegerDotProduct) {
    device_properties->dot_product |= 0x1u;
  }

  if (coop_matrix_features.cooperativeMatrix &&
      instance_syms->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
    uint32_t count = 0;
    IREE_RETURN_IF_ERROR(VK_RESULT_TO_STATUS(
        instance_syms->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
            physical_device, &count, NULL)));
    VkCooperativeMatrixPropertiesKHR* properties =
        (VkCooperativeMatrixPropertiesKHR*)iree_alloca(
            count * sizeof(VkCooperativeMatrixPropertiesKHR));
    IREE_RETURN_IF_ERROR(VK_RESULT_TO_STATUS(
        instance_syms->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
            physical_device, &count, properties)));
    for (uint32_t i = 0; i < count; ++i) {
      VkCooperativeMatrixPropertiesKHR* p = properties + i;
      if (p->AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
          p->BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
        if (p->CType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
          if (p->MSize == 16 && p->NSize == 16 && p->KSize == 16) {
            device_properties->cooperative_matrix |= 0x1u;
          }
        }
      }
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_vulkan_features_t requested_features,
    const iree_hal_vulkan_device_options_t* options,
    iree_hal_vulkan_syms_t* opaque_syms, VkInstance instance,
    VkPhysicalDevice physical_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  DynamicSymbols* instance_syms = (DynamicSymbols*)opaque_syms;

  // Find the extensions we need (or want) that are also available
  // on the device. This will fail when required ones are not present.
  // TODO(benvanik): replace with a real arena.
  iree::Arena arena(128 * 1024);
  iree_hal_vulkan_string_list_t required_extensions;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_query_extensibility_set(
      requested_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED, &arena,
      &required_extensions));
  iree_hal_vulkan_string_list_t optional_extensions;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_query_extensibility_set(
      requested_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL, &arena,
      &optional_extensions));
  iree_hal_vulkan_string_list_t enabled_extensions;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_match_available_device_extensions(
      instance_syms, physical_device, &required_extensions,
      &optional_extensions, &arena, &enabled_extensions));
  iree_hal_vulkan_device_extensions_t enabled_device_extensions =
      iree_hal_vulkan_populate_enabled_device_extensions(&enabled_extensions);

  // Find queue families we will expose as HAL queues.
  iree_hal_vulkan_queue_family_info_t queue_family_info;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_queue_families(
      options, physical_device, instance_syms, &queue_family_info));

  bool has_dedicated_transfer_queues =
      queue_family_info.transfer_queue_count > 0;

  // TODO(benvanik): convert to using the arena.
  // Setup the queue info we'll be using.
  // Each queue here (created from within a family) will map to a HAL queue.
  //
  // Note that we need to handle the case where we have transfer queues that
  // are of the same queue family as the dispatch queues: Vulkan requires that
  // all queues created from the same family are done in the same
  // VkDeviceQueueCreateInfo struct.
  std::vector<VkDeviceQueueCreateInfo> queue_create_info;
  // Reserve space for create infos. Note: must be the maximum used, or else
  // references used below will be invalidated as the vector grows.
  queue_create_info.reserve(2);
  std::vector<float> dispatch_queue_priorities;
  std::vector<float> transfer_queue_priorities;
  queue_create_info.push_back({});
  auto& dispatch_queue_info = queue_create_info.back();
  dispatch_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  dispatch_queue_info.pNext = NULL;
  dispatch_queue_info.flags = 0;
  dispatch_queue_info.queueFamilyIndex = queue_family_info.dispatch_index;
  dispatch_queue_info.queueCount = queue_family_info.dispatch_queue_count;
  if (has_dedicated_transfer_queues) {
    if (queue_family_info.dispatch_index == queue_family_info.transfer_index) {
      dispatch_queue_info.queueCount += queue_family_info.transfer_queue_count;
    } else {
      queue_create_info.push_back({});
      auto& transfer_queue_info = queue_create_info.back();
      transfer_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      transfer_queue_info.pNext = NULL;
      transfer_queue_info.queueFamilyIndex = queue_family_info.transfer_index;
      transfer_queue_info.queueCount = queue_family_info.transfer_queue_count;
      transfer_queue_info.flags = 0;
      transfer_queue_priorities.resize(transfer_queue_info.queueCount);
      transfer_queue_info.pQueuePriorities = transfer_queue_priorities.data();
    }
  }
  dispatch_queue_priorities.resize(dispatch_queue_info.queueCount);
  dispatch_queue_info.pQueuePriorities = dispatch_queue_priorities.data();

  // Collect supported physical device features.
  VkPhysicalDeviceFeatures2 available_features2;
  memset(&available_features2, 0, sizeof(available_features2));
  available_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

  // + Buffer device address features.
  VkPhysicalDeviceBufferDeviceAddressFeatures
      available_buffer_device_address_features;
  memset(&available_buffer_device_address_features, 0,
         sizeof(available_buffer_device_address_features));
  available_buffer_device_address_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  available_buffer_device_address_features.pNext = available_features2.pNext;
  available_features2.pNext = &available_buffer_device_address_features;

  // + Shader 16 bit storage features.
  VkPhysicalDevice16BitStorageFeatures available_16bit_storage_features;
  memset(&available_16bit_storage_features, 0,
         sizeof(available_16bit_storage_features));
  available_16bit_storage_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
  available_16bit_storage_features.pNext = available_features2.pNext;
  available_features2.pNext = &available_16bit_storage_features;

  // + Shader 8 bit storage features.
  VkPhysicalDevice8BitStorageFeatures available_8bit_storage_features;
  memset(&available_8bit_storage_features, 0,
         sizeof(available_8bit_storage_features));
  available_8bit_storage_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
  available_8bit_storage_features.pNext = available_features2.pNext;
  available_features2.pNext = &available_8bit_storage_features;

  // + Shader float16 and int8 features.
  VkPhysicalDeviceShaderFloat16Int8Features
      available_shader_float16_int8_features;
  memset(&available_shader_float16_int8_features, 0,
         sizeof(available_shader_float16_int8_features));
  available_shader_float16_int8_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  available_shader_float16_int8_features.pNext = available_features2.pNext;
  available_features2.pNext = &available_shader_float16_int8_features;

  // + Subgroup matrix features.
  VkPhysicalDeviceSubgroupProperties available_subgroup_properties;
  memset(&available_subgroup_properties, 0,
         sizeof(available_subgroup_properties));
  available_subgroup_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  available_subgroup_properties.pNext = available_features2.pNext;
  available_features2.pNext = &available_subgroup_properties;

  // + Cooperative matrix features.
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR available_coop_matrix_features;
  memset(&available_coop_matrix_features, 0,
         sizeof(available_coop_matrix_features));
  available_coop_matrix_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  available_coop_matrix_features.pNext = available_features2.pNext;
  available_features2.pNext = &available_coop_matrix_features;

  instance_syms->vkGetPhysicalDeviceFeatures2(physical_device,
                                              &available_features2);
  const VkPhysicalDeviceFeatures* available_features =
      &available_features2.features;

  // Create device and its queues.
  VkDeviceCreateInfo device_create_info;
  memset(&device_create_info, 0, sizeof(device_create_info));
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.enabledLayerCount = 0;
  device_create_info.ppEnabledLayerNames = NULL;
  device_create_info.enabledExtensionCount = enabled_extensions.count;
  device_create_info.ppEnabledExtensionNames = enabled_extensions.values;
  device_create_info.queueCreateInfoCount = queue_create_info.size();
  device_create_info.pQueueCreateInfos = queue_create_info.data();
  device_create_info.pEnabledFeatures = NULL;

  VkPhysicalDeviceFeatures2 enabled_features2;
  memset(&enabled_features2, 0, sizeof(enabled_features2));
  enabled_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  device_create_info.pNext = &enabled_features2;
  if (available_features->shaderInt64) {
    enabled_features2.features.shaderInt64 = VK_TRUE;
  }
  if (available_features->shaderInt16) {
    enabled_features2.features.shaderInt16 = VK_TRUE;
  }

  iree_hal_vulkan_features_t enabled_features = 0;

  IREE_TRACE({
    if (iree_all_bits_set(requested_features,
                          IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING)) {
      enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING;
    }
  });

  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING) &&
      available_features->sparseBinding) {
    enabled_features2.features.sparseBinding = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING;
  }
  if (iree_all_bits_set(
          requested_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED) &&
      available_features->sparseResidencyBuffer &&
      available_features->sparseResidencyAliased) {
    enabled_features2.features.sparseResidencyBuffer = VK_TRUE;
    enabled_features2.features.sparseResidencyAliased = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED;
  }

  if (iree_all_bits_set(requested_features,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS)) {
    if (available_features->robustBufferAccess != VK_TRUE) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "robust buffer access not supported by physical device");
    }
    enabled_features2.features.robustBufferAccess = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS;
  }

  VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features;
  if (iree_all_bits_set(
          requested_features,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES) &&
      available_buffer_device_address_features.bufferDeviceAddress) {
    memset(&buffer_device_address_features, 0,
           sizeof(buffer_device_address_features));
    buffer_device_address_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    buffer_device_address_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &buffer_device_address_features;
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
    enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES;
  }

  VkPhysicalDeviceTimelineSemaphoreFeatures semaphore_features;
  memset(&semaphore_features, 0, sizeof(semaphore_features));
  semaphore_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  semaphore_features.pNext = enabled_features2.pNext;
  enabled_features2.pNext = &semaphore_features;
  semaphore_features.timelineSemaphore = VK_TRUE;

  VkPhysicalDeviceHostQueryResetFeaturesEXT host_query_reset_features;
  if (enabled_device_extensions.host_query_reset) {
    memset(&host_query_reset_features, 0, sizeof(host_query_reset_features));
    host_query_reset_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT;
    host_query_reset_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &host_query_reset_features;
    host_query_reset_features.hostQueryReset = VK_TRUE;
  }

  VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_control_features;
  if (enabled_device_extensions.subgroup_size_control) {
    memset(&subgroup_control_features, 0, sizeof(subgroup_control_features));
    subgroup_control_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    subgroup_control_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &subgroup_control_features;
    subgroup_control_features.subgroupSizeControl = VK_TRUE;
  }

  // Enable all available 16- or 8-bit integer/floating-point features.
  available_16bit_storage_features.pNext = enabled_features2.pNext;
  enabled_features2.pNext = &available_16bit_storage_features;
  if (enabled_device_extensions.shader_8bit_storage) {
    available_8bit_storage_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &available_8bit_storage_features;
  }
  if (enabled_device_extensions.shader_float16_int8) {
    available_shader_float16_int8_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &available_shader_float16_int8_features;
  }

  // Enable all available cooperative matrix features.
  if (enabled_device_extensions.cooperative_matrix) {
    available_coop_matrix_features.pNext = enabled_features2.pNext;
    enabled_features2.pNext = &available_coop_matrix_features;
  }

  iree_hal_vulkan_device_properties_t device_properties;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_get_device_properties(
      instance_syms, physical_device, &device_properties));

  auto logical_device = new VkDeviceHandle(
      instance_syms, physical_device, enabled_features,
      enabled_device_extensions, device_properties,
      /*owns_device=*/true, host_allocator, /*allocator=*/NULL);

  iree_status_t status = VK_RESULT_TO_STATUS(
      instance_syms->vkCreateDevice(physical_device, &device_create_info,
                                    logical_device->allocator(),
                                    logical_device->mutable_value()),
      "vkCreateDevice");
  if (iree_status_is_ok(status)) {
    status = logical_device->syms()->LoadFromDevice(instance,
                                                    logical_device->value());
  }

  // Select queue indices and create command queues with them.
  iree_hal_vulkan_queue_set_t compute_queue_set;
  iree_hal_vulkan_queue_set_t transfer_queue_set;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_build_queue_sets(
        options, physical_device, logical_device->syms().get(),
        &compute_queue_set, &transfer_queue_set);
  }

  // Allocate and initialize the device.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_create_internal(
        driver, identifier, enabled_features, options, instance,
        physical_device, logical_device, &enabled_device_extensions,
        &device_properties, &compute_queue_set, &transfer_queue_set,
        host_allocator, out_device);
  }

  logical_device->ReleaseReference();
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_wrap_device(
    iree_string_view_t identifier,
    const iree_hal_vulkan_device_options_t* options,
    const iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    VkPhysicalDevice physical_device, VkDevice logical_device,
    const iree_hal_vulkan_queue_set_t* compute_queue_set,
    const iree_hal_vulkan_queue_set_t* transfer_queue_set,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(instance_syms);
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_device);

  if (iree_math_count_ones_u64(compute_queue_set->queue_indices) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one compute queue is required");
  }

  // Grab symbols from the device.
  auto device_syms = iree::make_ref<DynamicSymbols>();
  device_syms->vkGetInstanceProcAddr =
      ((const DynamicSymbols*)instance_syms)->vkGetInstanceProcAddr;
  IREE_RETURN_IF_ERROR(device_syms->LoadFromDevice(instance, logical_device));

  // Since the device is already created, we can't actually enable any
  // extensions or query if they are really enabled - we just have to trust
  // that the caller already enabled them for us or we may fail later. For the
  // optional extensions we check for the symbols but this is not always
  // guaranteed to work.
  iree_hal_vulkan_device_extensions_t enabled_device_extensions =
      iree_hal_vulkan_infer_enabled_device_extensions(device_syms.get());

  // We can still retrieve the correct device properties though.
  iree_hal_vulkan_device_properties_t device_properties;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_get_device_properties(
      device_syms.get(), physical_device, &device_properties));

  iree_hal_vulkan_features_t enabled_features = 0;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  enabled_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  // Wrap the provided VkDevice with a VkDeviceHandle for use within the HAL.
  auto logical_device_handle = new VkDeviceHandle(
      device_syms.get(), physical_device, enabled_features,
      enabled_device_extensions, device_properties,
      /*owns_device=*/false, host_allocator, /*allocator=*/NULL);
  *logical_device_handle->mutable_value() = logical_device;

  // Allocate and initialize the device.
  iree_status_t status = iree_hal_vulkan_device_create_internal(
      /*driver=*/NULL, identifier, enabled_features, options, instance,
      physical_device, logical_device_handle, &enabled_device_extensions,
      &device_properties, compute_queue_set, transfer_queue_set, host_allocator,
      out_device);

  logical_device_handle->ReleaseReference();
  return status;
}

static iree_string_view_t iree_hal_vulkan_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_vulkan_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_vulkan_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_vulkan_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_vulkan_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_vulkan_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_vulkan_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    if (iree_string_view_equal(key, IREE_SV("vulkan-spirv-fb"))) {
      // Base SPIR-V always supported.
      *out_value = 1;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("vulkan-spirv-fb-ptr"))) {
      // SPIR-V with device addresses is optionally supported based on whether
      // we have device feature support.
      *out_value = iree_all_bits_set(
                       device->logical_device->enabled_features(),
                       IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)
                       ? 1
                       : 0;
      return iree_ok_status();
    }
  }

  // Note that the device queries used here should match the ones used in
  // buildDeviceQueryRegion() on the compiler side.
  if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("compute.bitwidths.fp"))) {
      *out_value = device->logical_device->supported_properties().compute_float;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("compute.bitwidths.int"))) {
      *out_value = device->logical_device->supported_properties().compute_int;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("storage.bitwidths"))) {
      *out_value = device->logical_device->supported_properties().storage;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("subgroup.ops"))) {
      *out_value = device->logical_device->supported_properties().subgroup;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("dotprod.ops"))) {
      *out_value = device->logical_device->supported_properties().dot_product;
      return iree_ok_status();
    }
    if (iree_string_view_equal(key, IREE_SV("coopmatrix.ops"))) {
      *out_value =
          device->logical_device->supported_properties().cooperative_matrix;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

// Returns the queue to submit work to based on the |queue_affinity|.
static CommandQueue* iree_hal_vulkan_device_select_queue(
    iree_hal_vulkan_device_t* device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity) {
  // TODO(scotttodd): revisit queue selection logic and remove this
  //   * the unaligned buffer fill polyfill and tracing timestamp queries may
  //     both insert dispatches into command buffers that at compile time are
  //     expected to only contain transfer commands
  //   * we could set a bit at recording time if emulation or tracing is used
  //     and submit to the right queue based on that
  command_categories |= IREE_HAL_COMMAND_CATEGORY_DISPATCH;

  // TODO(benvanik): meaningful heuristics for affinity. We don't generate
  // anything from the compiler that uses multiple queues and until we do it's
  // best not to do anything too clever here.
  if (command_categories == IREE_HAL_COMMAND_CATEGORY_TRANSFER) {
    return device
        ->transfer_queues[queue_affinity % device->transfer_queue_count];
  }
  return device->dispatch_queues[queue_affinity % device->dispatch_queue_count];
}

static iree_status_t iree_hal_vulkan_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_vulkan_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);

  // TODO(scotttodd): revisit queue selection logic and remove this
  //   * the unaligned buffer fill polyfill and tracing timestamp queries may
  //     both insert dispatches into command buffers that at compile time are
  //     expected to only contain transfer commands
  //   * we could set a bit at recording time if emulation or tracing is used
  //     and submit to the right queue based on that
  command_categories |= IREE_HAL_COMMAND_CATEGORY_DISPATCH;

  // Select the command pool to used based on the types of commands used.
  // Note that we may not have a dedicated transfer command pool if there are
  // no dedicated transfer queues.
  VkCommandPoolHandle* command_pool = NULL;
  if (device->transfer_command_pool &&
      !iree_all_bits_set(command_categories,
                         IREE_HAL_COMMAND_CATEGORY_DISPATCH)) {
    command_pool = device->transfer_command_pool;
  } else {
    command_pool = device->dispatch_command_pool;
  }

  // The tracing context is tied to a particular queue so we must select here
  // even though ideally we'd do it during submission. This is informational
  // only and if the user does provide a different queue affinity during
  // submission it just means the commands will be attributed to the wrong
  // queue.
  CommandQueue* queue = iree_hal_vulkan_device_select_queue(
      device, command_categories, queue_affinity);

  return iree_hal_vulkan_direct_command_buffer_allocate(
      base_device, device->logical_device, command_pool, mode,
      command_categories, queue_affinity, binding_capacity,
      queue->tracing_context(), device->descriptor_pool_cache,
      device->builtin_executables, &device->block_pool, out_command_buffer);
}

static iree_status_t iree_hal_vulkan_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_descriptor_set_layout_create(
      device->logical_device, flags, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_vulkan_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_event_create(device->logical_device, out_event);
}

static iree_status_t iree_hal_vulkan_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_nop_executable_cache_create(
      device->logical_device, identifier, out_executable_cache);
}

static iree_status_t iree_hal_vulkan_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_vulkan_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_pipeline_layout_create(
      device->logical_device, push_constants, set_layout_count, set_layouts,
      out_pipeline_layout);
}

static iree_status_t iree_hal_vulkan_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_semaphore_create(device->logical_device,
                                                 initial_value, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_vulkan_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  if (iree_hal_vulkan_native_semaphore_isa(semaphore)) {
    // Fast-path for semaphores related to this device.
    // TODO(benvanik): ensure the creating devices are compatible in cases where
    // multiple devices are used.
    return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
  }
  // TODO(benvanik): semaphore APIs for querying allowed export formats. We
  // can check device caps to see what external semaphore types are supported.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_vulkan_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  // TODO(benvanik): queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      /*.loop=*/iree_loop_inline(&loop_status),
      /*.chunk_count=*/IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      /*.chunk_size=*/IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_vulkan_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      /*.loop=*/iree_loop_inline(&loop_status),
      /*.chunk_count=*/IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      /*.chunk_size=*/IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_vulkan_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  // NOTE: today we are not discriminating queues based on command type.
  CommandQueue* queue = iree_hal_vulkan_device_select_queue(
      device, IREE_HAL_COMMAND_CATEGORY_DISPATCH, queue_affinity);
  iree_hal_submission_batch_t batch = {
      /*.wait_semaphores=*/wait_semaphore_list,
      /*.command_buffer_count=*/command_buffer_count,
      /*.command_buffers=*/command_buffers,
      /*.signal_semaphores=*/signal_semaphore_list,
  };
  IREE_RETURN_IF_ERROR(queue->Submit(1, &batch));
  // HACK: we don't track async resource lifetimes so we have to block.
  return iree_hal_semaphore_list_wait(signal_semaphore_list,
                                      iree_infinite_timeout());
}

static iree_status_t iree_hal_vulkan_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  VkSemaphoreWaitFlags wait_flags = 0;
  if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
    wait_flags |= VK_SEMAPHORE_WAIT_ANY_BIT;
  }
  return iree_hal_vulkan_native_semaphore_multi_wait(
      device->logical_device, &semaphore_list, timeout, wait_flags);
}

static iree_status_t iree_hal_vulkan_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  (void)device;

  if (iree_all_bits_set(options->mode,
                        IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS)) {
    // AMD-specific - we could snoop the device to only do this for the vendor
    // but this is relatively cheap and could be useful to others. Ideally
    // there would be a khronos standard for this.
    // TODO(benvanik): figure out if we need to do this for all queues.
    auto& syms = device->logical_device->syms();
    if (syms->vkQueueInsertDebugUtilsLabelEXT) {
      VkDebugUtilsLabelEXT begin_label = {};
      begin_label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
      begin_label.pNext = NULL;
      begin_label.pLabelName = "AmdFrameBegin";
      device->logical_device->syms()->vkQueueInsertDebugUtilsLabelEXT(
          device->dispatch_queues[0]->handle(), &begin_label);
    }

    // For now we only support RenderDoc. As much as possible we should try to
    // use standardized Vulkan layers to do profiling configuration/control like
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_performance_query.html
    // to avoid the combinatorial explosion of vendor tooling hooks.
    // Since RenderDoc is fairly simple, cross-platform, and cross-vendor we
    // support it here. If this grows beyond a few lines of code we should
    // shuffle it off to another file.
#if defined(IREE_HAL_VULKAN_HAVE_RENDERDOC)
    iree_hal_vulkan_begin_renderdoc_capture(device->renderdoc_api,
                                            device->instance, options);
#endif  // IREE_HAL_VULKAN_HAVE_RENDERDOC
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  (void)device;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
  if (iree_all_bits_set(device->logical_device->enabled_features(),
                        IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING)) {
    for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
      iree_hal_vulkan_tracing_context_t* tracing_context =
          device->queues[i]->tracing_context();
      if (tracing_context) {
        iree_hal_vulkan_tracing_context_collect(tracing_context,
                                                VK_NULL_HANDLE);
      }
    }
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  (void)device;

#if defined(IREE_HAL_VULKAN_HAVE_RENDERDOC)
  iree_hal_vulkan_end_renderdoc_capture(device->renderdoc_api,
                                        device->instance);
#endif  // IREE_HAL_VULKAN_HAVE_RENDERDOC

  // AMD-specific.
  auto& syms = device->logical_device->syms();
  if (syms->vkQueueInsertDebugUtilsLabelEXT) {
    VkDebugUtilsLabelEXT end_label = {};
    end_label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    end_label.pNext = NULL;
    end_label.pLabelName = "AmdFrameEnd";
    device->logical_device->syms()->vkQueueInsertDebugUtilsLabelEXT(
        device->dispatch_queues[0]->handle(), &end_label);
  }

  return iree_ok_status();
}

namespace {
const iree_hal_device_vtable_t iree_hal_vulkan_device_vtable = {
    /*.destroy=*/iree_hal_vulkan_device_destroy,
    /*.id=*/iree_hal_vulkan_device_id,
    /*.host_allocator=*/iree_hal_vulkan_device_host_allocator,
    /*.device_allocator=*/iree_hal_vulkan_device_allocator,
    /*.replace_device_allocator=*/iree_hal_vulkan_replace_device_allocator,
    /*.replace_channel_provider=*/iree_hal_vulkan_replace_channel_provider,
    /*.trim=*/iree_hal_vulkan_device_trim,
    /*.query_i64=*/iree_hal_vulkan_device_query_i64,
    /*.create_channel=*/iree_hal_vulkan_device_create_channel,
    /*.create_command_buffer=*/iree_hal_vulkan_device_create_command_buffer,
    /*.create_descriptor_set_layout=*/
    iree_hal_vulkan_device_create_descriptor_set_layout,
    /*.create_event=*/iree_hal_vulkan_device_create_event,
    /*.create_executable_cache=*/
    iree_hal_vulkan_device_create_executable_cache,
    /*.import_file=*/iree_hal_vulkan_device_import_file,
    /*.create_pipeline_layout=*/
    iree_hal_vulkan_device_create_pipeline_layout,
    /*.create_semaphore=*/iree_hal_vulkan_device_create_semaphore,
    /*.query_semaphore_compatibility=*/
    iree_hal_vulkan_device_query_semaphore_compatibility,
    /*.queue_alloca=*/iree_hal_vulkan_device_queue_alloca,
    /*.queue_dealloca=*/iree_hal_vulkan_device_queue_dealloca,
    /*.queue_read=*/iree_hal_vulkan_device_queue_read,
    /*.queue_write=*/iree_hal_vulkan_device_queue_write,
    /*.queue_execute=*/iree_hal_vulkan_device_queue_execute,
    /*.queue_flush=*/iree_hal_vulkan_device_queue_flush,
    /*.wait_semaphores=*/iree_hal_vulkan_device_wait_semaphores,
    /*.profiling_begin=*/iree_hal_vulkan_device_profiling_begin,
    /*.profiling_flush=*/iree_hal_vulkan_device_profiling_flush,
    /*.profiling_end=*/iree_hal_vulkan_device_profiling_end,
};
}  // namespace
