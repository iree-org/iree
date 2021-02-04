// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/vulkan/vulkan_device.h"

#include <functional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "iree/base/internal/math.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/api.h"
#include "iree/hal/vulkan/command_queue.h"
#include "iree/hal/vulkan/descriptor_pool_cache.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/direct_command_queue.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/emulated_semaphore.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/handle_util.h"
#include "iree/hal/vulkan/native_descriptor_set.h"
#include "iree/hal/vulkan/native_descriptor_set_layout.h"
#include "iree/hal/vulkan/native_event.h"
#include "iree/hal/vulkan/native_executable_layout.h"
#include "iree/hal/vulkan/native_semaphore.h"
#include "iree/hal/vulkan/nop_executable_cache.h"
#include "iree/hal/vulkan/serializing_command_queue.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_allocator.h"

using namespace iree::hal::vulkan;

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_device_t extensibility util
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_vulkan_query_extensibility_set(
    iree_hal_vulkan_features_t requested_features,
    iree_hal_vulkan_extensibility_set_t set, iree_host_size_t string_capacity,
    const char** out_string_values, iree_host_size_t* out_string_count) {
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

  //===--------------------------------------------------------------------===//
  // Vulkan forward-compatibility shims
  //===--------------------------------------------------------------------===//
  // These are shims or extensions that are made core later in the spec and can
  // be removed once we require the core version that contains them.

  // VK_KHR_timeline_semaphore:
  // timeline semaphore support is optional and will be emulated if necessary.
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
          VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);

  // VK_LAYER_KHRONOS_timeline_semaphore:
  // polyfill layer - enable if present instead of our custom emulation. Ignored
  // if timeline semaphores are supported natively (Vulkan 1.2+).
  ADD_EXT(IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL,
          "VK_LAYER_KHRONOS_timeline_semaphore");

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

  *out_string_count = string_count;
  return status;
}

//===----------------------------------------------------------------------===//
// Queue selection
//===----------------------------------------------------------------------===//

#define IREE_HAL_VULKAN_INVALID_QUEUE_FAMILY_INDEX (-1)

typedef struct {
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

  // Try to find a dedicated compute queue (no graphics caps).
  // Some may support both transfer and compute. If that fails then fallback
  // to any queue that supports compute.
  out_family_info->dispatch_index =
      iree_hal_vulkan_find_first_queue_family_with_flags(
          queue_family_count, queue_family_properties, VK_QUEUE_COMPUTE_BIT,
          VK_QUEUE_GRAPHICS_BIT);
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
    VkPhysicalDevice physical_device, iree::hal::vulkan::DynamicSymbols* syms,
    iree_hal_vulkan_queue_set_t* out_compute_queue_set,
    iree_hal_vulkan_queue_set_t* out_transfer_queue_set) {
  // Select which queues to use (and fail the implementation can't handle them).
  iree_hal_vulkan_queue_family_info_t queue_family_info;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_select_queue_families(
      physical_device, syms, &queue_family_info));

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

typedef struct {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Optional driver that owns the instance. We retain it for our lifetime to
  // ensure the instance remains valid.
  iree_hal_driver_t* driver;

  // Flags overriding default device behavior.
  iree_hal_vulkan_device_flags_t flags;
  // Which optional extensions are active and available on the device.
  iree_hal_vulkan_device_extensions_t device_extensions;

  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDeviceHandle* logical_device;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

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

  DescriptorPoolCache* descriptor_pool_cache;

  VkCommandPoolHandle* dispatch_command_pool;
  VkCommandPoolHandle* transfer_command_pool;

  // Used only for emulated timeline semaphores.
  TimePointSemaphorePool* semaphore_pool;
  TimePointFencePool* fence_pool;
} iree_hal_vulkan_device_t;

extern const iree_hal_device_vtable_t iree_hal_vulkan_device_vtable;

static iree_hal_vulkan_device_t* iree_hal_vulkan_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_device_vtable);
  return (iree_hal_vulkan_device_t*)base_value;
}

IREE_API_EXPORT void IREE_API_CALL iree_hal_vulkan_device_options_initialize(
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
  create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
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
    uint32_t queue_index, TimePointFencePool* fence_pool) {
  VkQueue queue = VK_NULL_HANDLE;
  logical_device->syms()->vkGetDeviceQueue(*logical_device, queue_family_index,
                                           queue_index, &queue);
  std::string queue_name;
  if (!iree_all_bits_set(command_category,
                         IREE_HAL_COMMAND_CATEGORY_DISPATCH)) {
    queue_name = "q(t):";
  } else {
    queue_name = "q(d):";
  }
  queue_name += std::to_string(queue_index);

  // When emulating timeline semaphores we use a special queue that allows us to
  // sequence the semaphores correctly.
  if (fence_pool != NULL) {
    return new SerializingCommandQueue(logical_device, std::move(queue_name),
                                       command_category, queue, fence_pool);
  }

  return new DirectCommandQueue(logical_device, std::move(queue_name),
                                command_category, queue);
}

// Creates command queues for the given sets of queues and populates the
// device queue lists.
static void iree_hal_vulkan_device_initialize_command_queues(
    iree_hal_vulkan_device_t* device, iree_string_view_t queue_prefix,
    const iree_hal_vulkan_queue_set_t* compute_queue_set,
    const iree_hal_vulkan_queue_set_t* transfer_queue_set) {
  device->queue_count = 0;
  device->dispatch_queue_count = 0;
  device->transfer_queue_count = 0;

  uint64_t compute_queue_count =
      iree_math_count_ones_u64(compute_queue_set->queue_indices);
  uint64_t transfer_queue_count =
      iree_math_count_ones_u64(transfer_queue_set->queue_indices);
  for (iree_host_size_t i = 0; i < compute_queue_count; ++i) {
    if (!(compute_queue_set->queue_indices & (1ull << i))) continue;
    CommandQueue* queue = iree_hal_vulkan_device_create_queue(
        device->logical_device, IREE_HAL_COMMAND_CATEGORY_ANY,
        compute_queue_set->queue_family_index, i, device->fence_pool);
    device->queues[device->queue_count++] = queue;
    device->dispatch_queues[device->dispatch_queue_count++] = queue;
    if (!transfer_queue_count) {
      // If we don't have any dedicated transfer queues then use all dispatch
      // queues as transfer queues.
      device->transfer_queues[device->transfer_queue_count++] = queue;
    }
  }
  for (iree_host_size_t i = 0; i < transfer_queue_count; ++i) {
    if (!(transfer_queue_set->queue_indices & (1ull << i))) continue;
    CommandQueue* queue = iree_hal_vulkan_device_create_queue(
        device->logical_device, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        transfer_queue_set->queue_family_index, i, device->fence_pool);
    device->queues[device->queue_count++] = queue;
    device->transfer_queues[device->transfer_queue_count++] = queue;
  }
}

static iree_status_t iree_hal_vulkan_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_vulkan_device_options_t* options, VkInstance instance,
    VkPhysicalDevice physical_device, VkDeviceHandle* logical_device,
    const iree_hal_vulkan_device_extensions_t* device_extensions,
    const iree_hal_vulkan_queue_set_t* compute_queue_set,
    const iree_hal_vulkan_queue_set_t* transfer_queue_set,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  auto& device_syms = logical_device->syms();

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
      total_queue_count * sizeof(device->transfer_queues[0]);
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
  device->instance = instance;
  device->physical_device = physical_device;
  device->logical_device = logical_device;
  device->logical_device->AddReference();

  // Point the queue storage into the new device allocation. The queues
  // themselves are populated
  device->queues = (CommandQueue**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->queues[0]);
  device->dispatch_queues = (CommandQueue**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->dispatch_queues[0]);
  device->transfer_queues = (CommandQueue**)buffer_ptr;
  buffer_ptr += total_queue_count * sizeof(device->transfer_queues[0]);

  device->descriptor_pool_cache =
      new DescriptorPoolCache(device->logical_device);

  // Create the device memory allocator that will service all buffer
  // allocation requests.
  VmaRecordSettings vma_record_settings;
  memset(&vma_record_settings, 0, sizeof(vma_record_settings));
  iree_status_t status = iree_hal_vulkan_vma_allocator_create(
      instance, physical_device, logical_device, vma_record_settings,
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

  // Emulate timeline semaphores when the extension is not available and we are
  // ony Vulkan versions prior to 1.2 when they were made core.
  bool emulate_timeline_semaphores =
      device_syms->vkGetSemaphoreCounterValue == NULL ||
      iree_all_bits_set(
          options->flags,
          IREE_HAL_VULKAN_DEVICE_FORCE_TIMELINE_SEMAPHORE_EMULATION);
  if (emulate_timeline_semaphores && iree_status_is_ok(status)) {
    status = TimePointSemaphorePool::Create(device->logical_device,
                                            &device->semaphore_pool);
  }
  if (emulate_timeline_semaphores && iree_status_is_ok(status)) {
    status =
        TimePointFencePool::Create(device->logical_device, &device->fence_pool);
  }

  // Initialize queues now that we've completed the rest of the device
  // initialization; this happens last as the queues require the pools allocated
  // above.
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_device_initialize_command_queues(
        device, identifier, compute_queue_set, transfer_queue_set);
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
  }

  // Drop command pools now that we know there are no more outstanding command
  // buffers.
  delete device->dispatch_command_pool;
  delete device->transfer_command_pool;

  // Now that no commands are outstanding we can release all resources that may
  // have been in use.
  delete device->descriptor_pool_cache;
  delete device->semaphore_pool;
  delete device->fence_pool;

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

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
      requested_features, set, 0, NULL, &out_string_list->count));
  out_string_list->values = (const char**)arena->AllocateBytes(
      out_string_list->count * sizeof(out_string_list->values[0]));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_query_extensibility_set(
      requested_features, set, out_string_list->count, out_string_list->values,
      &out_string_list->count));
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_vulkan_features_t enabled_features,
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
      enabled_features,
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED, &arena,
      &required_extensions));
  iree_hal_vulkan_string_list_t optional_extensions;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_query_extensibility_set(
      enabled_features,
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
      physical_device, instance_syms, &queue_family_info));

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
  absl::InlinedVector<VkDeviceQueueCreateInfo, 2> queue_create_info;
  absl::InlinedVector<float, 4> dispatch_queue_priorities;
  absl::InlinedVector<float, 4> transfer_queue_priorities;
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

  VkPhysicalDeviceFeatures2 features2;
  memset(&features2, 0, sizeof(features2));
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  device_create_info.pNext = &features2;

  VkPhysicalDeviceTimelineSemaphoreFeatures semaphore_features;
  bool emulate_timeline_semaphores =
      !enabled_device_extensions.timeline_semaphore ||
      iree_all_bits_set(
          options->flags,
          IREE_HAL_VULKAN_DEVICE_FORCE_TIMELINE_SEMAPHORE_EMULATION);
  if (!emulate_timeline_semaphores) {
    memset(&semaphore_features, 0, sizeof(semaphore_features));
    semaphore_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    semaphore_features.pNext = features2.pNext;
    features2.pNext = &semaphore_features;
    semaphore_features.timelineSemaphore = VK_TRUE;
  }

  auto logical_device = new VkDeviceHandle(
      instance_syms, enabled_device_extensions,
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
        physical_device, logical_device->syms().get(), &compute_queue_set,
        &transfer_queue_set);
  }

  // Allocate and initialize the device.
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_device_create_internal(
        driver, identifier, options, instance, physical_device, logical_device,
        &enabled_device_extensions, &compute_queue_set, &transfer_queue_set,
        host_allocator, out_device);
  }

  logical_device->ReleaseReference();
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_vulkan_wrap_device(
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

  // Wrap the provided VkDevice with a VkDeviceHandle for use within the HAL.
  auto logical_device_handle = new VkDeviceHandle(
      device_syms.get(), enabled_device_extensions,
      /*owns_device=*/false, host_allocator, /*allocator=*/NULL);
  *logical_device_handle->mutable_value() = logical_device;

  // Allocate and initialize the device.
  iree_status_t status = iree_hal_vulkan_device_create_internal(
      /*driver=*/NULL, identifier, options, instance, physical_device,
      logical_device_handle, &enabled_device_extensions, compute_queue_set,
      transfer_queue_set, host_allocator, out_device);

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

static iree_status_t iree_hal_vulkan_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);

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

  return iree_hal_vulkan_direct_command_buffer_allocate(
      device->logical_device, command_pool, mode, command_categories,
      device->descriptor_pool_cache, out_command_buffer);
}

static iree_status_t iree_hal_vulkan_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  // TODO(benvanik): rework the create fn to take the bindings.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "non-push descriptor sets still need work");
}

static iree_status_t iree_hal_vulkan_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_descriptor_set_layout_create(
      device->logical_device, usage_type, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_vulkan_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_event_create(device->logical_device, out_event);
}

static iree_status_t iree_hal_vulkan_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_nop_executable_cache_create(
      device->logical_device, identifier, out_executable_cache);
}

static iree_status_t iree_hal_vulkan_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constants,
    iree_hal_executable_layout_t** out_executable_layout) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  return iree_hal_vulkan_native_executable_layout_create(
      device->logical_device, set_layout_count, set_layouts, push_constants,
      out_executable_layout);
}

static iree_status_t iree_hal_vulkan_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  if (device->semaphore_pool != NULL) {
    return iree_hal_vulkan_emulated_semaphore_create(
        device->logical_device, device->semaphore_pool, device->queue_count,
        device->queues, initial_value, out_semaphore);
  }
  return iree_hal_vulkan_native_semaphore_create(device->logical_device,
                                                 initial_value, out_semaphore);
}

// Returns the queue to submit work to based on the |queue_affinity|.
static CommandQueue* iree_hal_vulkan_device_select_queue(
    iree_hal_vulkan_device_t* device,
    iree_hal_command_category_t command_categories, uint64_t queue_affinity) {
  // TODO(benvanik): meaningful heuristics for affinity. We don't generate
  // anything from the compiler that uses multiple queues and until we do it's
  // best not to do anything too clever here.
  if (command_categories == IREE_HAL_COMMAND_CATEGORY_TRANSFER) {
    return device
        ->transfer_queues[queue_affinity % device->transfer_queue_count];
  }
  return device->dispatch_queues[queue_affinity % device->dispatch_queue_count];
}

static iree_status_t iree_hal_vulkan_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories, uint64_t queue_affinity,
    iree_host_size_t batch_count, const iree_hal_submission_batch_t* batches) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  CommandQueue* queue = iree_hal_vulkan_device_select_queue(
      device, command_categories, queue_affinity);
  return queue->Submit(batch_count, batches);
}

static iree_status_t iree_hal_vulkan_device_wait_semaphores_with_deadline(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  VkSemaphoreWaitFlags wait_flags = 0;
  if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
    wait_flags |= VK_SEMAPHORE_WAIT_ANY_BIT;
  }
  if (device->semaphore_pool != NULL) {
    return iree_hal_vulkan_emulated_semaphore_multi_wait(
        device->logical_device, semaphore_list, deadline_ns, wait_flags);
  }
  return iree_hal_vulkan_native_semaphore_multi_wait(
      device->logical_device, semaphore_list, deadline_ns, wait_flags);
}

static iree_status_t iree_hal_vulkan_device_wait_semaphores_with_timeout(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list,
    iree_duration_t timeout_ns) {
  return iree_hal_vulkan_device_wait_semaphores_with_deadline(
      base_device, wait_mode, semaphore_list,
      iree_relative_timeout_to_deadline_ns(timeout_ns));
}

static iree_status_t iree_hal_vulkan_device_wait_idle_with_deadline(
    iree_hal_device_t* base_device, iree_time_t deadline_ns) {
  iree_hal_vulkan_device_t* device = iree_hal_vulkan_device_cast(base_device);
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    // Fast path for using vkDeviceWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    return VK_RESULT_TO_STATUS(device->logical_device->syms()->vkDeviceWaitIdle(
                                   *device->logical_device),
                               "vkDeviceWaitIdle");
  }
  for (iree_host_size_t i = 0; i < device->queue_count; ++i) {
    IREE_RETURN_IF_ERROR(device->queues[i]->WaitIdle(deadline_ns));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_device_wait_idle_with_timeout(
    iree_hal_device_t* base_device, iree_duration_t timeout_ns) {
  return iree_hal_vulkan_device_wait_idle_with_deadline(
      base_device, iree_relative_timeout_to_deadline_ns(timeout_ns));
}

const iree_hal_device_vtable_t iree_hal_vulkan_device_vtable = {
    /*.destroy=*/iree_hal_vulkan_device_destroy,
    /*.id=*/iree_hal_vulkan_device_id,
    /*.host_allocator=*/iree_hal_vulkan_device_host_allocator,
    /*.device_allocator=*/iree_hal_vulkan_device_allocator,
    /*.create_command_buffer=*/iree_hal_vulkan_device_create_command_buffer,
    /*.create_descriptor_set=*/iree_hal_vulkan_device_create_descriptor_set,
    /*.create_descriptor_set_layout=*/
    iree_hal_vulkan_device_create_descriptor_set_layout,
    /*.create_event=*/iree_hal_vulkan_device_create_event,
    /*.create_executable_cache=*/
    iree_hal_vulkan_device_create_executable_cache,
    /*.create_executable_layout=*/
    iree_hal_vulkan_device_create_executable_layout,
    /*.create_semaphore=*/iree_hal_vulkan_device_create_semaphore,
    /*.queue_submit=*/iree_hal_vulkan_device_queue_submit,
    /*.wait_semaphores_with_deadline=*/
    iree_hal_vulkan_device_wait_semaphores_with_deadline,
    /*.wait_semaphores_with_timeout=*/
    iree_hal_vulkan_device_wait_semaphores_with_timeout,
    /*.wait_idle_with_deadline=*/
    iree_hal_vulkan_device_wait_idle_with_deadline,
    /*.wait_idle_with_timeout=*/
    iree_hal_vulkan_device_wait_idle_with_timeout,
};
