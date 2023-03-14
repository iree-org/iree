// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/status_util.h"

iree_status_t iree_hal_vulkan_result_to_status(VkResult result,
                                               const char* file,
                                               uint32_t line) {
  switch (result) {
    // Success codes.
    case VK_SUCCESS:
      // Command successfully completed.
      return iree_ok_status();
    case VK_NOT_READY:
      // A fence or query has not yet completed.
      return iree_ok_status();
    case VK_TIMEOUT:
      // A wait operation has not completed in the specified time.
      return iree_ok_status();
    case VK_EVENT_SET:
      // An event is signaled.
      return iree_ok_status();
    case VK_EVENT_RESET:
      // An event is unsignaled.
      return iree_ok_status();
    case VK_INCOMPLETE:
      // A return array was too small for the result.
      return iree_ok_status();
    case VK_SUBOPTIMAL_KHR:
      // A swapchain no longer matches the surface properties exactly, but can
      // still be used to present to the surface successfully.
      return iree_ok_status();

    // Error codes.
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      // A host memory allocation has failed.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "VK_ERROR_OUT_OF_HOST_MEMORY");
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      // A device memory allocation has failed.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "VK_ERROR_OUT_OF_DEVICE_MEMORY");
    case VK_ERROR_INITIALIZATION_FAILED:
      // Initialization of an object could not be completed for
      // implementation-specific reasons.
      return iree_make_status_with_location(file, line, IREE_STATUS_UNAVAILABLE,
                                            "VK_ERROR_INITIALIZATION_FAILED");
    case VK_ERROR_DEVICE_LOST:
      // The logical or physical device has been lost.
      //
      // A logical device may become lost for a number of
      // implementation-specific reasons, indicating that pending and future
      // command execution may fail and cause resources and backing memory to
      // become undefined.
      //
      // Typical reasons for device loss will include things like execution
      // timing out (to prevent denial of service), power management events,
      // platform resource management, or implementation errors.
      //
      // When this happens, certain commands will return
      // VK_ERROR_DEVICE_LOST (see Error Codes for a list of such
      // commands). After any such event, the logical device is considered lost.
      // It is not possible to reset the logical device to a non-lost state,
      // however the lost state is specific to a logical device (VkDevice), and
      // the corresponding physical device (VkPhysicalDevice) may be otherwise
      // unaffected.
      //
      // In some cases, the physical device may also be lost, and attempting to
      // create a new logical device will fail, returning VK_ERROR_DEVICE_LOST.
      // This is usually indicative of a problem with the underlying
      // implementation, or its connection to the host. If the physical device
      // has not been lost, and a new logical device is successfully created
      // from that physical device, it must be in the non-lost state.
      //
      // Whilst logical device loss may be recoverable, in the case of physical
      // device loss, it is unlikely that an application will be able to recover
      // unless additional, unaffected physical devices exist on the system. The
      // error is largely informational and intended only to inform the user
      // that a platform issue has occurred, and should be investigated further.
      // For example, underlying hardware may have developed a fault or become
      // physically disconnected from the rest of the system. In many cases,
      // physical device loss may cause other more serious issues such as the
      // operating system crashing; in which case it may not be reported via the
      // Vulkan API.
      //
      // Undefined behavior caused by an application error may cause a device to
      // become lost. However, such undefined behavior may also cause
      // unrecoverable damage to the process, and it is then not guaranteed that
      // the API objects, including the VkPhysicalDevice or the VkInstance are
      // still valid or that the error is recoverable.
      //
      // When a device is lost, its child objects are not implicitly destroyed
      // and their handles are still valid. Those objects must still be
      // destroyed before their parents or the device can be destroyed (see the
      // Object Lifetime section). The host address space corresponding to
      // device memory mapped using vkMapMemory is still valid, and host memory
      // accesses to these mapped regions are still valid, but the contents are
      // undefined. It is still legal to call any API command on the device and
      // child objects.
      //
      // Once a device is lost, command execution may fail, and commands that
      // return a VkResult may return VK_ERROR_DEVICE_LOST.
      // Commands that do not allow run-time errors must still operate correctly
      // for valid usage and, if applicable, return valid data.
      //
      // Commands that wait indefinitely for device execution (namely
      // vkDeviceWaitIdle, vkQueueWaitIdle, vkWaitForFences with a maximum
      // timeout, and vkGetQueryPoolResults with the VK_QUERY_RESULT_WAIT_BIT
      // bit set in flags) must return in finite time even in the case
      // of a lost device, and return either VK_SUCCESS or
      // VK_ERROR_DEVICE_LOST. For any command that may return
      // VK_ERROR_DEVICE_LOST, for the purpose of determining whether a
      // command buffer is in the pending state, or whether resources are
      // considered in-use by the device, a return value of
      // VK_ERROR_DEVICE_LOST is equivalent to VK_SUCCESS.
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "VK_ERROR_DEVICE_LOST");
    case VK_ERROR_MEMORY_MAP_FAILED:
      // Mapping of a memory object has failed.
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "VK_ERROR_MEMORY_MAP_FAILED");
    case VK_ERROR_LAYER_NOT_PRESENT:
      // A requested layer is not present or could not be loaded.
      return iree_make_status_with_location(
          file, line, IREE_STATUS_UNIMPLEMENTED, "VK_ERROR_LAYER_NOT_PRESENT");
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      // A requested extension is not supported.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_UNIMPLEMENTED,
                                            "VK_ERROR_EXTENSION_NOT_PRESENT");
    case VK_ERROR_FEATURE_NOT_PRESENT:
      // A requested feature is not supported.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_UNIMPLEMENTED,
                                            "VK_ERROR_FEATURE_NOT_PRESENT");
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      // The requested version of Vulkan is not supported by the driver or is
      // otherwise incompatible for implementation-specific reasons.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_FAILED_PRECONDITION,
                                            "VK_ERROR_INCOMPATIBLE_DRIVER");
    case VK_ERROR_TOO_MANY_OBJECTS:
      // Too many objects of the type have already been created.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "VK_ERROR_TOO_MANY_OBJECTS");
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      // A requested format is not supported on this device.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_UNIMPLEMENTED,
                                            "VK_ERROR_FORMAT_NOT_SUPPORTED");
    case VK_ERROR_FRAGMENTED_POOL:
      // A pool allocation has failed due to fragmentation of the pool’s
      // memory. This must only be returned if no attempt to allocate host
      // or device memory was made to accommodate the new allocation.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "VK_ERROR_FRAGMENTED_POOL");
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      // A pool memory allocation has failed. This must only be returned if no
      // attempt to allocate host or device memory was made to accommodate the
      // new allocation. If the failure was definitely due to fragmentation of
      // the pool, VK_ERROR_FRAGMENTED_POOL should be returned instead.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "VK_ERROR_OUT_OF_POOL_MEMORY");
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      // An external handle is not a valid handle of the specified type.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_INVALID_ARGUMENT,
                                            "VK_ERROR_INVALID_EXTERNAL_HANDLE");
    case VK_ERROR_SURFACE_LOST_KHR:
      // A surface is no longer available.
      return iree_make_status_with_location(file, line, IREE_STATUS_UNAVAILABLE,
                                            "VK_ERROR_SURFACE_LOST_KHR");
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
      // The requested window is already in use by Vulkan or another API in a
      // manner which prevents it from being used again.
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INVALID_ARGUMENT,
          "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR");
    case VK_ERROR_OUT_OF_DATE_KHR:
      // A surface has changed in such a way that it is no longer compatible
      // with the swapchain, and further presentation requests using the
      // swapchain will fail. Applications must query the new surface properties
      // and recreate their swapchain if they wish to continue presenting to the
      // surface.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_FAILED_PRECONDITION,
                                            "VK_ERROR_OUT_OF_DATE_KHR");
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
      // The display used by a swapchain does not use the same presentable image
      // layout, or is incompatible in a way that prevents sharing an image.
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INVALID_ARGUMENT,
          "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR");
    case VK_ERROR_VALIDATION_FAILED_EXT:
      // Validation layer testing failed. It is not expected that an
      // application would see this this error code during normal use of the
      // validation layers.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_INVALID_ARGUMENT,
                                            "VK_ERROR_VALIDATION_FAILED_EXT");
    case VK_ERROR_INVALID_SHADER_NV:
      // One or more shaders failed to compile or link. More details are
      // reported back to the application when the validation layer is enabled
      // using the extension VK_EXT_debug_report.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_INVALID_ARGUMENT,
                                            "VK_ERROR_INVALID_SHADER_NV");
    case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
      // When creating an image with
      // VkImageDrmFormatModifierExplicitCreateInfoEXT, it is the application’s
      // responsibility to satisfy all Valid Usage requirements. However, the
      // implementation must validate that the provided pPlaneLayouts, when
      // combined with the provided drmFormatModifier and other creation
      // parameters in VkImageCreateInfo and its pNext chain, produce a valid
      // image. (This validation is necessarily implementation-dependent and
      // outside the scope of Vulkan, and therefore not described by Valid Usage
      // requirements). If this validation fails, then vkCreateImage returns
      // VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT.
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INVALID_ARGUMENT,
          "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT");
    case VK_ERROR_FRAGMENTATION_EXT:
      // A descriptor pool creation has failed due to fragmentation.
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_RESOURCE_EXHAUSTED,
                                            "VK_ERROR_FRAGMENTATION_EXT");
    case VK_ERROR_NOT_PERMITTED_EXT:
      // When creating a queue, the caller does not have sufficient privileges
      // to request to acquire a priority above the default priority
      // (VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT).
      return iree_make_status_with_location(file, line,
                                            IREE_STATUS_PERMISSION_DENIED,
                                            "VK_ERROR_NOT_PERMITTED_EXT");
    case VK_ERROR_INVALID_DEVICE_ADDRESS_EXT:
      // A buffer creation failed because the requested address is not
      // available.
      return iree_make_status_with_location(
          file, line, IREE_STATUS_OUT_OF_RANGE,
          "VK_ERROR_INVALID_DEVICE_ADDRESS_EXT");
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      // An operation on a swapchain created with
      // VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT failed as it did
      // not have exlusive full-screen access. This may occur due to
      // implementation-dependent reasons, outside of the application’s control.
      return iree_make_status_with_location(
          file, line, IREE_STATUS_UNAVAILABLE,
          "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT");
    default:
      return iree_make_status_with_location(file, line, IREE_STATUS_UNKNOWN,
                                            "VkResult=%u", (uint32_t)result);
  }
}
