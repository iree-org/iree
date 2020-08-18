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

#include "iree/hal/vulkan/status_util.h"

namespace iree {
namespace hal {
namespace vulkan {

Status VkResultToStatus(VkResult result, SourceLocation loc) {
  switch (result) {
    // Success codes.
    case VK_SUCCESS:
      // Command successfully completed.
      return OkStatus();
    case VK_NOT_READY:
      // A fence or query has not yet completed.
      return OkStatus();
    case VK_TIMEOUT:
      // A wait operation has not completed in the specified time.
      return OkStatus();
    case VK_EVENT_SET:
      // An event is signaled.
      return OkStatus();
    case VK_EVENT_RESET:
      // An event is unsignaled.
      return OkStatus();
    case VK_INCOMPLETE:
      // A return array was too small for the result.
      return OkStatus();
    case VK_SUBOPTIMAL_KHR:
      // A swapchain no longer matches the surface properties exactly, but can
      // still be used to present to the surface successfully.
      return OkStatus();

    // Error codes.
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      // A host memory allocation has failed.
      return ResourceExhaustedErrorBuilder(loc)
             << "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      // A device memory allocation has failed.
      return ResourceExhaustedErrorBuilder(loc)
             << "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      // Initialization of an object could not be completed for
      // implementation-specific reasons.
      return InternalErrorBuilder(loc) << "VK_ERROR_INITIALIZATION_FAILED";
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
      return InternalErrorBuilder(loc) << "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      // Mapping of a memory object has failed.
      return InternalErrorBuilder(loc) << "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      // A requested layer is not present or could not be loaded.
      return UnimplementedErrorBuilder(loc) << "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      // A requested extension is not supported.
      return UnimplementedErrorBuilder(loc) << "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      // A requested feature is not supported.
      return UnimplementedErrorBuilder(loc) << "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      // The requested version of Vulkan is not supported by the driver or is
      // otherwise incompatible for implementation-specific reasons.
      return FailedPreconditionErrorBuilder(loc)
             << "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      // Too many objects of the type have already been created.
      return ResourceExhaustedErrorBuilder(loc) << "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      // A requested format is not supported on this device.
      return UnimplementedErrorBuilder(loc) << "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      // A pool allocation has failed due to fragmentation of the pool’s memory.
      // This must only be returned if no attempt to allocate host or device
      // memory was made to accommodate the new allocation.
      return ResourceExhaustedErrorBuilder(loc) << "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      // A pool memory allocation has failed. This must only be returned if no
      // attempt to allocate host or device memory was made to accommodate the
      // new allocation. If the failure was definitely due to fragmentation of
      // the pool, VK_ERROR_FRAGMENTED_POOL should be returned instead.
      return ResourceExhaustedErrorBuilder(loc)
             << "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      // An external handle is not a valid handle of the specified type.
      return InvalidArgumentErrorBuilder(loc)
             << "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_SURFACE_LOST_KHR:
      // A surface is no longer available.
      return UnavailableErrorBuilder(loc) << "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
      // The requested window is already in use by Vulkan or another API in a
      // manner which prevents it from being used again.
      return InvalidArgumentErrorBuilder(loc)
             << "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
      // A surface has changed in such a way that it is no longer compatible
      // with the swapchain, and further presentation requests using the
      // swapchain will fail. Applications must query the new surface properties
      // and recreate their swapchain if they wish to continue presenting to the
      // surface.
      return FailedPreconditionErrorBuilder(loc) << "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
      // The display used by a swapchain does not use the same presentable image
      // layout, or is incompatible in a way that prevents sharing an image.
      return InvalidArgumentErrorBuilder(loc)
             << "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT:
      // Validation layer testing failed. It is not expected that an
      // application would see this this error code during normal use of the
      // validation layers.
      return InvalidArgumentErrorBuilder(loc)
             << "VK_ERROR_VALIDATION_FAILED_EXT";
    case VK_ERROR_INVALID_SHADER_NV:
      // One or more shaders failed to compile or link. More details are
      // reported back to the application when the validation layer is enabled
      // using the extension VK_EXT_debug_report.
      return InvalidArgumentErrorBuilder(loc) << "VK_ERROR_INVALID_SHADER_NV";
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
      return InvalidArgumentErrorBuilder(loc)
             << "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
    case VK_ERROR_FRAGMENTATION_EXT:
      // A descriptor pool creation has failed due to fragmentation.
      return ResourceExhaustedErrorBuilder(loc) << "VK_ERROR_FRAGMENTATION_EXT";
    case VK_ERROR_NOT_PERMITTED_EXT:
      // When creating a queue, the caller does not have sufficient privileges
      // to request to acquire a priority above the default priority
      // (VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT).
      return PermissionDeniedErrorBuilder(loc) << "VK_ERROR_NOT_PERMITTED_EXT";
    case VK_ERROR_INVALID_DEVICE_ADDRESS_EXT:
      // A buffer creation failed because the requested address is not
      // available.
      return OutOfRangeErrorBuilder(loc)
             << "VK_ERROR_INVALID_DEVICE_ADDRESS_EXT";
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      // An operation on a swapchain created with
      // VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT failed as it did
      // not have exlusive full-screen access. This may occur due to
      // implementation-dependent reasons, outside of the application’s control.
      return UnavailableErrorBuilder(loc)
             << "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    default:
      return UnknownErrorBuilder(loc) << result;
  }
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
