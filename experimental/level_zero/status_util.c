// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/status_util.h"

#include <stddef.h>

#include "experimental/level_zero/dynamic_symbols.h"

iree_status_t iree_hal_level_zero_result_to_status(
    iree_hal_level_zero_dynamic_symbols_t *syms, ze_result_t result,
    const char *file, uint32_t line) {
  switch (result) {
    case ZE_RESULT_SUCCESS:
      // Command successfully completed.
      return iree_ok_status();
    case ZE_RESULT_NOT_READY:
      // A fence or query has not yet completed.
      return iree_ok_status();
    case ZE_RESULT_WARNING_DROPPED_DATA:
      // [Tools] data may have been dropped
      return iree_ok_status();
    case ZE_RESULT_ERROR_DEVICE_LOST:
      // Lost device, may be due to device hung, reset, removed, or driver
      // update occurred.
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_DEVICE_LOST");
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      // [Core] insufficient host memory to satisfy call
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY");
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      // [Core] insufficient device memory to satisfy call
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY");
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
      // [Core] error occurred when building module, see build log for details
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE");
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
      // [Core] error occurred when linking modules, see build log for details
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_MODULE_LINK_FAILURE");
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
      // [Core] device requires a reset
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET");
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
      // [Core] device currently in low power state
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE");
    case ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX:
      // [Core, Expoerimental] device is not represented by a fabric vertex
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX");
    case ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE:
      // [Core, Experimental] fabric vertex does not represent a device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE");
    case ZE_RESULT_EXP_ERROR_REMOTE_DEVICE:
      // [Core, Expoerimental] fabric vertex represents a remote device or
      // subdevice
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_EXP_ERROR_REMOTE_DEVICE");
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
      // [Sysman] access denied due to permission level
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS");
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
      // [Sysman] resource already in use and simultaneous access not allowed or
      // resource was removed
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_NOT_AVAILABLE");
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
      // [Tools] external required dependency is unavailable or missing
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE");
    case ZE_RESULT_ERROR_UNINITIALIZED:
      // [Validation] driver is not initialized
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_UNINITIALIZED");
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      // [Validation] generic error code for unsupported versions
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_UNSUPPORTED_VERSION");
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      // [Validation] generic error code for unsupported features
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE");
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      // [Validation] generic error code for invalid arguments
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_INVALID_ARGUMENT");
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      // [Validation] handle argument is not valid
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_NULL_HANDLE");
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      // [Validation] object pointed to by handle still in-use by device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE");
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      // [Validation] pointer argument may not be NULL
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_NULL_POINTER");
    case ZE_RESULT_ERROR_INVALID_SIZE:
      // [Validation] size argument is invalid (e.g., must not be zero)
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_INVALID_SIZE");
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      // [Validation] size argument is not supported by the device (e.g., too
      // large)
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_UNSUPPORTED_SIZE");
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
      // [Validation] alignment argument is not supported by the device (e.g.,
      // too small)
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT");
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
      // [Validation] synchronization object in invalid state
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT");
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
      // [Validation] enumerator argument is not valid
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_ENUMERATION");
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
      // [Validation] enumerator argument is not supported
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION");
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
      // [Validation] image format is not supported by the device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT");
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
      // [Validation] native binary is not supported by the device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY");
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
      // [Validation] global variable is not found in the module
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME");
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
      // [Validation] kernel name is not found in the module
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_KERNEL_NAME");
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
      // [Validation] function name is not found in the module
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME");
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
      // [Validation] group size dimension is not valid for the kernel or device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION");
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
      // [Validation] global width dimension is not valid for the kernel or
      // device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION");
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
      // [Validation] kernel argument index is not valid for kernel
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX");
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
      // [Validation] kernel argument size does not match kernel
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE");
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
      // [Validation] value of kernel attribute is not valid for the kernel or
      // device
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE");
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
      // [Validation] module with imports needs to be linked before kernels can
      // be created from it
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED");
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
      // [Validation] command list type does not match command queue type
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE");
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
      // [Validation] copy operations do not support overlapping regions of
      // memory
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_ERROR_OVERLAPPING_REGIONS");
    case ZE_RESULT_WARNING_ACTION_REQUIRED:
      // [Sysman] an action is required to complete the desired operation
      return iree_make_status_with_location(
          file, line, IREE_STATUS_INTERNAL,
          "ZE_RESULT_WARNING_ACTION_REQUIRED");
    case ZE_RESULT_ERROR_UNKNOWN:
      // [Core] unknown or internal error
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_ERROR_UNKNOWN");
    case ZE_RESULT_FORCE_UINT32:
      // [Core] unknown or internal error
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "ZE_RESULT_FORCE_UINT32");
    default:
      // [Core] unknown or internal error
      return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                            "Unknown error found");
  }
}
