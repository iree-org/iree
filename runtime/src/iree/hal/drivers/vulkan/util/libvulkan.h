// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_UTIL_LIBVULKAN_H_
#define IREE_HAL_DRIVERS_VULKAN_UTIL_LIBVULKAN_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_dynamic_library_t iree_dynamic_library_t;

//===----------------------------------------------------------------------===//
// Compile-time Configuration
//===----------------------------------------------------------------------===//

// Dynamically loads the Vulkan loader by default so that ordinary IREE binaries
// can run on systems without Vulkan installed. Tiny or hermetic builds may set
// `-DIREE_HAL_VULKAN_LIBVULKAN_STATIC=1` and link against the platform Vulkan
// loader directly. Static mode imports `vkGetInstanceProcAddr` and bypasses all
// dynamic-library lookup machinery.
#if !defined(IREE_HAL_VULKAN_LIBVULKAN_STATIC)
#define IREE_HAL_VULKAN_LIBVULKAN_STATIC 0
// For manual testing:
// #define IREE_HAL_VULKAN_LIBVULKAN_STATIC 1
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

typedef uint32_t iree_hal_vulkan_libvulkan_flags_t;

enum iree_hal_vulkan_libvulkan_flag_bits_e {
  IREE_HAL_VULKAN_LIBVULKAN_FLAG_NONE = 0u,
};

typedef enum iree_hal_vulkan_libvulkan_source_e {
  IREE_HAL_VULKAN_LIBVULKAN_SOURCE_NONE = 0,
  IREE_HAL_VULKAN_LIBVULKAN_SOURCE_EXTERNAL,
  IREE_HAL_VULKAN_LIBVULKAN_SOURCE_STATIC,
  IREE_HAL_VULKAN_LIBVULKAN_SOURCE_DYNAMIC,
} iree_hal_vulkan_libvulkan_source_t;

// Loaded Vulkan loader entry points.
//
// This is a small by-value loader object modeled after AMDGPU libhsa. It can be
// embedded in long-lived driver/device state without adding an extra heap
// indirection to every Vulkan dispatch table access. When copied,
// iree_hal_vulkan_libvulkan_copy must be used so dynamic-library ownership is
// retained correctly.
//
// Thread-safe; immutable after initialization.
typedef struct iree_hal_vulkan_libvulkan_t {
  // True if the structure contains a usable Vulkan loader entry point.
  bool initialized;

  // Identifies how vkGetInstanceProcAddr was provided.
  iree_hal_vulkan_libvulkan_source_t source;

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC
  // Loaded Vulkan dynamic library, if source is dynamic.
  iree_dynamic_library_t* library;
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

  // Root Vulkan loader entry point used to populate instance/device tables.
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
} iree_hal_vulkan_libvulkan_t;

// Initializes |out_libvulkan| by loading the Vulkan loader from disk unless the
// driver is compiled in static mode. The populated structure is immutable once
// initialized but if copied the iree_hal_vulkan_libvulkan_copy API must be
// used.
//
// |search_paths| overrides default library search paths and may contain either
// directories or exact library files. The IREE_HAL_VULKAN_LIBVULKAN_PATH
// environment variable is checked after explicit paths.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_initialize(
    iree_hal_vulkan_libvulkan_flags_t flags,
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan);

// Initializes |out_libvulkan| from an externally provided loader entry point.
// The caller retains ownership of the code containing |vkGetInstanceProcAddr|.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_initialize_from_loader(
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan);

// Deinitializes |libvulkan| by unloading its backing library when owned.
IREE_API_EXPORT void iree_hal_vulkan_libvulkan_deinitialize(
    iree_hal_vulkan_libvulkan_t* libvulkan);

// Copies all resolved symbols from |libvulkan| to |out_libvulkan| and retains
// the dynamic library when present. The target is assumed uninitialized.
IREE_API_EXPORT iree_status_t
iree_hal_vulkan_libvulkan_copy(const iree_hal_vulkan_libvulkan_t* libvulkan,
                               iree_hal_vulkan_libvulkan_t* out_libvulkan);

// Appends the absolute path of the shared library or DLL providing the dynamic
// Vulkan symbols, or a textual source name for static/external loaders.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_append_path_to_builder(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_UTIL_LIBVULKAN_H_
