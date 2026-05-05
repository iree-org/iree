// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_UTIL_LIBVULKAN_H_
#define IREE_HAL_DRIVERS_VULKAN_UTIL_LIBVULKAN_H_

#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES
#endif  // !VK_NO_PROTOTYPES
#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "vulkan/vulkan.h"

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

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC
#define IREE_HAL_VULKAN_LOADER_PFN(result_type, symbol, decl, args) \
  PFN_##symbol symbol;
#define IREE_HAL_VULKAN_INSTANCE_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(...)
#define IREE_HAL_VULKAN_DEVICE_PFN(...)
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: export
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC
} iree_hal_vulkan_libvulkan_t;

// Instance-level Vulkan dispatch table.
typedef struct iree_hal_vulkan_instance_syms_t {
#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  // Placeholder field so the type remains embeddable in static mode.
  uint8_t reserved;
#else
#define IREE_HAL_VULKAN_LOADER_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PFN(result_type, symbol, decl, args) \
  PFN_##symbol symbol;
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN IREE_HAL_VULKAN_INSTANCE_PFN
#define IREE_HAL_VULKAN_DEVICE_PFN(...)
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: export
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC
} iree_hal_vulkan_instance_syms_t;

// Device-level Vulkan dispatch table.
typedef struct iree_hal_vulkan_device_syms_t {
#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  // Placeholder field so the type remains embeddable in static mode.
  uint8_t reserved;
#else
#define IREE_HAL_VULKAN_LOADER_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(...)
#define IREE_HAL_VULKAN_DEVICE_PFN(result_type, symbol, decl, args) \
  PFN_##symbol symbol;
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: export
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC
} iree_hal_vulkan_device_syms_t;

// Dispatch table entries are load invariants, not capability predicates.
// Extension and feature decisions must be made from cached Vulkan inventory:
// instance/device extension names, promoted core features, and queried property
// structs. Static builds can expose linked entry points even when a device has
// not enabled the corresponding extension, and dynamic loaders may return
// trampolines whose presence says nothing about device support.

// Returns an IREE status for |result| with source location and symbol context.
IREE_API_EXPORT iree_status_t iree_status_from_vk_result(const char* file,
                                                         uint32_t line,
                                                         VkResult result,
                                                         const char* symbol);

// Loads instance-level Vulkan functions.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_load_instance_syms(
    const iree_hal_vulkan_libvulkan_t* libvulkan, VkInstance instance,
    iree_hal_vulkan_instance_syms_t* out_syms);

// Loads device-level Vulkan functions.
IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_load_device_syms(
    const iree_hal_vulkan_instance_syms_t* instance_syms, VkDevice device,
    iree_hal_vulkan_device_syms_t* out_syms);

//===----------------------------------------------------------------------===//
// Vulkan API Wrappers
//===----------------------------------------------------------------------===//

// Wraps an iree_hal_vulkan_libvulkan_t* for use with loader-level API wrappers.
// All calls should either embed their file and line information directly or use
// this macro.
#define IREE_LIBVULKAN(libvulkan) (libvulkan), __FILE__, __LINE__

// Wraps an iree_hal_vulkan_instance_syms_t* for instance-level API wrappers.
#define IREE_VULKAN_INSTANCE(instance_syms) (instance_syms), __FILE__, __LINE__

// Wraps an iree_hal_vulkan_device_syms_t* for device-level API wrappers.
#define IREE_VULKAN_DEVICE(device_syms) (device_syms), __FILE__, __LINE__

#define IREE_HAL_VULKAN_DECLARE_LOADER_VkResult(result_type, symbol, decl)    \
  IREE_API_EXPORT iree_status_t iree_##symbol(                                \
      const iree_hal_vulkan_libvulkan_t* IREE_RESTRICT libvulkan,             \
      const char* file, uint32_t line _COMMA_DECL(decl));                     \
  IREE_API_EXPORT result_type iree_##symbol##_raw(                            \
      const iree_hal_vulkan_libvulkan_t* IREE_RESTRICT libvulkan _COMMA_DECL( \
          decl));
#define IREE_HAL_VULKAN_DECLARE_INSTANCE_VkResult(result_type, symbol, decl) \
  IREE_API_EXPORT iree_status_t iree_##symbol(                               \
      const iree_hal_vulkan_instance_syms_t* IREE_RESTRICT syms,             \
      const char* file, uint32_t line _COMMA_DECL(decl));                    \
  IREE_API_EXPORT result_type iree_##symbol##_raw(                           \
      const iree_hal_vulkan_instance_syms_t* IREE_RESTRICT syms _COMMA_DECL( \
          decl));
#define IREE_HAL_VULKAN_DECLARE_INSTANCE_void(result_type, symbol, decl) \
  IREE_API_EXPORT void iree_##symbol(                                    \
      const iree_hal_vulkan_instance_syms_t* IREE_RESTRICT syms,         \
      const char* file, uint32_t line _COMMA_DECL(decl));
#define IREE_HAL_VULKAN_DECLARE_DEVICE_void(result_type, symbol, decl) \
  IREE_API_EXPORT void iree_##symbol(                                  \
      const iree_hal_vulkan_device_syms_t* IREE_RESTRICT syms,         \
      const char* file, uint32_t line _COMMA_DECL(decl));
#define IREE_HAL_VULKAN_DECLARE_DEVICE_VkResult(result_type, symbol, decl) \
  IREE_API_EXPORT iree_status_t iree_##symbol(                             \
      const iree_hal_vulkan_device_syms_t* IREE_RESTRICT syms,             \
      const char* file, uint32_t line _COMMA_DECL(decl));                  \
  IREE_API_EXPORT result_type iree_##symbol##_raw(                         \
      const iree_hal_vulkan_device_syms_t* IREE_RESTRICT syms _COMMA_DECL( \
          decl));

#define IREE_HAL_VULKAN_LOADER_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DECLARE_LOADER_##result_type(result_type, symbol, DECL(decl))
#define IREE_HAL_VULKAN_INSTANCE_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DECLARE_INSTANCE_##result_type(result_type, symbol, \
                                                 DECL(decl))
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(...)
#define IREE_HAL_VULKAN_DEVICE_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DECLARE_DEVICE_##result_type(result_type, symbol, DECL(decl))
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#define _COMMA_DECL(...) __VA_OPT__(, ) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: export
#undef _COMMA_DECL
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN
#undef IREE_HAL_VULKAN_DECLARE_DEVICE_void
#undef IREE_HAL_VULKAN_DECLARE_DEVICE_VkResult
#undef IREE_HAL_VULKAN_DECLARE_INSTANCE_void
#undef IREE_HAL_VULKAN_DECLARE_INSTANCE_VkResult
#undef IREE_HAL_VULKAN_DECLARE_LOADER_VkResult

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
