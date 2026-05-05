// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"

//===----------------------------------------------------------------------===//
// VkResult interop
//===----------------------------------------------------------------------===//

static iree_status_code_t iree_hal_vulkan_status_code(VkResult result) {
  switch (result) {
    default:
      return IREE_STATUS_UNKNOWN;
    case VK_SUCCESS:
      return IREE_STATUS_OK;
    case VK_NOT_READY:
      return IREE_STATUS_ABORTED;
    case VK_TIMEOUT:
      return IREE_STATUS_DEADLINE_EXCEEDED;
    case VK_EVENT_SET:
    case VK_EVENT_RESET:
    case VK_INCOMPLETE:
      return IREE_STATUS_ABORTED;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_TOO_MANY_OBJECTS:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case VK_ERROR_INITIALIZATION_FAILED:
      return IREE_STATUS_FAILED_PRECONDITION;
    case VK_ERROR_DEVICE_LOST:
      return IREE_STATUS_DATA_LOSS;
    case VK_ERROR_MEMORY_MAP_FAILED:
      return IREE_STATUS_INTERNAL;
    case VK_ERROR_LAYER_NOT_PRESENT:
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return IREE_STATUS_NOT_FOUND;
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return IREE_STATUS_UNAVAILABLE;
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return IREE_STATUS_INCOMPATIBLE;
    case VK_ERROR_FRAGMENTED_POOL:
    case VK_ERROR_FRAGMENTATION:
      return IREE_STATUS_RESOURCE_EXHAUSTED;
    case VK_ERROR_UNKNOWN:
      return IREE_STATUS_UNKNOWN;
  }
}

static const char* iree_hal_vulkan_result_string(VkResult result) {
  switch (result) {
    default:
      return "VK_RESULT_UNKNOWN";
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
      return "VK_ERROR_UNKNOWN";
    case VK_ERROR_FRAGMENTATION:
      return "VK_ERROR_FRAGMENTATION";
  }
}

IREE_API_EXPORT iree_status_t iree_status_from_vk_result(const char* file,
                                                         uint32_t line,
                                                         VkResult result,
                                                         const char* symbol) {
  if (result == VK_SUCCESS) return iree_ok_status();
  return iree_status_allocate_f(iree_hal_vulkan_status_code(result), file, line,
                                "[%s] %s", symbol,
                                iree_hal_vulkan_result_string(result));
}

//===----------------------------------------------------------------------===//
// Symbol lookup
//===----------------------------------------------------------------------===//

#if IREE_HAL_VULKAN_LIBVULKAN_STATIC

extern PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance,
                                                           const char* name);

#define IREE_HAL_VULKAN_DECLARE_STATIC(result_type, symbol, decl) \
  extern result_type VKAPI_CALL symbol(decl);
#define IREE_HAL_VULKAN_LOADER_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DECLARE_STATIC(result_type, symbol, DECL(decl))
#define IREE_HAL_VULKAN_INSTANCE_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DECLARE_STATIC(result_type, symbol, DECL(decl))
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN IREE_HAL_VULKAN_INSTANCE_PFN
#define IREE_HAL_VULKAN_DEVICE_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DECLARE_STATIC(result_type, symbol, DECL(decl))
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: keep
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN
#undef IREE_HAL_VULKAN_DECLARE_STATIC

#else

static iree_status_t iree_hal_vulkan_lookup_loader(
    const iree_hal_vulkan_libvulkan_t* libvulkan, const char* symbol,
    PFN_vkVoidFunction* out_fn) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = NULL;
  if (!libvulkan->vkGetInstanceProcAddr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan loader is not initialized");
  }
  *out_fn = libvulkan->vkGetInstanceProcAddr(VK_NULL_HANDLE, symbol);
  if (!*out_fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Vulkan loader symbol '%s' not found", symbol);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_lookup_instance(
    const iree_hal_vulkan_libvulkan_t* libvulkan, VkInstance instance,
    const char* symbol, PFN_vkVoidFunction* out_fn) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = NULL;
  if (!libvulkan->vkGetInstanceProcAddr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan loader is not initialized");
  }
  *out_fn = libvulkan->vkGetInstanceProcAddr(instance, symbol);
  if (!*out_fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Vulkan instance symbol '%s' not found", symbol);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_lookup_device(
    const iree_hal_vulkan_instance_syms_t* instance_syms, VkDevice device,
    const char* symbol, PFN_vkVoidFunction* out_fn) {
  IREE_ASSERT_ARGUMENT(instance_syms);
  IREE_ASSERT_ARGUMENT(out_fn);
  *out_fn = NULL;
  if (!instance_syms->vkGetDeviceProcAddr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan device symbol resolver is not loaded");
  }
  *out_fn = instance_syms->vkGetDeviceProcAddr(device, symbol);
  if (!*out_fn) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Vulkan device symbol '%s' not found", symbol);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_libvulkan_load_loader_syms(
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
#define IREE_HAL_VULKAN_LOADER_PFN(result_type, symbol, decl, args)           \
  if (iree_status_is_ok(status)) {                                            \
    status = iree_hal_vulkan_lookup_loader(                                   \
        out_libvulkan, #symbol, (PFN_vkVoidFunction*)&out_libvulkan->symbol); \
  }
#define IREE_HAL_VULKAN_INSTANCE_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(...)
#define IREE_HAL_VULKAN_DEVICE_PFN(...)
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: keep
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC

IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_load_instance_syms(
    const iree_hal_vulkan_libvulkan_t* libvulkan, VkInstance instance,
    iree_hal_vulkan_instance_syms_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  memset(out_syms, 0, sizeof(*out_syms));
  IREE_TRACE_ZONE_BEGIN(z0);

#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  (void)libvulkan;
  (void)instance;
  iree_status_t status = iree_ok_status();
#else
  iree_status_t status = iree_ok_status();
#define IREE_HAL_VULKAN_LOADER_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PFN(result_type, symbol, decl, args)          \
  if (iree_status_is_ok(status)) {                                             \
    status = iree_hal_vulkan_lookup_instance(                                  \
        libvulkan, instance, #symbol, (PFN_vkVoidFunction*)&out_syms->symbol); \
  }
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN IREE_HAL_VULKAN_INSTANCE_PFN
#define IREE_HAL_VULKAN_DEVICE_PFN(...)
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: keep
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN

  if (!iree_status_is_ok(status) && !out_syms->vkDestroyInstance) {
    memset(out_syms, 0, sizeof(*out_syms));
  }
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_load_device_syms(
    const iree_hal_vulkan_instance_syms_t* instance_syms, VkDevice device,
    iree_hal_vulkan_device_syms_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  memset(out_syms, 0, sizeof(*out_syms));
  IREE_TRACE_ZONE_BEGIN(z0);

#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  (void)instance_syms;
  (void)device;
  iree_status_t status = iree_ok_status();
#else
  iree_status_t status = iree_ok_status();
#define IREE_HAL_VULKAN_LOADER_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PFN(...)
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(...)
#define IREE_HAL_VULKAN_DEVICE_PFN(result_type, symbol, decl, args)            \
  if (iree_status_is_ok(status)) {                                             \
    status =                                                                   \
        iree_hal_vulkan_lookup_device(instance_syms, device, #symbol,          \
                                      (PFN_vkVoidFunction*)&out_syms->symbol); \
  }
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: keep
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN

  if (!iree_status_is_ok(status) && !out_syms->vkDestroyDevice) {
    memset(out_syms, 0, sizeof(*out_syms));
  }
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Vulkan API Wrappers
//===----------------------------------------------------------------------===//

#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
#define IREE_HAL_VULKAN_LOADER_LIBPTR(libvulkan)
#define IREE_HAL_VULKAN_INSTANCE_LIBPTR(syms)
#define IREE_HAL_VULKAN_DEVICE_LIBPTR(syms)
#define IREE_HAL_VULKAN_ASSERT_LOADED(syms, symbol)
#define IREE_HAL_VULKAN_STATIC_UNUSED(value) (void)(value)
#else
#define IREE_HAL_VULKAN_LOADER_LIBPTR(libvulkan) (libvulkan)->
#define IREE_HAL_VULKAN_INSTANCE_LIBPTR(syms) (syms)->
#define IREE_HAL_VULKAN_DEVICE_LIBPTR(syms) (syms)->
#define IREE_HAL_VULKAN_ASSERT_LOADED(syms, symbol) \
  IREE_ASSERT((syms)->symbol != NULL, "Vulkan symbol not loaded: %s", #symbol)
#define IREE_HAL_VULKAN_STATIC_UNUSED(value)
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC

#define IREE_HAL_VULKAN_DEFINE_LOADER_VkResult(result_type, symbol, decl,     \
                                               args)                          \
  IREE_API_EXPORT result_type iree_##symbol##_raw(                            \
      const iree_hal_vulkan_libvulkan_t* IREE_RESTRICT libvulkan _COMMA_DECL( \
          decl)) {                                                            \
    IREE_HAL_VULKAN_STATIC_UNUSED(libvulkan);                                 \
    IREE_HAL_VULKAN_ASSERT_LOADED(libvulkan, symbol);                         \
    return IREE_HAL_VULKAN_LOADER_LIBPTR(libvulkan) symbol(args);             \
  }                                                                           \
  IREE_API_EXPORT iree_status_t iree_##symbol(                                \
      const iree_hal_vulkan_libvulkan_t* IREE_RESTRICT libvulkan,             \
      const char* file, uint32_t line _COMMA_DECL(decl)) {                    \
    VkResult result = iree_##symbol##_raw(libvulkan _COMMA_ARGS(args));       \
    return iree_status_from_vk_result(file, line, result, #symbol);           \
  }

#define IREE_HAL_VULKAN_DEFINE_INSTANCE_VkResult(result_type, symbol, decl,  \
                                                 args)                       \
  IREE_API_EXPORT result_type iree_##symbol##_raw(                           \
      const iree_hal_vulkan_instance_syms_t* IREE_RESTRICT syms _COMMA_DECL( \
          decl)) {                                                           \
    IREE_HAL_VULKAN_STATIC_UNUSED(syms);                                     \
    IREE_HAL_VULKAN_ASSERT_LOADED(syms, symbol);                             \
    return IREE_HAL_VULKAN_INSTANCE_LIBPTR(syms) symbol(args);               \
  }                                                                          \
  IREE_API_EXPORT iree_status_t iree_##symbol(                               \
      const iree_hal_vulkan_instance_syms_t* IREE_RESTRICT syms,             \
      const char* file, uint32_t line _COMMA_DECL(decl)) {                   \
    VkResult result = iree_##symbol##_raw(syms _COMMA_ARGS(args));           \
    return iree_status_from_vk_result(file, line, result, #symbol);          \
  }

#define IREE_HAL_VULKAN_DEFINE_INSTANCE_void(result_type, symbol, decl, args) \
  IREE_API_EXPORT void iree_##symbol(                                         \
      const iree_hal_vulkan_instance_syms_t* IREE_RESTRICT syms,              \
      const char* file, uint32_t line _COMMA_DECL(decl)) {                    \
    (void)file;                                                               \
    (void)line;                                                               \
    IREE_HAL_VULKAN_STATIC_UNUSED(syms);                                      \
    IREE_HAL_VULKAN_ASSERT_LOADED(syms, symbol);                              \
    IREE_HAL_VULKAN_INSTANCE_LIBPTR(syms) symbol(args);                       \
  }

#define IREE_HAL_VULKAN_DEFINE_DEVICE_void(result_type, symbol, decl, args) \
  IREE_API_EXPORT void iree_##symbol(                                       \
      const iree_hal_vulkan_device_syms_t* IREE_RESTRICT syms,              \
      const char* file, uint32_t line _COMMA_DECL(decl)) {                  \
    (void)file;                                                             \
    (void)line;                                                             \
    IREE_HAL_VULKAN_STATIC_UNUSED(syms);                                    \
    IREE_HAL_VULKAN_ASSERT_LOADED(syms, symbol);                            \
    IREE_HAL_VULKAN_DEVICE_LIBPTR(syms) symbol(args);                       \
  }
#define IREE_HAL_VULKAN_DEFINE_DEVICE_VkResult(result_type, symbol, decl,  \
                                               args)                       \
  IREE_API_EXPORT result_type iree_##symbol##_raw(                         \
      const iree_hal_vulkan_device_syms_t* IREE_RESTRICT syms _COMMA_DECL( \
          decl)) {                                                         \
    IREE_HAL_VULKAN_STATIC_UNUSED(syms);                                   \
    IREE_HAL_VULKAN_ASSERT_LOADED(syms, symbol);                           \
    return IREE_HAL_VULKAN_DEVICE_LIBPTR(syms) symbol(args);               \
  }                                                                        \
  IREE_API_EXPORT iree_status_t iree_##symbol(                             \
      const iree_hal_vulkan_device_syms_t* IREE_RESTRICT syms,             \
      const char* file, uint32_t line _COMMA_DECL(decl)) {                 \
    VkResult result = iree_##symbol##_raw(syms _COMMA_ARGS(args));         \
    return iree_status_from_vk_result(file, line, result, #symbol);        \
  }

#define IREE_HAL_VULKAN_LOADER_PFN(result_type, symbol, decl, args)            \
  IREE_HAL_VULKAN_DEFINE_LOADER_##result_type(result_type, symbol, DECL(decl), \
                                              ARGS(args))
#define IREE_HAL_VULKAN_INSTANCE_PFN(result_type, symbol, decl, args) \
  IREE_HAL_VULKAN_DEFINE_INSTANCE_##result_type(result_type, symbol,  \
                                                DECL(decl), ARGS(args))
#define IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(...)
#define IREE_HAL_VULKAN_DEVICE_PFN(result_type, symbol, decl, args)            \
  IREE_HAL_VULKAN_DEFINE_DEVICE_##result_type(result_type, symbol, DECL(decl), \
                                              ARGS(args))
#define DECL(...) __VA_ARGS__
#define ARGS(...) __VA_ARGS__
#define _COMMA_DECL(...) __VA_OPT__(, ) __VA_ARGS__
#define _COMMA_ARGS(...) __VA_OPT__(, ) __VA_ARGS__
#include "iree/hal/drivers/vulkan/util/libvulkan_tables.h"  // IWYU pragma: keep
#undef _COMMA_ARGS
#undef _COMMA_DECL
#undef ARGS
#undef DECL
#undef IREE_HAL_VULKAN_DEVICE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN
#undef IREE_HAL_VULKAN_INSTANCE_PFN
#undef IREE_HAL_VULKAN_LOADER_PFN
#undef IREE_HAL_VULKAN_DEFINE_DEVICE_void
#undef IREE_HAL_VULKAN_DEFINE_DEVICE_VkResult
#undef IREE_HAL_VULKAN_DEFINE_INSTANCE_void
#undef IREE_HAL_VULKAN_DEFINE_INSTANCE_VkResult
#undef IREE_HAL_VULKAN_DEFINE_LOADER_VkResult
#undef IREE_HAL_VULKAN_STATIC_UNUSED
#undef IREE_HAL_VULKAN_ASSERT_LOADED
#undef IREE_HAL_VULKAN_DEVICE_LIBPTR
#undef IREE_HAL_VULKAN_INSTANCE_LIBPTR
#undef IREE_HAL_VULKAN_LOADER_LIBPTR

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_libvulkan_t
//===----------------------------------------------------------------------===//

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

static iree_status_t iree_hal_vulkan_libvulkan_load_symbols(
    iree_dynamic_library_t* library,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(
      library, "vkGetInstanceProcAddr",
      (void**)&out_libvulkan->vkGetInstanceProcAddr));
  return iree_hal_vulkan_libvulkan_load_loader_syms(out_libvulkan);
}

static iree_status_t iree_hal_vulkan_libvulkan_try_load_library_from_file(
    iree_hal_vulkan_libvulkan_flags_t flags, const char* file_path,
    iree_string_builder_t* error_builder, iree_allocator_t host_allocator,
    iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, file_path);
  *out_library = NULL;

  iree_status_t status = iree_dynamic_library_load_from_file(
      file_path, flags, host_allocator, out_library);
  if (!iree_status_is_ok(status)) {
    iree_status_t load_status = status;
    status = iree_string_builder_append_format(
        error_builder, "\n  Tried: %s\n    ", file_path);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_status(error_builder, load_status);
    }
    iree_status_free(load_status);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const char* iree_hal_vulkan_libvulkan_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "vulkan-1.dll",
#elif defined(IREE_PLATFORM_ANDROID)
    "libvulkan.so",
#elif defined(IREE_PLATFORM_APPLE)
    "libvulkan.1.dylib",
    "libvulkan.dylib",
    "libMoltenVK.dylib",
#else
    "libvulkan.so.1",
    "libvulkan.so",
#endif  // IREE_PLATFORM_WINDOWS
};

static iree_status_t iree_hal_vulkan_libvulkan_try_load_library_from_path(
    iree_hal_vulkan_libvulkan_flags_t flags, iree_string_view_t path_fragment,
    iree_string_builder_t* error_builder, iree_allocator_t host_allocator,
    iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path_fragment.data, path_fragment.size);
  *out_library = NULL;

  iree_string_builder_t path_builder;
  iree_string_builder_initialize(host_allocator, &path_builder);
  iree_status_t status = iree_ok_status();

  if (iree_file_path_is_dynamic_library(path_fragment)) {
    status = iree_string_builder_append_string(&path_builder, path_fragment);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_libvulkan_try_load_library_from_file(
          flags, iree_string_builder_buffer(&path_builder), error_builder,
          host_allocator, out_library);
    }
  } else {
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) &&
         i < IREE_ARRAYSIZE(iree_hal_vulkan_libvulkan_names) && !*out_library;
         ++i) {
      iree_string_builder_reset(&path_builder);
      status = iree_string_builder_append_format(
          &path_builder, "%.*s/%s", (int)path_fragment.size, path_fragment.data,
          iree_hal_vulkan_libvulkan_names[i]);
      if (iree_status_is_ok(status)) {
        path_builder.size = iree_file_path_canonicalize(
            (char*)iree_string_builder_buffer(&path_builder),
            iree_string_builder_size(&path_builder));
        status = iree_hal_vulkan_libvulkan_try_load_library_from_file(
            flags, iree_string_builder_buffer(&path_builder), error_builder,
            host_allocator, out_library);
      }
    }
  }

  iree_string_builder_deinitialize(&path_builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_libvulkan_load_library(
    iree_hal_vulkan_libvulkan_flags_t flags,
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_ASSERT_ARGUMENT(out_libvulkan);
  IREE_TRACE_ZONE_BEGIN(z0);
  out_libvulkan->library = NULL;

  iree_string_builder_t error_builder;
  iree_string_builder_initialize(host_allocator, &error_builder);

  iree_dynamic_library_t* library = NULL;
  iree_status_t status = iree_ok_status();

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < search_paths.count && !library; ++i) {
    status = iree_hal_vulkan_libvulkan_try_load_library_from_path(
        flags, search_paths.values[i], &error_builder, host_allocator,
        &library);
  }

  iree_string_view_t env_path =
      iree_make_cstring_view(getenv("IREE_HAL_VULKAN_LIBVULKAN_PATH"));
  if (iree_status_is_ok(status) && !library &&
      !iree_string_view_is_empty(env_path)) {
    status = iree_hal_vulkan_libvulkan_try_load_library_from_path(
        flags, env_path, &error_builder, host_allocator, &library);
  }

  if (iree_status_is_ok(status) && !library) {
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) &&
         i < IREE_ARRAYSIZE(iree_hal_vulkan_libvulkan_names) && !library;
         ++i) {
      status = iree_hal_vulkan_libvulkan_try_load_library_from_file(
          flags, iree_hal_vulkan_libvulkan_names[i], &error_builder,
          host_allocator, &library);
    }
  }

  if (iree_status_is_ok(status) && !library) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "Vulkan loader library not found; ensure it is installed and on a "
        "valid search path (or specified with "
        "IREE_HAL_VULKAN_LIBVULKAN_PATH): %.*s",
        (int)iree_string_builder_size(&error_builder),
        iree_string_builder_buffer(&error_builder));
  }
  iree_string_builder_deinitialize(&error_builder);

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_libvulkan_load_symbols(library, out_libvulkan);
    if (!iree_status_is_ok(status) && out_libvulkan->vkGetInstanceProcAddr) {
      iree_string_builder_t annotation_builder;
      iree_string_builder_initialize(host_allocator, &annotation_builder);
      iree_status_t annotation_status =
          iree_dynamic_library_append_symbol_path_to_builder(
              (void*)out_libvulkan->vkGetInstanceProcAddr, &annotation_builder);
      if (iree_status_is_ok(annotation_status)) {
        status = iree_status_annotate_f(
            status, "using %.*s",
            (int)iree_string_builder_size(&annotation_builder),
            iree_string_builder_buffer(&annotation_builder));
      } else {
        status = iree_status_join(status, annotation_status);
      }
      iree_string_builder_deinitialize(&annotation_builder);
    }
  }

  if (iree_status_is_ok(status)) {
    out_libvulkan->library = library;
    out_libvulkan->source = IREE_HAL_VULKAN_LIBVULKAN_SOURCE_DYNAMIC;
  } else {
    iree_dynamic_library_release(library);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_libvulkan_unload_library(
    iree_hal_vulkan_libvulkan_t* libvulkan) {
  iree_dynamic_library_release(libvulkan->library);
}

static void iree_hal_vulkan_libvulkan_copy_library(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  memcpy(out_libvulkan, libvulkan, sizeof(*out_libvulkan));
  if (out_libvulkan->library) {
    iree_dynamic_library_retain(out_libvulkan->library);
  }
}

#else

extern PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance,
                                                           const char* name);

static iree_status_t iree_hal_vulkan_libvulkan_load_library(
    iree_hal_vulkan_libvulkan_flags_t flags,
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  (void)flags;
  (void)search_paths;
  (void)host_allocator;
  out_libvulkan->vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  out_libvulkan->source = IREE_HAL_VULKAN_LIBVULKAN_SOURCE_STATIC;
  return iree_ok_status();
}

static void iree_hal_vulkan_libvulkan_unload_library(
    iree_hal_vulkan_libvulkan_t* libvulkan) {}

static void iree_hal_vulkan_libvulkan_copy_library(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  memcpy(out_libvulkan, libvulkan, sizeof(*out_libvulkan));
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_initialize(
    iree_hal_vulkan_libvulkan_flags_t flags,
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_ASSERT_ARGUMENT(out_libvulkan);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, flags);

  memset(out_libvulkan, 0, sizeof(*out_libvulkan));
  iree_status_t status = iree_hal_vulkan_libvulkan_load_library(
      flags, search_paths, host_allocator, out_libvulkan);
  if (iree_status_is_ok(status)) {
    out_libvulkan->initialized = true;
  } else {
    iree_hal_vulkan_libvulkan_deinitialize(out_libvulkan);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_initialize_from_loader(
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_ASSERT_ARGUMENT(vkGetInstanceProcAddr);
  IREE_ASSERT_ARGUMENT(out_libvulkan);
  (void)host_allocator;

  memset(out_libvulkan, 0, sizeof(*out_libvulkan));
  out_libvulkan->initialized = true;
  out_libvulkan->source = IREE_HAL_VULKAN_LIBVULKAN_SOURCE_EXTERNAL;
  out_libvulkan->vkGetInstanceProcAddr = vkGetInstanceProcAddr;
#if IREE_HAL_VULKAN_LIBVULKAN_STATIC
  return iree_ok_status();
#else
  iree_status_t status =
      iree_hal_vulkan_libvulkan_load_loader_syms(out_libvulkan);
  if (!iree_status_is_ok(status)) {
    memset(out_libvulkan, 0, sizeof(*out_libvulkan));
  }
  return status;
#endif  // IREE_HAL_VULKAN_LIBVULKAN_STATIC
}

IREE_API_EXPORT void iree_hal_vulkan_libvulkan_deinitialize(
    iree_hal_vulkan_libvulkan_t* libvulkan) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (libvulkan->initialized) {
    iree_hal_vulkan_libvulkan_unload_library(libvulkan);
  }
  memset(libvulkan, 0, sizeof(*libvulkan));

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t
iree_hal_vulkan_libvulkan_copy(const iree_hal_vulkan_libvulkan_t* libvulkan,
                               iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(out_libvulkan);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_libvulkan_copy_library(libvulkan, out_libvulkan);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_libvulkan_append_path_to_builder(
    const iree_hal_vulkan_libvulkan_t* libvulkan,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(libvulkan);
  IREE_ASSERT_ARGUMENT(builder);

  switch (libvulkan->source) {
    case IREE_HAL_VULKAN_LIBVULKAN_SOURCE_EXTERNAL:
      return iree_string_builder_append_cstring(builder, "<external>");
    case IREE_HAL_VULKAN_LIBVULKAN_SOURCE_STATIC:
      return iree_string_builder_append_cstring(builder, "<statically linked>");
    case IREE_HAL_VULKAN_LIBVULKAN_SOURCE_DYNAMIC:
#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC
      return iree_dynamic_library_append_symbol_path_to_builder(
          (void*)libvulkan->vkGetInstanceProcAddr, builder);
#else
      return iree_string_builder_append_cstring(builder, "<statically linked>");
#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC
    case IREE_HAL_VULKAN_LIBVULKAN_SOURCE_NONE:
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Vulkan loader is not initialized");
  }
}
