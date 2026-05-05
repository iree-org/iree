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
// iree_hal_vulkan_libvulkan_t
//===----------------------------------------------------------------------===//

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

static iree_status_t iree_hal_vulkan_libvulkan_load_symbols(
    iree_dynamic_library_t* library,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  return iree_dynamic_library_lookup_symbol(
      library, "vkGetInstanceProcAddr",
      (void**)&out_libvulkan->vkGetInstanceProcAddr);
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
  return iree_ok_status();
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
