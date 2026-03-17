// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/libverbs.h"

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"

//===----------------------------------------------------------------------===//
// iree_net_libverbs_t
//===----------------------------------------------------------------------===//

#if !IREE_NET_LIBVERBS_STATIC

static iree_status_t iree_net_libverbs_load_symbols(
    iree_dynamic_library_t* library, iree_net_libverbs_t* out_libverbs) {
#define IREE_NET_LIBVERBS_PFN(result_type, symbol, ...)    \
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol( \
      library, #symbol, (void**)&out_libverbs->symbol));
#define DECL(...)
#define ARGS(...)
#include "iree/net/carrier/rdma/libverbs_tables.h"  // IWYU pragma: keep
  return iree_ok_status();
}

static void iree_net_libverbs_load_optional_symbols(
    iree_dynamic_library_t* library, iree_net_libverbs_t* out_libverbs) {
  // ibv_reg_dmabuf_mr is optional (kernel 5.12+, libibverbs 34+).
  // NULL means not available.
  out_libverbs->ibv_reg_dmabuf_mr =
      iree_dynamic_library_try_lookup_symbol(library, "ibv_reg_dmabuf_mr");
}

static bool iree_net_libverbs_try_load_library_from_file(
    const char* file_path, iree_string_builder_t* error_builder,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, file_path);
  *out_library = NULL;

  iree_status_t status = iree_dynamic_library_load_from_file(
      file_path, IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, out_library);

  if (!iree_status_is_ok(status)) {
    IREE_IGNORE_ERROR(iree_string_builder_append_format(
        error_builder, "\n  Tried: %s\n    ", file_path));
    IREE_IGNORE_ERROR(iree_string_builder_append_status(error_builder, status));
  }

  iree_status_ignore(status);
  IREE_TRACE_ZONE_END(z0);
  return *out_library != NULL;
}

static const char* iree_net_libverbs_names[] = {
    "libibverbs.so.1",
    "libibverbs.so",
};

static bool iree_net_libverbs_try_load_library_from_path(
    iree_string_view_t path_fragment, iree_string_builder_t* error_builder,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path_fragment.data, path_fragment.size);
  *out_library = NULL;

  iree_string_builder_t path_builder;
  iree_string_builder_initialize(host_allocator, &path_builder);

  if (iree_file_path_is_dynamic_library(path_fragment)) {
    // User provided a filename - try to use it directly.
    iree_status_t status =
        iree_string_builder_append_string(&path_builder, path_fragment);
    if (iree_status_is_ok(status)) {
      iree_net_libverbs_try_load_library_from_file(
          iree_string_builder_buffer(&path_builder), error_builder,
          host_allocator, out_library);
    } else {
      iree_status_ignore(status);
      // OOM - skip this path, don't try to load from corrupted buffer.
    }
  } else {
    // Join the provided path with each canonical name and try that.
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_net_libverbs_names) && !*out_library; ++i) {
      iree_string_builder_reset(&path_builder);
      iree_status_t status = iree_string_builder_append_format(
          &path_builder, "%.*s/%s", (int)path_fragment.size, path_fragment.data,
          iree_net_libverbs_names[i]);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        // OOM - skip this path.
        continue;
      }
      path_builder.size = iree_file_path_canonicalize(
          (char*)iree_string_builder_buffer(&path_builder),
          iree_string_builder_size(&path_builder));
      iree_net_libverbs_try_load_library_from_file(
          iree_string_builder_buffer(&path_builder), error_builder,
          host_allocator, out_library);
    }
  }

  iree_string_builder_deinitialize(&path_builder);

  IREE_TRACE_ZONE_END(z0);
  return *out_library != NULL;
}

static iree_status_t iree_net_libverbs_load_library(
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_net_libverbs_t* out_libverbs) {
  IREE_ASSERT_ARGUMENT(out_libverbs);
  IREE_TRACE_ZONE_BEGIN(z0);
  out_libverbs->library = NULL;

  iree_string_builder_t error_builder;
  iree_string_builder_initialize(host_allocator, &error_builder);

  iree_dynamic_library_t* library = NULL;

  // If the caller provided explicit paths we always try to use those first.
  for (iree_host_size_t i = 0; i < search_paths.count && !library; ++i) {
    iree_net_libverbs_try_load_library_from_path(
        search_paths.values[i], &error_builder, host_allocator, &library);
  }

  // If no user path provided the library try the environment variable.
  iree_string_view_t env_path =
      iree_make_cstring_view(getenv("IREE_NET_LIBVERBS_PATH"));
  if (!library && !iree_string_view_is_empty(env_path)) {
    iree_net_libverbs_try_load_library_from_path(env_path, &error_builder,
                                                 host_allocator, &library);
  }

  // Fallback: try loading with the canonical library names from system paths.
  if (!library) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_net_libverbs_names) && !library; ++i) {
      if (iree_net_libverbs_try_load_library_from_file(
              iree_net_libverbs_names[i], &error_builder, host_allocator,
              &library)) {
        break;
      }
    }
  }

  iree_status_t status = iree_ok_status();
  if (!library) {
    status =
        iree_make_status(IREE_STATUS_NOT_FOUND,
                         "libibverbs library not found; ensure rdma-core is "
                         "installed and on a valid search path (or specified "
                         "with IREE_NET_LIBVERBS_PATH): %.*s",
                         (int)iree_string_builder_size(&error_builder),
                         iree_string_builder_buffer(&error_builder));
  }
  iree_string_builder_deinitialize(&error_builder);

  if (iree_status_is_ok(status)) {
    status = iree_net_libverbs_load_symbols(library, out_libverbs);
    if (!iree_status_is_ok(status) && out_libverbs->ibv_get_device_list) {
      iree_string_builder_t annotation_builder;
      iree_string_builder_initialize(host_allocator, &annotation_builder);
      IREE_IGNORE_ERROR(iree_dynamic_library_append_symbol_path_to_builder(
          out_libverbs->ibv_get_device_list, &annotation_builder));
      status = iree_status_annotate_f(
          status, "using %.*s",
          (int)iree_string_builder_size(&annotation_builder),
          iree_string_builder_buffer(&annotation_builder));
      iree_string_builder_deinitialize(&annotation_builder);
    }
  }

  if (iree_status_is_ok(status)) {
    iree_net_libverbs_load_optional_symbols(library, out_libverbs);
    out_libverbs->library = library;
  } else if (library) {
    iree_dynamic_library_release(library);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_net_libverbs_unload_library(iree_net_libverbs_t* libverbs) {
  if (libverbs->library) {
    iree_dynamic_library_release(libverbs->library);
  }
}

IREE_API_EXPORT iree_status_t iree_net_libverbs_append_path_to_builder(
    const iree_net_libverbs_t* libverbs, iree_string_builder_t* builder) {
  return iree_dynamic_library_append_symbol_path_to_builder(
      libverbs->ibv_get_device_list, builder);
}

#else  // IREE_NET_LIBVERBS_STATIC

static iree_status_t iree_net_libverbs_load_library(
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_net_libverbs_t* out_libverbs) {
  return iree_ok_status();
}

static void iree_net_libverbs_unload_library(iree_net_libverbs_t* libverbs) {}

IREE_API_EXPORT iree_status_t iree_net_libverbs_append_path_to_builder(
    const iree_net_libverbs_t* libverbs, iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "<statically linked>");
}

#endif  // !IREE_NET_LIBVERBS_STATIC

IREE_API_EXPORT iree_status_t iree_net_libverbs_initialize(
    iree_string_view_list_t search_paths, iree_allocator_t host_allocator,
    iree_net_libverbs_t* out_libverbs) {
  IREE_ASSERT_ARGUMENT(out_libverbs);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_libverbs, 0, sizeof(*out_libverbs));
  iree_status_t status = iree_net_libverbs_load_library(
      search_paths, host_allocator, out_libverbs);

  if (!iree_status_is_ok(status)) {
    iree_net_libverbs_deinitialize(out_libverbs);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_net_libverbs_deinitialize(
    iree_net_libverbs_t* libverbs) {
  IREE_ASSERT_ARGUMENT(libverbs);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_libverbs_unload_library(libverbs);
  memset(libverbs, 0, sizeof(*libverbs));

  IREE_TRACE_ZONE_END(z0);
}
