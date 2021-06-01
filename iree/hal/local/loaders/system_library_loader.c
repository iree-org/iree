// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/system_library_loader.h"

#include "iree/base/tracing.h"
#include "iree/hal/local/local_executable.h"

// flatcc schemas:
#include "iree/base/internal/flatcc.h"

//===----------------------------------------------------------------------===//
// iree_hal_system_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_system_executable_t {
  iree_hal_local_executable_t base;

  // TODO(benvanik): library handle for ownership.

  union {
    const iree_hal_executable_library_header_t* header;
    const iree_hal_executable_library_v0_t* v0;
  } library;
} iree_hal_system_executable_t;

static const iree_hal_local_executable_vtable_t
    iree_hal_system_executable_vtable;

static iree_status_t iree_hal_system_executable_create(
    iree_hal_executable_layout_t* base_layout,
    const iree_hal_executable_library_header_t* library_header,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(library_header);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_system_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_layout_count * sizeof(iree_hal_local_executable_layout_t);
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_layout_t** executable_layouts_ptr =
        (iree_hal_local_executable_layout_t**)(((uint8_t*)executable) +
                                               sizeof(*executable));
    iree_hal_local_executable_initialize(
        &iree_hal_system_executable_vtable, executable_layout_count,
        executable_layouts, executable_layouts_ptr, host_allocator,
        &executable->base);
    executable->library.header = library_header;
    *out_executable = (iree_hal_executable_t*)executable;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_system_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;

  iree_host_size_t ordinal_count = executable->library.v0->entry_point_count;
  if (IREE_UNLIKELY(ordinal >= ordinal_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_string_view_t entry_point_name = iree_make_cstring_view(
      executable->library.v0->entry_point_names[ordinal]);
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_dylib_call");
  }
  IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, entry_point_name.data,
                                      entry_point_name.size);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  int ret = executable->library.v0->entry_points[ordinal](dispatch_state,
                                                          workgroup_id);

  IREE_TRACE_ZONE_END(z0);

  return ret == 0 ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_INTERNAL,
                        "executable entry point returned catastrophic error %d",
                        ret);
}

static const iree_hal_local_executable_vtable_t
    iree_hal_system_executable_vtable = {
        .base =
            {
                .destroy = iree_hal_system_executable_destroy,
            },
        .issue_call = iree_hal_system_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_system_library_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_system_library_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_system_library_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable;

iree_status_t iree_hal_system_library_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_system_library_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_system_library_loader_vtable, &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_library_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_system_library_loader_t* executable_loader =
      (iree_hal_system_library_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_system_library_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("DYEX"));
}

static iree_status_t iree_hal_system_library_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                       "new executable library format not yet implemented");

  // Query the executable library to get the latest interface.
  // Will fail if the executable is using a newer interface than we support.
  // iree_hal_executable_library_header_t* header = NULL;
  // IREE_RETURN_AND_END_ZONE_IF_ERROR(
  //     z0, iree_hal_executable_library_handle_query(
  //             executable_handle, IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION,
  //             &header));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable = {
        .destroy = iree_hal_system_library_loader_destroy,
        .query_support = iree_hal_system_library_loader_query_support,
        .try_load = iree_hal_system_library_loader_try_load,
};
