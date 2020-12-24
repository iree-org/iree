// Copyright 2020 Google LLC
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

#include "iree/hal/local/loaders/system_library_loader.h"

#include "iree/base/tracing.h"
#include "iree/hal/local/local_executable.h"

// flatcc schemas:
#include "iree/base/flatcc.h"
#include "iree/schemas/dylib_executable_def_reader.h"
#include "iree/schemas/dylib_executable_def_verifier.h"

//===----------------------------------------------------------------------===//
// iree_hal_system_executable_t
//===----------------------------------------------------------------------===//

typedef struct {
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
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(base_layout);
  IREE_ASSERT_ARGUMENT(library_header);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_executable_layout_t* local_layout =
      iree_hal_local_executable_layout_cast(base_layout);
  IREE_ASSERT_ARGUMENT(local_layout);

  iree_hal_system_executable_t* executable = NULL;
  iree_status_t status = iree_allocator_malloc(
      local_layout->host_allocator, sizeof(*executable), (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_initialize(&iree_hal_system_executable_vtable,
                                         local_layout, &executable->base);
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
  iree_allocator_t host_allocator =
      iree_hal_device_host_allocator(executable->base.layout->device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_executable_deinitialize(base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_system_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_local_executable_call_t* call) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;

  iree_host_size_t ordinal_count = executable->library.v0->entry_point_count;
  if (IREE_UNLIKELY(ordinal >= iree_hal_system_executable_entry_point_count(
                                   base_executable))) {
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

  executable->library.v0->entry_points[ordinal](
      call->state, call->workgroup_id, call->workgroup_size,
      call->workgroup_count, call->push_constants, call->bindings);

  IREE_TRACE_ZONE_END(z0);

  return iree_ok_status();
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

typedef struct {
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
    iree_hal_executable_format_t executable_format,
    iree_hal_executable_caching_mode_t caching_mode) {
  return executable_format == iree_hal_make_executable_format("DYEX");
}

static iree_status_t iree_hal_system_library_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_layout_t* executable_layout,
    iree_hal_executable_format_t executable_format,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
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
