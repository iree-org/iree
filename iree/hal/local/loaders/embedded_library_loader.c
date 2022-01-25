// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/embedded_library_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/local_executable_layout.h"

//===----------------------------------------------------------------------===//
// iree_hal_elf_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_elf_executable_t {
  iree_hal_local_executable_t base;

  // Loaded ELF module.
  iree_elf_module_t module;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

  // Queried metadata from the library.
  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;

  iree_hal_local_executable_layout_t* layouts[];
} iree_hal_elf_executable_t;

static const iree_hal_local_executable_vtable_t iree_hal_elf_executable_vtable;

static iree_status_t iree_hal_elf_executable_query_library(
    iree_hal_elf_executable_t* executable) {
  // Get the exported symbol used to get the library metadata.
  iree_hal_executable_library_query_fn_t query_fn = NULL;
  IREE_RETURN_IF_ERROR(iree_elf_module_lookup_export(
      &executable->module, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME,
      (void**)&query_fn));

  // Query for a compatible version of the library.
  executable->library.header =
      (const iree_hal_executable_library_header_t**)iree_elf_call_p_ip(
          query_fn, IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION,
          /*reserved=*/NULL);
  if (!executable->library.header) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable does not support this version of the runtime (%d)",
        IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION);
  }
  const iree_hal_executable_library_header_t* header =
      *executable->library.header;

  // Ensure that if the library is built for a particular sanitizer that we also
  // were compiled with that sanitizer enabled.
  switch (header->sanitizer) {
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE:
      // Always safe even if the host has a sanitizer enabled; it just means
      // that we won't be able to catch anything from within the executable,
      // however checks outside will (often) still trigger when guard pages are
      // dirtied/etc.
      break;
    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "executable requires sanitizer but they are not "
                              "yet supported with embedded libraries: %u",
                              (uint32_t)header->sanitizer);
  }

  executable->identifier = iree_make_cstring_view(header->name);

  executable->base.dispatch_attrs = executable->library.v0->exports.attrs;

  return iree_ok_status();
}

// Resolves all of the imports declared by the executable using the given
// |import_provider|.
static iree_status_t iree_hal_elf_executable_resolve_imports(
    iree_hal_elf_executable_t* executable,
    const iree_hal_executable_import_provider_t import_provider) {
  const iree_hal_executable_import_table_v0_t* import_table =
      &executable->library.v0->imports;
  if (!import_table->count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // All calls from the loaded ELF route through our thunk function so that we
  // can adapt to ABI differences.
  executable->base.import_thunk =
      (iree_hal_executable_import_thunk_v0_t)iree_elf_thunk_i_p;

  // Allocate storage for the imports.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              executable->base.host_allocator,
              import_table->count * sizeof(*executable->base.imports),
              (void**)&executable->base.imports));

  // Try to resolve each import.
  // NOTE: imports are sorted alphabetically and if we cared we could use this
  // information to more efficiently resolve the symbols from providers (O(n)
  // walk vs potential O(nlogn)/O(n^2)).
  for (uint32_t i = 0; i < import_table->count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_executable_import_provider_resolve(
            import_provider, iree_make_cstring_view(import_table->symbols[i]),
            (void**)&executable->base.imports[i]));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_elf_executable_create(
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t elf_data, iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    const iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(elf_data.data && elf_data.data_length);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): rework this so that we load and query the library before
  // allocating so that we know the import count. Today since we allocate first
  // we need an additional allocation once we've seen the import table.
  iree_hal_elf_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_layout_count * sizeof(*executable->layouts);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_initialize(
        &iree_hal_elf_executable_vtable, executable_layout_count,
        executable_layouts, &executable->layouts[0], host_allocator,
        &executable->base);
  }
  if (iree_status_is_ok(status)) {
    // Attempt to load the ELF module.
    status = iree_elf_module_initialize_from_memory(
        elf_data, /*import_table=*/NULL, host_allocator, &executable->module);
  }
  if (iree_status_is_ok(status)) {
    // Query metadata and get the entry point function pointers.
    status = iree_hal_elf_executable_query_library(executable);
  }
  if (iree_status_is_ok(status)) {
    // Resolve imports, if any.
    status =
        iree_hal_elf_executable_resolve_imports(executable, import_provider);
  }
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(
          caching_mode,
          IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION)) {
    // Check to make sure that the entry point count matches the layouts
    // provided.
    if (executable->library.v0->exports.count != executable_layout_count) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable provides %u entry points but caller "
          "provided %zu; must match",
          executable->library.v0->exports.count, executable_layout_count);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_elf_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_elf_module_deinitialize(&executable->module);

  if (executable->base.imports != NULL) {
    iree_allocator_free(host_allocator, (void*)executable->base.imports);
  }

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_elf_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id, iree_byte_span_t local_memory) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  const iree_hal_executable_library_v0_t* library = executable->library.v0;

  if (IREE_UNLIKELY(ordinal >= library->exports.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_string_view_t entry_point_name = iree_string_view_empty();
  if (library->exports.names != NULL) {
    entry_point_name = iree_make_cstring_view(library->exports.names[ordinal]);
  }
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_elf_call");
  }
  IREE_TRACE_ZONE_BEGIN_EXTERNAL(
      z0, executable->identifier.data, executable->identifier.size, ordinal,
      entry_point_name.data, entry_point_name.size, NULL, 0);
  if (library->exports.tags != NULL) {
    const char* tag = library->exports.tags[ordinal];
    if (tag) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, tag);
    }
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  int ret =
      iree_elf_call_i_ppp(library->exports.ptrs[ordinal], (void*)dispatch_state,
                          (void*)workgroup_id, (void*)local_memory.data);

  IREE_TRACE_ZONE_END(z0);

  return ret == 0 ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_INTERNAL,
                        "executable entry point returned catastrophic error %d",
                        ret);
}

static const iree_hal_local_executable_vtable_t iree_hal_elf_executable_vtable =
    {
        .base =
            {
                .destroy = iree_hal_elf_executable_destroy,
            },
        .issue_call = iree_hal_elf_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_embedded_library_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_embedded_library_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_embedded_library_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_library_loader_vtable;

iree_status_t iree_hal_embedded_library_loader_create(
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_embedded_library_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_embedded_library_loader_vtable, import_provider,
        &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_embedded_library_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_embedded_library_loader_t* executable_loader =
      (iree_hal_embedded_library_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_embedded_library_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(
      executable_format, iree_make_cstring_view("embedded-elf-" IREE_ARCH));
}

static iree_status_t iree_hal_embedded_library_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_embedded_library_loader_t* executable_loader =
      (iree_hal_embedded_library_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform the load of the ELF and wrap it in an executable handle.
  iree_status_t status = iree_hal_elf_executable_create(
      executable_spec->caching_mode, executable_spec->executable_data,
      executable_spec->executable_layout_count,
      executable_spec->executable_layouts,
      base_executable_loader->import_provider,
      executable_loader->host_allocator, out_executable);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_library_loader_vtable = {
        .destroy = iree_hal_embedded_library_loader_destroy,
        .query_support = iree_hal_embedded_library_loader_query_support,
        .try_load = iree_hal_embedded_library_loader_try_load,
};
