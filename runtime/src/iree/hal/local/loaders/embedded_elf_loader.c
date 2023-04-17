// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/embedded_elf_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/executable_library_util.h"
#include "iree/hal/local/executable_plugin_manager.h"
#include "iree/hal/local/local_executable.h"

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

  iree_hal_pipeline_layout_t* layouts[];
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
          query_fn, IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
          &executable->base.environment);
  if (!executable->library.header) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable does not support this version of the runtime (%08X)",
        IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST);
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

static iree_status_t iree_hal_elf_executable_create(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(executable_params->executable_data.data &&
                       executable_params->executable_data.data_length);
  IREE_ASSERT_ARGUMENT(!executable_params->pipeline_layout_count ||
                       executable_params->pipeline_layouts);
  IREE_ASSERT_ARGUMENT(!executable_params->constant_count ||
                       executable_params->constants);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): rework this so that we load and query the library before
  // allocating so that we know the import count. Today since we allocate first
  // we need an additional allocation once we've seen the import table.
  iree_hal_elf_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_params->pipeline_layout_count * sizeof(*executable->layouts) +
      executable_params->constant_count * sizeof(*executable_params->constants);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_initialize(
        &iree_hal_elf_executable_vtable,
        executable_params->pipeline_layout_count,
        executable_params->pipeline_layouts, &executable->layouts[0],
        host_allocator, &executable->base);
  }

  // Copy executable constants so we own them.
  if (iree_status_is_ok(status) && executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + sizeof(*executable) +
                    executable_params->pipeline_layout_count *
                        sizeof(*executable->layouts));
    memcpy(target_constants, executable_params->constants,
           executable_params->constant_count *
               sizeof(*executable_params->constants));
    executable->base.environment.constants = target_constants;
  }

  // Attempt to load the ELF module.
  if (iree_status_is_ok(status)) {
    status = iree_elf_module_initialize_from_memory(
        executable_params->executable_data, /*import_table=*/NULL,
        host_allocator, &executable->module);
  }

  // Query metadata and get the entry point function pointers.
  if (iree_status_is_ok(status)) {
    status = iree_hal_elf_executable_query_library(executable);
  }

  // Resolve imports, if any.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_initialize_imports(
        &executable->base.environment, import_provider,
        &executable->library.v0->imports,
        (iree_hal_executable_import_thunk_v0_t)iree_elf_thunk_i_ppp,
        host_allocator);
  }

  // Verify that the library matches the executable params.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_verify(executable_params,
                                                executable->library.v0);
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

  iree_hal_executable_library_deinitialize_imports(
      &executable->base.environment, host_allocator);

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_elf_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  iree_hal_elf_executable_t* executable =
      (iree_hal_elf_executable_t*)base_executable;
  const iree_hal_executable_library_v0_t* library = executable->library.v0;

  if (IREE_UNLIKELY(ordinal >= library->exports.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

  IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN(z0, executable->identifier,
                                                    library, ordinal);
  int ret = iree_elf_call_i_ppp(library->exports.ptrs[ordinal],
                                (void*)&base_executable->environment,
                                (void*)dispatch_state, (void*)workgroup_state);
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
// iree_hal_embedded_elf_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_embedded_elf_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  iree_hal_executable_plugin_manager_t* plugin_manager;
} iree_hal_embedded_elf_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_elf_loader_vtable;

iree_status_t iree_hal_embedded_elf_loader_create(
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_embedded_elf_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_embedded_elf_loader_vtable,
        iree_hal_executable_plugin_manager_provider(plugin_manager),
        &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    executable_loader->plugin_manager = plugin_manager;
    iree_hal_executable_plugin_manager_retain(
        executable_loader->plugin_manager);
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_embedded_elf_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_embedded_elf_loader_t* executable_loader =
      (iree_hal_embedded_elf_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_plugin_manager_release(executable_loader->plugin_manager);
  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_embedded_elf_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_starts_with(
      executable_format, iree_make_cstring_view("embedded-elf-" IREE_ARCH));
}

static iree_status_t iree_hal_embedded_elf_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  iree_hal_embedded_elf_loader_t* executable_loader =
      (iree_hal_embedded_elf_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform the load of the ELF and wrap it in an executable handle.
  iree_status_t status = iree_hal_elf_executable_create(
      executable_params, base_executable_loader->import_provider,
      executable_loader->host_allocator, out_executable);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_embedded_elf_loader_vtable = {
        .destroy = iree_hal_embedded_elf_loader_destroy,
        .query_support = iree_hal_embedded_elf_loader_query_support,
        .try_load = iree_hal_embedded_elf_loader_try_load,
};
