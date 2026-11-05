// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/embedded_elf_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/api.h"
#include "iree/hal/local/elf/elf_module.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/executable_library_util.h"
#include "iree/hal/local/executable_plugin_manager.h"
#include "iree/hal/local/local_executable.h"
#include "nl_api.h"

//===----------------------------------------------------------------------===//
// iree_hal_nonlocal_elf_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_nonlocal_elf_executable_t {
  iree_hal_local_executable_t base;

  // Loaded ELF module.
  iree_elf_module_t *module;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

  // Queried metadata from the library.
  iree_hal_executable_library_v0_t *library;
} iree_hal_nonlocal_elf_executable_t;

static const iree_hal_local_executable_vtable_t iree_hal_nonlocal_elf_executable_vtable;

static iree_status_t iree_hal_nonlocal_elf_executable_create(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(executable_params->executable_data.data &&
                       executable_params->executable_data.data_length);
  IREE_ASSERT_ARGUMENT(!executable_params->constant_count ||
                       executable_params->constants);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): rework this so that we load and query the library before
  // allocating so that we know the import count. Today since we allocate first
  // we need an additional allocation once we've seen the import table.
  iree_hal_nonlocal_elf_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_params->constant_count * sizeof(*executable_params->constants);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_initialize(&iree_hal_nonlocal_elf_executable_vtable,
                                         host_allocator, &executable->base);
  }

  // Copy executable constants so we own them.
  if (iree_status_is_ok(status) && executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + sizeof(*executable));
    memcpy(target_constants, executable_params->constants,
           executable_params->constant_count *
               sizeof(*executable_params->constants));
    executable->base.environment.constants = target_constants;
  }

  // Attempt to load the ELF module.
  executable->module = nl_elf_executable_load( executable_params->executable_data.data, executable_params->executable_data.data_length);

  // Query metadata and get the entry point function pointers.
  executable->library = nl_elf_executable_init(executable->module);

  void *p;
  int n;
  nl_elf_executable_get_attrs(executable->library, &p, &n);
  executable->base.dispatch_attrs = (iree_hal_executable_dispatch_attrs_v0_t *)p;

#if 0
  // Resolve imports, if any.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_initialize_imports(
        &executable->base.environment, import_provider,
        &executable->library->imports,
        (iree_hal_executable_import_thunk_v0_t)iree_elf_thunk_i_ppp,
        host_allocator);
  }

  // Verify that the library matches the executable params.
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_library_verify(executable_params,
                                                executable->library);
  }

  // Publish the executable sources with the tracing infrastructure.
  if (iree_status_is_ok(status)) {
    iree_hal_executable_library_publish_source_files(executable->library);
  }
#endif

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_nonlocal_elf_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_nonlocal_elf_executable_t* executable =
      (iree_hal_nonlocal_elf_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  nl_elf_executable_destroy(executable->module);

  iree_hal_executable_library_deinitialize_imports(
      &executable->base.environment, host_allocator);

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_nonlocal_elf_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
    uint32_t worker_id) {
  iree_hal_nonlocal_elf_executable_t* executable =
      (iree_hal_nonlocal_elf_executable_t*)base_executable;

  int ret = nl_elf_executable_call(executable->library, ordinal,
                                (void*)dispatch_state, (void*)workgroup_state);

  return ret == 0 ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_INTERNAL,
                        "executable entry point returned catastrophic error %d",
                        ret);
}

static const iree_hal_local_executable_vtable_t iree_hal_nonlocal_elf_executable_vtable =
    {
        .base =
            {
                .destroy = iree_hal_nonlocal_elf_executable_destroy,
            },
        .issue_call = iree_hal_nonlocal_elf_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_nonlocal_embedded_elf_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_nonlocal_embedded_elf_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  iree_hal_executable_plugin_manager_t* plugin_manager;
} iree_hal_nonlocal_embedded_elf_loader_t;

static const iree_hal_executable_loader_vtable_t
    iree_hal_nonlocal_embedded_elf_loader_vtable;

iree_status_t iree_hal_nonlocal_embedded_elf_loader_create(
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_nonlocal_embedded_elf_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_nonlocal_embedded_elf_loader_vtable,
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

static void iree_hal_nonlocal_embedded_elf_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_nonlocal_embedded_elf_loader_t* executable_loader =
      (iree_hal_nonlocal_embedded_elf_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_plugin_manager_release(executable_loader->plugin_manager);
  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_nonlocal_embedded_elf_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_starts_with(
      executable_format, iree_make_cstring_view("embedded-elf-" IREE_ARCH));
}

static iree_status_t iree_hal_nonlocal_embedded_elf_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  iree_hal_nonlocal_embedded_elf_loader_t* executable_loader =
      (iree_hal_nonlocal_embedded_elf_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Perform the load of the ELF and wrap it in an executable handle.
  iree_status_t status = iree_hal_nonlocal_elf_executable_create(
      executable_params, base_executable_loader->import_provider,
      executable_loader->host_allocator, out_executable);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_executable_loader_vtable_t
    iree_hal_nonlocal_embedded_elf_loader_vtable = {
        .destroy = iree_hal_nonlocal_embedded_elf_loader_destroy,
        .query_support = iree_hal_nonlocal_embedded_elf_loader_query_support,
        .try_load = iree_hal_nonlocal_embedded_elf_loader_try_load,
};
