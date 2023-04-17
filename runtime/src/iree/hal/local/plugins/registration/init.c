// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/plugins/registration/init.h"

#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"

#if defined(IREE_HAVE_HAL_EXECUTABLE_SYSTEM_LIBRARY_PLUGIN)
#include "iree/hal/local/plugins/system_library_plugin.h"
#endif  // IREE_HAVE_HAL_EXECUTABLE_SYSTEM_LIBRARY_PLUGIN

#if defined(IREE_HAVE_HAL_EXECUTABLE_EMBEDDED_ELF_PLUGIN)
#include "iree/hal/local/plugins/embedded_elf_plugin.h"
#endif  // IREE_HAVE_HAL_EXECUTABLE_EMBEDDED_ELF_PLUGIN

IREE_FLAG_LIST(
    string, executable_plugin,
    "Load a local HAL executable plugin to resolve imports.\n"
    "See iree/hal/local/executable_plugin.h for the plugin API.\n"
    "By default plugins load using the system library loader and accept\n"
    "native system formats (.dll, .so, .dylib, etc).\n"
    "\n"
    "For plugins compiled to standalone portable ELF files the embedded ELF\n"
    "loader can be used even if OS support for dynamic linking is missing or\n"
    "slow. Prefix the paths with `embedded:` or use the `.sos` extension.\n"
    "\n"
    "If multiple plugins are specified they will be scanned for imports in\n"
    "reverse registration order (last plugin checked first).\n"
    "\n"
    "Examples:\n"
    "  --executable_plugin=some/system.dll\n"
    "  --executable_plugin=some/standalone.sos\n"
    "  --executable_plugin=embedded:some/standalone.so");

iree_status_t iree_hal_executable_plugin_manager_create_from_flags(
    iree_allocator_t host_allocator,
    iree_hal_executable_plugin_manager_t** out_manager) {
  IREE_ASSERT_ARGUMENT(out_manager);
  *out_manager = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t capacity = FLAG_executable_plugin_list().count;

  iree_hal_executable_plugin_manager_t* manager = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_plugin_manager_create(capacity, host_allocator,
                                                    &manager));

  iree_status_t status =
      iree_hal_register_executable_plugins_from_flags(manager, host_allocator);

  if (iree_status_is_ok(status)) {
    *out_manager = manager;
  } else {
    iree_hal_executable_plugin_manager_release(manager);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_register_executable_plugins_from_flags(
    iree_hal_executable_plugin_manager_t* manager,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(manager);
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < FLAG_executable_plugin_list().count; ++i) {
    iree_string_view_t flag = FLAG_executable_plugin_list().values[i];
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_register_executable_plugin_from_spec(manager, flag,
                                                          host_allocator));
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_register_executable_plugin_from_spec(
    iree_hal_executable_plugin_manager_t* manager, iree_string_view_t spec,
    iree_allocator_t host_allocator) {
  // TODO(benvanik): support parameterization via ?query or something.
  iree_string_view_t path = spec;
  iree_host_size_t param_count = 0;
  iree_string_pair_t params[1];

  // NOTE: it's possible for no plugins to be enabled.
  (void)path;
  (void)param_count;
  (void)params;

  iree_status_t status = iree_ok_status();
  iree_hal_executable_plugin_t* plugin = NULL;

#if defined(IREE_HAVE_HAL_EXECUTABLE_EMBEDDED_ELF_PLUGIN) && IREE_FILE_IO_ENABLE
  if (iree_string_view_consume_prefix(&path, IREE_SV("embedded:")) ||
      iree_string_view_ends_with(path, IREE_SV(".sos"))) {
    IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_executable_plugin_load_from_file(
        path.data, param_count, params, host_allocator, &plugin));
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_EMBEDDED_ELF_PLUGIN

#if defined(IREE_HAVE_HAL_EXECUTABLE_SYSTEM_LIBRARY_PLUGIN)
  if (!plugin && iree_status_is_ok(status)) {
    status = iree_hal_system_library_executable_plugin_load_from_file(
        path.data, param_count, params, host_allocator, &plugin);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_SYSTEM_LIBRARY_PLUGIN

  if (plugin && iree_status_is_ok(status)) {
    status =
        iree_hal_executable_plugin_manager_register_plugin(manager, plugin);
  }

  iree_hal_executable_plugin_release(plugin);
  return status;
}
