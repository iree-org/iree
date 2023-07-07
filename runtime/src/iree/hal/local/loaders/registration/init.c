// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/registration/init.h"

// NOTE: we register in a specific order to allow for prioritization:
// - system-library: used when embedded is not desired (TSAN/debugging/etc).
// - embedded-elf: default codegen portable ELF output format.
// - vmvx-module: reference fallback path using the IREE bytecode VM.

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY)
#include "iree/hal/local/loaders/system_library_loader.h"
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
#include "iree/hal/local/loaders/embedded_elf_loader.h"
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE

IREE_API_EXPORT iree_status_t iree_hal_create_all_available_executable_loaders(
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_host_size_t capacity, iree_host_size_t* out_count,
    iree_hal_executable_loader_t** loaders, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(out_count);
  IREE_ASSERT(!capacity || loaders);
  *out_count = 0;

  iree_host_size_t required_capacity = 0;
#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY)
  ++required_capacity;
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY
#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  ++required_capacity;
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF
#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
  ++required_capacity;
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE
  if (capacity < required_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
  }

  memset(loaders, 0, sizeof(*loaders) * required_capacity);

  iree_host_size_t count = 0;
  iree_status_t status = iree_ok_status();

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY)
  if (iree_status_is_ok(status)) {
    status = iree_hal_system_library_loader_create(
        plugin_manager, host_allocator, &loaders[count++]);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  if (iree_status_is_ok(status)) {
    status = iree_hal_embedded_elf_loader_create(plugin_manager, host_allocator,
                                                 &loaders[count++]);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
  if (iree_status_is_ok(status)) {
    status = iree_hal_vmvx_module_loader_create_isolated(
        /*user_module_count=*/0, /*user_modules=*/NULL, host_allocator,
        &loaders[count++]);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE

  if (iree_status_is_ok(status)) {
    *out_count = count;
  } else {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_executable_loader_release(loaders[i]);
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_create_executable_loader_by_name(
    iree_string_view_t name,
    iree_hal_executable_plugin_manager_t* plugin_manager,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  if (iree_string_view_starts_with(name, IREE_SV("embedded-elf"))) {
    return iree_hal_embedded_elf_loader_create(plugin_manager, host_allocator,
                                               out_executable_loader);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY)
  if (iree_string_view_starts_with(name, IREE_SV("system-library"))) {
    return iree_hal_system_library_loader_create(plugin_manager, host_allocator,
                                                 out_executable_loader);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY

#if defined(IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE)
  if (iree_string_view_starts_with(name, IREE_SV("vmvx-module"))) {
    return iree_hal_vmvx_module_loader_create_isolated(
        /*user_module_count=*/0, /*user_modules=*/NULL, host_allocator,
        out_executable_loader);
  }
#endif  // IREE_HAVE_HAL_EXECUTABLE_LOADER_VMVX_MODULE

  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "no executable loader linked in with identifier '%.*s'", (int)name.size,
      name.data);
}
