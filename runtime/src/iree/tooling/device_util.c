// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_util.h"

#include "iree/base/internal/call_once.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/init.h"

//===----------------------------------------------------------------------===//
// Shared driver registry
//===----------------------------------------------------------------------===//

static iree_once_flag iree_hal_driver_registry_init_flag = IREE_ONCE_FLAG_INIT;
static void iree_hal_driver_registry_init_from_flags(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
  IREE_TRACE_ZONE_END(z0);
}

iree_hal_driver_registry_t* iree_hal_available_driver_registry(void) {
  iree_call_once(&iree_hal_driver_registry_init_flag,
                 iree_hal_driver_registry_init_from_flags);
  return iree_hal_driver_registry_default();
}

//===----------------------------------------------------------------------===//
// Driver and device listing commands
//===----------------------------------------------------------------------===//

static void iree_hal_flags_print_action_header(void) {
  fprintf(stdout,
          "# "
          "===================================================================="
          "========\n");
}

static void iree_hal_flags_print_action_separator(void) {
  fprintf(stdout,
          "# "
          "===-----------------------------------------------------------"
          "-----------===\n");
}

static void iree_hal_flags_print_action_flag(iree_string_view_t flag_name,
                                             void* storage, FILE* file) {
  fprintf(file, "# --%.*s\n", (int)flag_name.size, flag_name.data);
}

static iree_status_t iree_hal_flags_list_drivers(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value) {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_hal_flags_print_action_header();
  fprintf(stdout, "# Available HAL drivers\n");
  iree_hal_flags_print_action_header();
  fprintf(
      stdout,
      "# Use --list_devices={driver name} to enumerate available devices.\n\n");

  iree_hal_driver_registry_t* driver_registry =
      iree_hal_available_driver_registry();
  iree_host_size_t driver_info_count = 0;
  iree_hal_driver_info_t* driver_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
      driver_registry, host_allocator, &driver_info_count, &driver_infos));

  for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
    const iree_hal_driver_info_t* driver_info = &driver_infos[i];
    fprintf(stdout, "%16.*s: %.*s\n", (int)driver_info->driver_name.size,
            driver_info->driver_name.data, (int)driver_info->full_name.size,
            driver_info->full_name.data);
  }

  iree_allocator_free(host_allocator, driver_infos);

  exit(0);
  return iree_ok_status();
}

IREE_FLAG_CALLBACK(iree_hal_flags_list_drivers,
                   iree_hal_flags_print_action_flag, NULL, list_drivers,
                   "Lists all available HAL drivers compiled into the binary.");

static iree_status_t iree_hal_flags_list_driver_device(
    iree_string_view_t driver_name, iree_hal_driver_t* driver,
    const iree_hal_device_info_t* device_info, iree_allocator_t host_allocator,
    FILE* file) {
  fprintf(file, "%.*s://%.*s\n", (int)driver_name.size, driver_name.data,
          (int)device_info->path.size, device_info->path.data);
  return iree_ok_status();
}

static iree_status_t iree_hal_flags_list_driver_devices(
    iree_hal_driver_registry_t* driver_registry, iree_string_view_t driver_name,
    iree_allocator_t host_allocator, FILE* file) {
  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
      driver_registry, driver_name, host_allocator, &driver));

  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  iree_status_t status = iree_hal_driver_query_available_devices(
      driver, host_allocator, &device_info_count, &device_infos);

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < device_info_count; ++i) {
    status = iree_hal_flags_list_driver_device(
        driver_name, driver, &device_infos[i], host_allocator, file);
  }

  iree_allocator_free(host_allocator, device_infos);
  iree_hal_driver_release(driver);
  return iree_ok_status();
}

static iree_status_t iree_hal_flags_list_devices(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_driver_registry_t* driver_registry =
      iree_hal_available_driver_registry();

  if (iree_string_view_is_empty(value)) {
    // List all devices on all drivers.
    iree_host_size_t driver_info_count = 0;
    iree_hal_driver_info_t* driver_infos = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
        driver_registry, host_allocator, &driver_info_count, &driver_infos));
    for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
      const iree_hal_driver_info_t* driver_info = &driver_infos[i];
      IREE_RETURN_IF_ERROR(iree_hal_flags_list_driver_devices(
          driver_registry, driver_info->driver_name, host_allocator, stdout));
    }
    iree_allocator_free(host_allocator, driver_infos);
  } else {
    // List all devices from a particular driver.
    IREE_RETURN_IF_ERROR(iree_hal_flags_list_driver_devices(
        driver_registry, value, host_allocator, stdout));
  }

  exit(0);
  return iree_ok_status();
}

IREE_FLAG_CALLBACK(
    iree_hal_flags_list_devices, iree_hal_flags_print_action_flag, NULL,
    list_devices,
    "Lists all available HAL devices from all drivers or a specific driver.\n"
    "Examples:\n"
    "  Show all devices from all drivers: --list_devices\n"
    "  Show all devices from a particular driver: --list_devices=vulkan\n");

static iree_status_t iree_hal_flags_dump_driver_device(
    iree_string_view_t driver_name, iree_hal_driver_t* driver,
    const iree_hal_device_info_t* device_info, iree_allocator_t host_allocator,
    FILE* file) {
  iree_hal_flags_print_action_separator();
  fprintf(file, "# --device=%.*s://%.*s\n", (int)driver_name.size,
          driver_name.data, (int)device_info->path.size,
          device_info->path.data);
  // TODO(benvanik): driver full name.
  fprintf(file, "#   %.*s\n", (int)device_info->name.size,
          device_info->name.data);
  iree_hal_flags_print_action_separator();

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  IREE_RETURN_IF_ERROR(iree_hal_driver_dump_device_info(
      driver, device_info->device_id, &builder));
  if (iree_string_builder_size(&builder) > 0) {
    fprintf(file, "%.*s", (int)iree_string_builder_size(&builder),
            iree_string_builder_buffer(&builder));
  }

  iree_string_builder_deinitialize(&builder);

  return iree_ok_status();
}

static iree_status_t iree_hal_flags_dump_driver_devices(
    iree_hal_driver_registry_t* driver_registry, iree_string_view_t driver_name,
    iree_allocator_t host_allocator, FILE* file) {
  iree_hal_flags_print_action_header();
  fprintf(stdout, "# Enumerated devices for driver '%.*s'\n",
          (int)driver_name.size, driver_name.data);
  iree_hal_flags_print_action_header();
  fprintf(stdout, "\n");

  iree_hal_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
      driver_registry, driver_name, host_allocator, &driver));

  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  iree_status_t status = iree_hal_driver_query_available_devices(
      driver, host_allocator, &device_info_count, &device_infos);

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < device_info_count; ++i) {
    status = iree_hal_flags_dump_driver_device(
        driver_name, driver, &device_infos[i], host_allocator, file);
  }

  iree_allocator_free(host_allocator, device_infos);
  iree_hal_driver_release(driver);
  return iree_ok_status();
}

static iree_status_t iree_hal_flags_dump_devices(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_hal_driver_registry_t* driver_registry =
      iree_hal_available_driver_registry();

  if (iree_string_view_is_empty(value)) {
    // List all devices on all drivers.
    iree_host_size_t driver_info_count = 0;
    iree_hal_driver_info_t* driver_infos = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_driver_registry_enumerate(
        driver_registry, host_allocator, &driver_info_count, &driver_infos));
    for (iree_host_size_t i = 0; i < driver_info_count; ++i) {
      if (i) fprintf(stdout, "\n");
      const iree_hal_driver_info_t* driver_info = &driver_infos[i];
      IREE_RETURN_IF_ERROR(iree_hal_flags_dump_driver_devices(
          driver_registry, driver_info->driver_name, host_allocator, stdout));
    }
    iree_allocator_free(host_allocator, driver_infos);
  } else {
    // List all devices from a particular driver.
    IREE_RETURN_IF_ERROR(iree_hal_flags_dump_driver_devices(
        driver_registry, value, host_allocator, stdout));
  }

  exit(0);
  return iree_ok_status();
}

IREE_FLAG_CALLBACK(
    iree_hal_flags_dump_devices, iree_hal_flags_print_action_flag, NULL,
    dump_devices,
    "Dumps detailed information on all available HAL devices from all drivers\n"
    " or a specific driver.\n"
    "Examples:\n"
    "  Show all devices from all drivers: --dump_devices\n"
    "  Show all devices from a particular driver: --dump_devices=vulkan\n");
