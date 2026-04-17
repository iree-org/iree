// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_util.h"

#include "iree/base/threading/call_once.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/init.h"
#include "iree/hal/utils/allocators.h"
#include "iree/hal/utils/mpi_channel_provider.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_handle.h"

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

iree_string_view_t iree_hal_default_device_uri(void) {
  // TODO(benvanik): query the registry and find the first available :shrug:
  return IREE_SV("local-task");
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
    iree_allocator_t host_allocator, bool fail_on_driver_error, FILE* file) {
  iree_hal_driver_t* driver = NULL;
  iree_status_t driver_status = iree_hal_driver_registry_try_create(
      driver_registry, driver_name, host_allocator, &driver);
  if (!iree_status_is_ok(driver_status) && fail_on_driver_error) {
    // Driver could not be created (not found, unavailable, etc) and user asked
    // for it explicitly and should be notified.
    return driver_status;
  } else if (!iree_status_is_ok(driver_status)) {
    // Driver could not be created so there are no devices to list.
    iree_status_ignore(driver_status);
    return iree_ok_status();
  }

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
  return status;
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
          driver_registry, driver_info->driver_name, host_allocator,
          /*fail_on_driver_error=*/false, stdout));
    }
    iree_allocator_free(host_allocator, driver_infos);
  } else {
    // List all devices from a particular driver.
    IREE_RETURN_IF_ERROR(iree_hal_flags_list_driver_devices(
        driver_registry, value, host_allocator, /*fail_on_driver_error=*/true,
        stdout));
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
    "  Show all devices from a particular driver: --list_devices=vulkan");

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
    iree_allocator_t host_allocator, bool fail_on_driver_error, FILE* file) {
  iree_hal_flags_print_action_header();
  fprintf(stdout, "# Enumerated devices for driver '%.*s'\n",
          (int)driver_name.size, driver_name.data);
  iree_hal_flags_print_action_header();
  fprintf(stdout, "\n");

  iree_hal_driver_t* driver = NULL;
  iree_status_t driver_status = iree_hal_driver_registry_try_create(
      driver_registry, driver_name, host_allocator, &driver);
  if (!iree_status_is_ok(driver_status) && fail_on_driver_error) {
    // Driver could not be created (not found, unavailable, etc) and user asked
    // for it explicitly and should be notified.
    return driver_status;
  } else if (!iree_status_is_ok(driver_status)) {
    // Driver could not be created so there are no devices to list.
    iree_status_ignore(driver_status);
    return iree_ok_status();
  }

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
  return status;
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
          driver_registry, driver_info->driver_name, host_allocator,
          /*fail_on_driver_error=*/false, stdout));
    }
    iree_allocator_free(host_allocator, driver_infos);
  } else {
    // List all devices from a particular driver.
    IREE_RETURN_IF_ERROR(iree_hal_flags_dump_driver_devices(
        driver_registry, value, host_allocator, /*fail_on_driver_error=*/true,
        stdout));
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
    "  Show all devices from a particular driver: --dump_devices=vulkan");

//===----------------------------------------------------------------------===//
// Allocator configuration
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, device_allocator,
    "Specifies one or more HAL device allocator specs to augment the base\n"
    "device allocator. See each allocator type for supported configurations.");

// Configures the |device| allocator based on the --device_allocator= flag.
// This will wrap the underlying device allocator in zero or more configurable
// allocator shims.
//
// WARNING: not thread-safe and must only be called immediately after device
// creation.
static iree_status_t iree_hal_configure_allocator_from_flags(
    iree_hal_device_t* device) {
  const iree_flag_string_list_t list = FLAG_device_allocator_list();
  return iree_hal_configure_allocator_from_specs(list.count, list.values,
                                                 device);
}

//===----------------------------------------------------------------------===//
// Collectives configuration
//===----------------------------------------------------------------------===//

// TODO(multi-device): support more provider types/have a provider registry.
// MPI is insufficient for heterogeneous/multi-device configurations. Currently
// we set the same provider for every device and that'll really confuse things
// as MPI rank/count configuration is global in the environment and not per
// device. Hosting frameworks/runtimes can set their own providers that have
// more meaningful representation of multi-device/multi-node.
iree_status_t iree_hal_device_set_default_channel_provider(
    iree_hal_device_t* device) {
  if (!iree_hal_mpi_is_configured()) return iree_ok_status();
  iree_hal_channel_provider_t* channel_provider = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_mpi_channel_provider_create(
          iree_hal_device_host_allocator(device), &channel_provider),
      "creating MPI channel provider as detected in environment");
  iree_hal_device_replace_channel_provider(device, channel_provider);
  iree_hal_channel_provider_release(channel_provider);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Device selection
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, device,
    "Specifies one or more HAL devices to use for execution.\n"
    "Use --list_devices/--dump_devices to see available devices and their\n"
    "canonical URI used with this flag.");

iree_string_view_list_t iree_hal_device_flag_list(void) {
  return FLAG_device_list();
}

iree_status_t iree_hal_create_device_from_flags(
    iree_hal_driver_registry_t* driver_registry,
    iree_string_view_t default_device,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(create_params);
  iree_hal_device_list_t* device_list = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_create_devices_from_flags(
      driver_registry, default_device, create_params, host_allocator,
      &device_list));
  iree_hal_device_t* device = iree_hal_device_list_at(device_list, 0);
  iree_hal_device_retain(device);
  iree_hal_device_list_free(device_list);
  *out_device = device;
  return iree_ok_status();
}

iree_status_t iree_hal_create_devices_from_flags(
    iree_hal_driver_registry_t* driver_registry,
    iree_string_view_t default_device,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_list_t** out_device_list) {
  IREE_ASSERT_ARGUMENT(create_params);
  iree_flag_string_list_t flag_list = FLAG_device_list();
  if (flag_list.count == 0) {
    // No devices specified. Use default if provided.
    if (iree_string_view_is_empty(default_device)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no devices specified; use --list_devices to see the "
          "available devices and specify one or more with --device=");
    }
    flag_list.count = 1;
    flag_list.values = &default_device;
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE({
    for (iree_host_size_t i = 0; i < flag_list.count; ++i) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, flag_list.values[i].data,
                                  flag_list.values[i].size);
    }
  });

  iree_hal_device_list_t* device_list = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_list_allocate(flag_list.count, host_allocator,
                                        &device_list));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < flag_list.count; ++i) {
    // Create the device, which may be slow and dynamically load big
    // dependencies (CUDA, Vulkan, etc).
    iree_hal_device_t* device = NULL;
    status = iree_hal_create_device(driver_registry, flag_list.values[i],
                                    create_params, host_allocator, &device);

    // Optionally wrap the base device allocator with caching/pooling.
    // Doing this here satisfies the requirement that no buffers have been
    // allocated yet - if we returned the device without doing this the caller
    // can more easily break the rules.
    if (iree_status_is_ok(status)) {
      status = iree_hal_configure_allocator_from_flags(device);
    }

    // Optionally set a collective channel provider used by devices to
    // initialize their default channels. Hosting libraries or applications can
    // do the same to interface with their own implementations. Note that this
    // currently sets the same provider for all devices.
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_set_default_channel_provider(device);
    }

    // Add the device to the list to retain it for the caller.
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_list_push_back(device_list, device);
    }
    iree_hal_device_release(device);

    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_device_list = device_list;
  } else {
    iree_hal_device_list_free(device_list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

IREE_FLAG(
    string, device_profiling_mode, "",
    "HAL device profiling mode (one of ['queue', 'dispatch', 'executable', "
    "'trace'])\n"
    "or empty to disable profiling. HAL implementations may require\n"
    "additional flags in order to configure profiling support on their\n"
    "devices.");
IREE_FLAG(
    string, device_profiling_file, "",
    "Optional file path/prefix for profiling file output. Some\n"
    "implementations may require a file name in order to capture profiling\n"
    "information.");
IREE_FLAG(
    string, device_profiling_output, "",
    "Optional path for a raw IREE HAL profiling bundle. The output is written\n"
    "by tooling using a generic profile sink and does not change the\n"
    "implementation-specific meaning of --device_profiling_file.");
IREE_FLAG(
    string, device_profiling_filter_export, "",
    "Optional glob pattern selecting executable export names that should emit\n"
    "heavy profiling artifacts such as dispatch timings or hardware counter\n"
    "samples. Cheap session metadata remains available so filtered captures\n"
    "can still be decoded.");
IREE_FLAG(
    int64_t, device_profiling_filter_command_buffer, -1,
    "Optional nonzero command buffer id selecting operations that should emit\n"
    "heavy profiling artifacts. Use iree-profile command/queue output from an\n"
    "earlier capture to find command buffer ids.");
IREE_FLAG(
    int64_t, device_profiling_filter_command_index, -1,
    "Optional zero-based command-buffer operation index selecting operations\n"
    "that should emit heavy profiling artifacts.");
IREE_FLAG(
    int64_t, device_profiling_filter_physical_device, -1,
    "Optional physical device ordinal selecting operations that should emit\n"
    "heavy profiling artifacts.");
IREE_FLAG(int64_t, device_profiling_filter_queue, -1,
          "Optional queue ordinal selecting operations that should emit heavy\n"
          "profiling artifacts.");
IREE_FLAG_LIST(
    string, device_profiling_counter,
    "Optional implementation-specific hardware counter name to capture. May "
    "be\n"
    "specified multiple times; the selected HAL driver decides which counter\n"
    "names and combinations are supported.");

static iree_status_t iree_hal_profile_capture_filter_set_u32_from_flag(
    int64_t flag_value, iree_hal_profile_capture_filter_flags_t flag,
    const char* flag_name, uint32_t* out_value,
    iree_hal_profile_capture_filter_t* inout_filter) {
  if (flag_value < 0) return iree_ok_status();
  if (flag_value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s value exceeds uint32_t", flag_name);
  }
  *out_value = (uint32_t)flag_value;
  inout_filter->flags |= flag;
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_capture_filter_from_flags(
    iree_hal_profile_capture_filter_t* out_filter) {
  *out_filter = iree_hal_profile_capture_filter_default();
  if (strlen(FLAG_device_profiling_filter_export) != 0) {
    out_filter->flags |=
        IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_EXPORT_PATTERN;
    out_filter->executable_export_pattern =
        iree_make_cstring_view(FLAG_device_profiling_filter_export);
  }
  if (FLAG_device_profiling_filter_command_buffer == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_filter_command_buffer must be nonzero");
  } else if (FLAG_device_profiling_filter_command_buffer > 0) {
    out_filter->flags |= IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID;
    out_filter->command_buffer_id =
        (uint64_t)FLAG_device_profiling_filter_command_buffer;
  }
  IREE_RETURN_IF_ERROR(iree_hal_profile_capture_filter_set_u32_from_flag(
      FLAG_device_profiling_filter_command_index,
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX,
      "device_profiling_filter_command_index", &out_filter->command_index,
      out_filter));
  IREE_RETURN_IF_ERROR(iree_hal_profile_capture_filter_set_u32_from_flag(
      FLAG_device_profiling_filter_physical_device,
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL,
      "device_profiling_filter_physical_device",
      &out_filter->physical_device_ordinal, out_filter));
  IREE_RETURN_IF_ERROR(iree_hal_profile_capture_filter_set_u32_from_flag(
      FLAG_device_profiling_filter_queue,
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL,
      "device_profiling_filter_queue", &out_filter->queue_ordinal, out_filter));
  return iree_ok_status();
}

static iree_status_t iree_hal_profile_sink_create_from_flags(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;

  if (strlen(FLAG_device_profiling_output) == 0) return iree_ok_status();

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_create(
      IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SEQUENTIAL_SCAN |
          IREE_IO_FILE_MODE_SHARE_READ,
      iree_make_cstring_view(FLAG_device_profiling_output),
      /*initial_size=*/0, host_allocator, &file_handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_sink_create(file_handle, host_allocator,
                                               out_sink);
  }
  iree_io_file_handle_release(file_handle);
  return status;
}

iree_status_t iree_hal_begin_profiling_from_flags(iree_hal_device_t* device) {
  if (!device) return iree_ok_status();

  // Today we treat these as exclusive. When we have more implementations we
  // can figure out how best to combine them.
  const iree_flag_string_list_t counter_names =
      FLAG_device_profiling_counter_list();
  iree_hal_device_profiling_options_t options = {0};
  if (strlen(FLAG_device_profiling_mode) == 0) {
    if (counter_names.count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device_profiling_counter requires --device_profiling_mode");
    }
    return iree_ok_status();
  } else if (strcmp(FLAG_device_profiling_mode, "queue") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;
  } else if (strcmp(FLAG_device_profiling_mode, "dispatch") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS;
  } else if (strcmp(FLAG_device_profiling_mode, "executable") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS;
  } else if (strcmp(FLAG_device_profiling_mode, "trace") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_TRACES;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profiling mode '%s'",
                            FLAG_device_profiling_mode);
  }

  // We don't validate the file path as each tool has their own style.
  options.file_path = FLAG_device_profiling_file;
  IREE_RETURN_IF_ERROR(
      iree_hal_profile_capture_filter_from_flags(&options.capture_filter));
  iree_hal_profile_counter_set_selection_t counter_set = {0};
  if (counter_names.count != 0) {
    counter_set.counter_name_count = counter_names.count;
    counter_set.counter_names = counter_names.values;
    options.counter_set_count = 1;
    options.counter_sets = &counter_set;
  }

  iree_hal_profile_sink_t* sink = NULL;
  iree_status_t status =
      iree_hal_profile_sink_create_from_flags(iree_allocator_system(), &sink);
  if (iree_status_is_ok(status)) {
    options.sink = sink;
    status = iree_hal_device_profiling_begin(device, &options);
  }
  iree_hal_profile_sink_release(sink);
  return status;
}

iree_status_t iree_hal_end_profiling_from_flags(iree_hal_device_t* device) {
  if (!device) return iree_ok_status();
  if (strlen(FLAG_device_profiling_mode) == 0) return iree_ok_status();
  return iree_hal_device_profiling_end(device);
}
