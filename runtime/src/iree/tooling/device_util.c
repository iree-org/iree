// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/device_util.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/init.h"
#include "iree/hal/utils/allocators.h"
#include "iree/hal/utils/mpi_channel_provider.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/hal/utils/statistics_sink.h"
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
    iree_status_free(driver_status);
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
    iree_status_free(driver_status);
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
    "HAL device profiling data families as a comma-separated list drawn from\n"
    "['queue-events', 'host-execution', 'device-queue-events',\n"
    "'dispatch-events', 'memory-events', 'device-metrics',\n"
    "'command-region-events', 'counters', 'counter-ranges',\n"
    "'executable-metadata',\n"
    "'executable-traces'] or empty to disable profiling. HAL implementations\n"
    "may require additional flags in order to configure profiling support on\n"
    "their devices. Tooling may force VM-created command buffers to retain\n"
    "metadata for modes that need command/dispatch attribution; leave this\n"
    "empty for production timing runs.");
IREE_FLAG(
    string, device_profiling_output, "",
    "Path for a raw IREE HAL profiling bundle. Required when\n"
    "--device_profiling_mode is nonempty. The output is written by tooling\n"
    "using a generic profile sink.");
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
    "names and combinations are supported. Use "
    "--device_profiling_mode=counters\n"
    "for dispatch-scoped attribution, which may inject packets around "
    "selected\n"
    "dispatches and perturb queue timing. Use\n"
    "--device_profiling_mode=counter-ranges for low-disturbance range "
    "samples.");
IREE_FLAG(
    int64_t, device_profiling_flush_interval_ms, 0,
    "Optional interval in milliseconds for a tooling-owned background thread\n"
    "to call iree_hal_device_profiling_flush while HAL-native profiling is\n"
    "active. 0 disables periodic flushing. The selected HAL backend must\n"
    "support safe in-flight profiling snapshots for the requested data.\n"
    "Sink failures are reported when profiling ends.");
IREE_FLAG(
    bool, print_device_statistics, false,
    "Enables a lightweight HAL profiling session and prints aggregate device\n"
    "statistics at shutdown. This is a command-line convenience path: it does\n"
    "not write a profiling bundle and cannot be combined with\n"
    "--device_profiling_output. When --device_profiling_mode is empty,\n"
    "tooling requests the producer's lightweight execution-statistics mode.");

IREE_FLAG(
    string, device_capture_tool, "",
    "Optional external profiler/tool capture provider such as 'renderdoc' or\n"
    "'metal'. External captures produce provider-specific artifacts and are\n"
    "separate from HAL-native --device_profiling_mode output.");
IREE_FLAG(string, device_capture_file, "",
          "Optional provider-specific external capture output path or path "
          "template.");
IREE_FLAG(string, device_capture_label, "",
          "Optional provider-specific external capture range label.");

struct iree_hal_profiling_from_flags_t {
  // Host allocator used for this session object.
  iree_allocator_t host_allocator;

  // Devices retained while profiling/capture state is active.
  iree_hal_device_t** devices;

  // Number of entries in |devices| and the active-state arrays.
  iree_host_size_t device_count;

  // Per-device true when a nonempty HAL-native profiling session is active.
  bool* native_profile_active;

  // Per-device true when iree_hal_device_external_capture_begin succeeded.
  bool* external_capture_active;

  // Periodic flush interval in milliseconds, or 0 when disabled.
  iree_duration_t flush_interval_ms;

  // Mutex serializing explicit and periodic flush calls.
  iree_slim_mutex_t flush_mutex;

  // Background flush thread owned by the session when periodic flush is active.
  iree_thread_t* flush_thread;

  // In-memory aggregate sink used by --print_device_statistics.
  iree_hal_profile_statistics_sink_t* statistics_sink;

  // Notification used to wake the flush thread during shutdown.
  iree_notification_t flush_notification;

  // Nonzero when the flush thread should stop.
  iree_atomic_int32_t flush_stop_requested;

  // Mutex protecting |flush_status|.
  iree_slim_mutex_t flush_status_mutex;

  // First background flush failure, joined with any later failures.
  iree_status_t flush_status;
};

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

static iree_status_t iree_hal_device_profiling_data_families_from_flags(
    iree_hal_device_profiling_data_families_t* out_data_families) {
  *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  iree_string_view_t remaining =
      iree_string_view_trim(iree_make_cstring_view(FLAG_device_profiling_mode));
  while (remaining.size != 0) {
    iree_string_view_t family_part = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &family_part, &remaining);
    family_part = iree_string_view_trim(family_part);
    if (family_part.size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_mode contains an empty data "
                              "family");
    } else if (iree_string_view_equal(family_part, IREE_SV("queue-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("host-execution"))) {
      *out_data_families |=
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("device-queue-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("dispatch-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("memory-events"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("device-metrics"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("command-region-events"))) {
      *out_data_families |=
          IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS;
    } else if (iree_string_view_equal(family_part, IREE_SV("counters"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
    } else if (iree_string_view_equal(family_part, IREE_SV("counter-ranges")) ||
               iree_string_view_equal(family_part, IREE_SV("pmc-ranges"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("executable-metadata"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
    } else if (iree_string_view_equal(family_part,
                                      IREE_SV("executable-traces"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported profiling data family '%.*s'",
                              (int)family_part.size, family_part.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

iree_status_t
iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
    bool* out_required) {
  IREE_ASSERT_ARGUMENT(out_required);
  *out_required = false;

  iree_hal_device_profiling_data_families_t data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_device_profiling_data_families_from_flags(&data_families));

  if (data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    *out_required = FLAG_print_device_statistics;
    return iree_ok_status();
  }

  const iree_hal_device_profiling_data_families_t command_buffer_data =
      IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
      IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES |
      IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS;
  *out_required = iree_any_bit_set(data_families, command_buffer_data);
  return iree_ok_status();
}

static bool iree_hal_device_profiling_filter_flags_present(void) {
  return strlen(FLAG_device_profiling_filter_export) != 0 ||
         FLAG_device_profiling_filter_command_buffer >= 0 ||
         FLAG_device_profiling_filter_command_index >= 0 ||
         FLAG_device_profiling_filter_physical_device >= 0 ||
         FLAG_device_profiling_filter_queue >= 0;
}

static bool iree_hal_device_external_capture_flags_present(void) {
  return strlen(FLAG_device_capture_tool) != 0 ||
         strlen(FLAG_device_capture_file) != 0 ||
         strlen(FLAG_device_capture_label) != 0;
}

static iree_status_t iree_hal_device_external_capture_begin_from_flags(
    iree_hal_device_t* device) {
  iree_hal_device_external_capture_options_t options = {0};
  options.provider = iree_make_cstring_view(FLAG_device_capture_tool);
  options.file_path = iree_make_cstring_view(FLAG_device_capture_file);
  options.label = iree_make_cstring_view(FLAG_device_capture_label);
  return iree_hal_device_external_capture_begin(device, &options);
}

static bool iree_hal_profiling_from_flags_any_native_profile_active(
    const iree_hal_profiling_from_flags_t* profiling) {
  for (iree_host_size_t i = 0; i < profiling->device_count; ++i) {
    if (profiling->native_profile_active[i]) return true;
  }
  return false;
}

static bool iree_hal_profiling_from_flags_flush_should_stop(void* arg) {
  iree_hal_profiling_from_flags_t* profiling =
      (iree_hal_profiling_from_flags_t*)arg;
  return iree_atomic_load(&profiling->flush_stop_requested,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_profiling_from_flags_record_flush_status(
    iree_hal_profiling_from_flags_t* profiling, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  iree_slim_mutex_lock(&profiling->flush_status_mutex);
  profiling->flush_status = iree_status_join(profiling->flush_status, status);
  iree_slim_mutex_unlock(&profiling->flush_status_mutex);
}

static iree_status_t iree_hal_profiling_from_flags_consume_flush_status(
    iree_hal_profiling_from_flags_t* profiling) {
  iree_slim_mutex_lock(&profiling->flush_status_mutex);
  iree_status_t status = profiling->flush_status;
  profiling->flush_status = iree_ok_status();
  iree_slim_mutex_unlock(&profiling->flush_status_mutex);
  return status;
}

iree_status_t iree_hal_flush_profiling_from_flags(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling ||
      !iree_hal_profiling_from_flags_any_native_profile_active(profiling)) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&profiling->flush_mutex);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < profiling->device_count; ++i) {
    if (profiling->native_profile_active[i]) {
      status = iree_status_join(
          status, iree_hal_device_profiling_flush(profiling->devices[i]));
    }
  }
  iree_slim_mutex_unlock(&profiling->flush_mutex);
  return status;
}

static int iree_hal_profiling_from_flags_flush_thread_main(void* arg) {
  iree_hal_profiling_from_flags_t* profiling =
      (iree_hal_profiling_from_flags_t*)arg;

  while (!iree_hal_profiling_from_flags_flush_should_stop(profiling)) {
    const bool should_stop = iree_notification_await(
        &profiling->flush_notification,
        iree_hal_profiling_from_flags_flush_should_stop, profiling,
        iree_make_timeout_ms(profiling->flush_interval_ms));
    if (should_stop) break;

    iree_status_t status = iree_hal_flush_profiling_from_flags(profiling);
    if (!iree_status_is_ok(status)) {
      iree_hal_profiling_from_flags_record_flush_status(profiling, status);
      break;
    }
  }

  return 0;
}

static iree_status_t iree_hal_profiling_from_flags_start_periodic_flush(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!iree_hal_profiling_from_flags_any_native_profile_active(profiling) ||
      profiling->flush_interval_ms == 0) {
    return iree_ok_status();
  }

  iree_notification_initialize(&profiling->flush_notification);
  iree_slim_mutex_initialize(&profiling->flush_status_mutex);
  iree_atomic_store(&profiling->flush_stop_requested, 0,
                    iree_memory_order_relaxed);

  const iree_thread_create_params_t thread_params = {
      .name = IREE_SV("iree-hal-profile-flush"),
      .priority_class = IREE_THREAD_PRIORITY_CLASS_LOW,
  };
  iree_status_t status = iree_thread_create(
      iree_hal_profiling_from_flags_flush_thread_main, profiling, thread_params,
      profiling->host_allocator, &profiling->flush_thread);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_deinitialize(&profiling->flush_status_mutex);
    iree_notification_deinitialize(&profiling->flush_notification);
  }
  return status;
}

static iree_status_t iree_hal_profiling_from_flags_stop_periodic_flush(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling || !profiling->flush_thread) return iree_ok_status();

  iree_atomic_store(&profiling->flush_stop_requested, 1,
                    iree_memory_order_release);
  iree_notification_post(&profiling->flush_notification, IREE_ALL_WAITERS);
  // Final release joins the thread. Calling iree_thread_join before release
  // would double-join pthread-backed threads, which is undefined behavior.
  iree_thread_release(profiling->flush_thread);
  profiling->flush_thread = NULL;

  iree_status_t status =
      iree_hal_profiling_from_flags_consume_flush_status(profiling);
  iree_slim_mutex_deinitialize(&profiling->flush_status_mutex);
  iree_notification_deinitialize(&profiling->flush_notification);
  return status;
}

static void iree_hal_profiling_from_flags_release_devices(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling) return;
  for (iree_host_size_t i = 0; i < profiling->device_count; ++i) {
    iree_hal_device_release(profiling->devices[i]);
  }
  iree_allocator_free(profiling->host_allocator,
                      profiling->external_capture_active);
  iree_allocator_free(profiling->host_allocator,
                      profiling->native_profile_active);
  iree_allocator_free(profiling->host_allocator, profiling->devices);
}

static iree_status_t iree_hal_begin_device_list_profiling_from_flags(
    iree_host_size_t device_count, iree_hal_device_t* const* devices,
    iree_allocator_t host_allocator,
    iree_hal_profiling_from_flags_t** out_profiling) {
  IREE_ASSERT_ARGUMENT(out_profiling);
  *out_profiling = NULL;
  if (device_count == 0) return iree_ok_status();

  const iree_flag_string_list_t counter_names =
      FLAG_device_profiling_counter_list();
  iree_hal_device_profiling_options_t options = {0};
  IREE_RETURN_IF_ERROR(iree_hal_device_profiling_data_families_from_flags(
      &options.data_families));
  const bool external_capture_requested = strlen(FLAG_device_capture_tool) != 0;
  const bool statistics_requested = FLAG_print_device_statistics;
  if (!external_capture_requested &&
      iree_hal_device_external_capture_flags_present()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device_capture_file and --device_capture_label "
                            "require --device_capture_tool");
  }
  if (statistics_requested && strlen(FLAG_device_profiling_output) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--print_device_statistics cannot be combined with "
                            "--device_profiling_output");
  }
  if (FLAG_device_profiling_flush_interval_ms < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_flush_interval_ms must be non-negative");
  }
  if (options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    if (counter_names.count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_counter requires "
                              "--device_profiling_mode=counters or "
                              "--device_profiling_mode=counter-ranges");
    }
    if (strlen(FLAG_device_profiling_output) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device_profiling_output requires --device_profiling_mode");
    }
    if (iree_hal_device_profiling_filter_flags_present() &&
        !statistics_requested) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device_profiling_filter_* requires --device_profiling_mode or "
          "--print_device_statistics");
    }
    if (FLAG_device_profiling_flush_interval_ms != 0 && !statistics_requested) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--device_profiling_flush_interval_ms requires "
                              "--device_profiling_mode or "
                              "--print_device_statistics");
    }
    if (statistics_requested) {
      options.flags |= IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
    } else if (!external_capture_requested) {
      return iree_ok_status();
    }
  }
  if (counter_names.count != 0 &&
      !iree_any_bit_set(options.data_families,
                        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                            IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device_profiling_counter requires --device_profiling_mode=counters "
        "or --device_profiling_mode=counter-ranges");
  }
  if (options.data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      strlen(FLAG_device_profiling_output) == 0 && !statistics_requested) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--device_profiling_mode requires "
                            "--device_profiling_output");
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_profile_capture_filter_from_flags(&options.capture_filter));
  iree_hal_profile_counter_set_selection_t counter_set = {0};
  if (counter_names.count != 0) {
    counter_set.counter_name_count = counter_names.count;
    counter_set.counter_names = counter_names.values;
    options.counter_set_count = 1;
    options.counter_sets = &counter_set;
  }

  iree_hal_profiling_from_flags_t* profiling = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*profiling),
                                             (void**)&profiling));
  memset(profiling, 0, sizeof(*profiling));
  profiling->host_allocator = host_allocator;
  profiling->flush_interval_ms = FLAG_device_profiling_flush_interval_ms;
  iree_slim_mutex_initialize(&profiling->flush_mutex);

  iree_host_size_t devices_size = 0;
  iree_host_size_t active_size = 0;
  bool allocation_sizes_valid =
      iree_host_size_checked_mul(device_count, sizeof(*profiling->devices),
                                 &devices_size) &&
      iree_host_size_checked_mul(device_count,
                                 sizeof(*profiling->native_profile_active),
                                 &active_size);
  if (IREE_UNLIKELY(!allocation_sizes_valid)) {
    iree_slim_mutex_deinitialize(&profiling->flush_mutex);
    iree_allocator_free(host_allocator, profiling);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profiling device list is too large");
  }

  iree_status_t status = iree_allocator_malloc(host_allocator, devices_size,
                                               (void**)&profiling->devices);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, active_size,
                                   (void**)&profiling->native_profile_active);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, active_size,
                                   (void**)&profiling->external_capture_active);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_profiling_from_flags_release_devices(profiling);
    iree_slim_mutex_deinitialize(&profiling->flush_mutex);
    iree_allocator_free(host_allocator, profiling);
    return status;
  }
  memset(profiling->devices, 0, devices_size);
  memset(profiling->native_profile_active, 0, active_size);
  memset(profiling->external_capture_active, 0, active_size);
  profiling->device_count = device_count;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    if (!devices[i]) {
      iree_hal_profiling_from_flags_release_devices(profiling);
      iree_slim_mutex_deinitialize(&profiling->flush_mutex);
      iree_allocator_free(host_allocator, profiling);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "profiling device list contains a null device");
    }
    profiling->devices[i] = devices[i];
    iree_hal_device_retain(profiling->devices[i]);
  }

  iree_hal_profile_sink_t* sink = NULL;
  iree_hal_profile_statistics_sink_t* statistics_sink = NULL;
  if (statistics_requested) {
    status = iree_hal_profile_statistics_sink_create(host_allocator,
                                                     &statistics_sink);
    if (iree_status_is_ok(status)) {
      sink = iree_hal_profile_statistics_sink_base(statistics_sink);
    }
  } else {
    status = iree_hal_profile_sink_create_from_flags(host_allocator, &sink);
  }
  if (iree_status_is_ok(status)) {
    options.sink = sink;
    for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_device_profiling_begin(profiling->devices[i], &options);
      profiling->native_profile_active[i] =
          iree_status_is_ok(status) &&
          (options.data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
           iree_hal_device_profiling_options_requests_lightweight_statistics(
               &options));
    }
  }
  if (iree_status_is_ok(status) && external_capture_requested) {
    for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
         ++i) {
      status = iree_hal_device_external_capture_begin_from_flags(
          profiling->devices[i]);
      profiling->external_capture_active[i] = iree_status_is_ok(status);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_profiling_from_flags_start_periodic_flush(profiling);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_profiling_from_flags_stop_periodic_flush(profiling));
    for (iree_host_size_t i = profiling->device_count; i > 0; --i) {
      if (profiling->external_capture_active[i - 1]) {
        status = iree_status_join(status, iree_hal_device_external_capture_end(
                                              profiling->devices[i - 1]));
      }
    }
    for (iree_host_size_t i = profiling->device_count; i > 0; --i) {
      if (profiling->native_profile_active[i - 1]) {
        status = iree_status_join(
            status, iree_hal_device_profiling_end(profiling->devices[i - 1]));
      }
    }
    iree_hal_profiling_from_flags_release_devices(profiling);
    iree_slim_mutex_deinitialize(&profiling->flush_mutex);
    iree_allocator_free(host_allocator, profiling);
  } else {
    profiling->statistics_sink = statistics_sink;
    statistics_sink = NULL;
    *out_profiling = profiling;
  }
  if (statistics_sink) {
    iree_hal_profile_statistics_sink_release(statistics_sink);
  } else if (!statistics_requested) {
    iree_hal_profile_sink_release(sink);
  }
  return status;
}

iree_status_t iree_hal_begin_profiling_from_flags(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    iree_hal_profiling_from_flags_t** out_profiling) {
  IREE_ASSERT_ARGUMENT(out_profiling);
  *out_profiling = NULL;
  if (!device) return iree_ok_status();
  return iree_hal_begin_device_list_profiling_from_flags(
      1, &device, host_allocator, out_profiling);
}

iree_status_t iree_hal_begin_device_group_profiling_from_flags(
    iree_hal_device_group_t* device_group, iree_allocator_t host_allocator,
    iree_hal_profiling_from_flags_t** out_profiling) {
  IREE_ASSERT_ARGUMENT(out_profiling);
  *out_profiling = NULL;
  if (!device_group) return iree_ok_status();
  iree_host_size_t device_count =
      iree_hal_device_group_device_count(device_group);
  if (device_count == 0) return iree_ok_status();

  iree_hal_device_t** devices = NULL;
  iree_host_size_t devices_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(device_count, sizeof(*devices),
                                                &devices_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profiling device group is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, devices_size, (void**)&devices));
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    devices[i] = iree_hal_device_group_device_at(device_group, i);
  }
  iree_status_t status = iree_hal_begin_device_list_profiling_from_flags(
      device_count, devices, host_allocator, out_profiling);
  iree_allocator_free(host_allocator, devices);
  return status;
}

iree_status_t iree_hal_end_profiling_from_flags(
    iree_hal_profiling_from_flags_t* profiling) {
  if (!profiling) return iree_ok_status();

  iree_status_t status =
      iree_hal_profiling_from_flags_stop_periodic_flush(profiling);
  for (iree_host_size_t i = profiling->device_count; i > 0; --i) {
    if (profiling->external_capture_active[i - 1]) {
      status = iree_status_join(status, iree_hal_device_external_capture_end(
                                            profiling->devices[i - 1]));
    }
  }
  for (iree_host_size_t i = profiling->device_count; i > 0; --i) {
    if (profiling->native_profile_active[i - 1]) {
      status = iree_status_join(
          status, iree_hal_device_profiling_end(profiling->devices[i - 1]));
    }
  }
  if (profiling->statistics_sink) {
    status = iree_status_join(status, iree_hal_profile_statistics_sink_fprint(
                                          stderr, profiling->statistics_sink));
    iree_hal_profile_statistics_sink_release(profiling->statistics_sink);
  }
  iree_hal_profiling_from_flags_release_devices(profiling);
  iree_slim_mutex_deinitialize(&profiling->flush_mutex);
  iree_allocator_free(profiling->host_allocator, profiling);
  return status;
}
