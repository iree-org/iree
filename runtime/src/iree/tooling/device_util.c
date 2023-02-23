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
#include "iree/hal/utils/caching_allocator.h"
#include "iree/hal/utils/debug_allocator.h"

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
    "  Show all devices from a particular driver: --dump_devices=vulkan\n");

//===----------------------------------------------------------------------===//
// Allocator configuration
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, device_allocator,
    "Specifies one or more HAL device allocator specs to augment the base\n"
    "device allocator. See each allocator type for supported configurations.");

// Selects a heap from |heaps| matching the given |heap_key|.
// Fails if no heap matches the given key. Optionally a buffer usage bitfield
// can be provided. Wildcards can be used with either to match the first heap
// that meets either requirement.
//
// Examples:
//   *: first heap
//   *;transfer: first heap with transfer usage
//   device_local: first heap with the IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL bit
//   device_local|host_visible
//   device_local;transfer|dispatch_storage
static iree_status_t iree_hal_select_heap(
    iree_string_view_t heap_key, iree_host_size_t heap_count,
    const iree_hal_allocator_memory_heap_t* heaps,
    const iree_hal_allocator_memory_heap_t** out_heap) {
  iree_string_view_t memory_type_str = iree_string_view_empty();
  iree_string_view_t buffer_usage_str = iree_string_view_empty();
  iree_string_view_split(heap_key, ';', &memory_type_str, &buffer_usage_str);

  // Parse the provided filters, if any.
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (!iree_string_view_is_empty(memory_type_str) &&
      !iree_string_view_equal(memory_type_str, IREE_SV("*"))) {
    IREE_RETURN_IF_ERROR(
        iree_hal_memory_type_parse(memory_type_str, &memory_type));
  }
  if (!iree_string_view_is_empty(buffer_usage_str) &&
      !iree_string_view_equal(buffer_usage_str, IREE_SV("*"))) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_usage_parse(buffer_usage_str, &buffer_usage));
  }

  // Return the first heap satisfying all filters.
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    if ((!memory_type || iree_all_bits_set(heaps[i].type, memory_type)) &&
        (!buffer_usage ||
         iree_all_bits_set(heaps[i].allowed_usage, buffer_usage))) {
      *out_heap = &heaps[i];
      return iree_ok_status();
    }
  }

  // No matching heap found; can happen if the device doesn't have the kind of
  // heaps the user was expecting with the configuration.
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "no heap matching requested config params "
                          "memory_type='%.*s', buffer_usage='%.*s'",
                          (int)memory_type_str.size, memory_type_str.data,
                          (int)buffer_usage_str.size, buffer_usage_str.data);
}

// Parses a byte size in |size_str| and returns the value in |out_size|.
//
// Examples:
//   1073741824 => 1073741824
//          1gb => 1000000000
//         1gib => 1073741824
static iree_status_t iree_hal_parse_device_size(iree_string_view_t size_str,
                                                iree_device_size_t* out_size) {
  // TODO(benvanik): probably worth to-lowering here on the size. Having copies
  // of all the string view utils for just this case is code size overkill. For
  // now only accept lazy lowercase.
  iree_device_size_t scale = 1;
  if (iree_string_view_consume_suffix(&size_str, IREE_SV("kb"))) {
    scale = 1000;
  } else if (iree_string_view_consume_suffix(&size_str, IREE_SV("kib"))) {
    scale = 1024;
  } else if (iree_string_view_consume_suffix(&size_str, IREE_SV("mb"))) {
    scale = 1000 * 1000;
  } else if (iree_string_view_consume_suffix(&size_str, IREE_SV("mib"))) {
    scale = 1024 * 1024;
  } else if (iree_string_view_consume_suffix(&size_str, IREE_SV("gb"))) {
    scale = 1000 * 1000 * 1000;
  } else if (iree_string_view_consume_suffix(&size_str, IREE_SV("gib"))) {
    scale = 1024 * 1024 * 1024;
  }
  uint64_t size = 0;
  if (!iree_string_view_atoi_uint64(size_str, &size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "size must be an integer, got '%.*s'",
                            (int)size_str.size, size_str.data);
  }
  *out_size = size * scale;
  return iree_ok_status();
}

// Configures a new caching allocator with the given key-value |config_pairs|.
// When no |config_pairs| are provided the caching allocator will be created as
// unbounded, retaining all allocations of all sizes in all heaps. If pairs are
// provided then each specifies a pool in the allocator that maps to a heap
// based on the heap key as parsed by iree_hal_select_heap. Multiple pools may
// share the same heap but with different limits, for example allowing at most
// one device local allocation greater than 100MB to be retained while 10 less
// than 100MB can be retained. Wildcards can be used to indicate max values or
// defaults.
//
// Expected form:
//   heap_key=max_allocation_size;max_allocation_capacity;max_free_allocation_count
// Example:
//   device_local=1gib;1gib;8
//   host_local=*;*;32
static iree_status_t iree_hal_configure_caching_allocator(
    iree_string_view_t config_pairs, iree_hal_device_t* device,
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_t** out_wrapped_allocator) {
  if (iree_string_view_is_empty(config_pairs)) {
    // No parameters implies unbounded; we'll hang on to all memory forever.
    // This is only useful in very specific usage patterns such as statically
    // shaped and deterministic benchmarks that always allocate the same amounts
    // of memory per invocation.
    return iree_hal_caching_allocator_create_unbounded(
        base_allocator, iree_hal_allocator_host_allocator(base_allocator),
        out_wrapped_allocator);
  }

  // Query all heaps from the base allocator. We'll use this list to match the
  // user-provided pool parameters to heaps. It's likely that not all heaps
  // will be selected by the user.
  iree_host_size_t heap_count = 0;
  iree_hal_allocator_memory_heap_t heaps[16];
  IREE_RETURN_IF_ERROR(iree_hal_allocator_query_memory_heaps(
      base_allocator, IREE_ARRAYSIZE(heaps), heaps, &heap_count));

  // Build a list of pools based on user specification.
  iree_host_size_t pool_count = 0;
  iree_hal_caching_allocator_pool_params_t pool_params_storage[16];
  do {
    if (pool_count + 1 > IREE_ARRAYSIZE(pool_params_storage)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many pools specified");
    }

    // Pop the key=value config pair from the list.
    iree_string_view_t config_pair = iree_string_view_empty();
    iree_string_view_split(config_pairs, ',', &config_pair, &config_pairs);
    iree_string_view_t heap_key = iree_string_view_empty();
    iree_string_view_t pool_config = iree_string_view_empty();
    iree_string_view_split(config_pair, '=', &heap_key, &pool_config);
    heap_key = iree_string_view_trim(heap_key);
    if (iree_string_view_is_empty(heap_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "heap key must specified in pool params");
    }

    // Select the heap based on the key.
    const iree_hal_allocator_memory_heap_t* heap = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_select_heap(heap_key, heap_count, heaps, &heap));

    // Configure the pool based on the provided parameters.
    iree_hal_caching_allocator_pool_params_t* pool_params =
        &pool_params_storage[pool_count++];
    iree_hal_caching_allocator_pool_params_initialize(*heap, pool_params);

    iree_string_view_t max_allocation_size_str = iree_string_view_empty();
    iree_string_view_t max_allocation_capacity_str = iree_string_view_empty();
    iree_string_view_t max_free_allocation_count_str = iree_string_view_empty();
    iree_string_view_split(pool_config, ';', &max_allocation_size_str,
                           &pool_config);
    iree_string_view_split(pool_config, ';', &max_allocation_capacity_str,
                           &pool_config);
    iree_string_view_split(pool_config, ';', &max_free_allocation_count_str,
                           &pool_config);
    max_allocation_size_str = iree_string_view_trim(max_allocation_size_str);
    if (!iree_string_view_is_empty(max_allocation_size_str) &&
        !iree_string_view_equal(max_allocation_size_str, IREE_SV("*"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_parse_device_size(max_allocation_size_str,
                                     &pool_params->max_allocation_size),
          "parsing max_allocation_size");
    }
    max_allocation_capacity_str =
        iree_string_view_trim(max_allocation_capacity_str);
    if (!iree_string_view_is_empty(max_allocation_capacity_str) &&
        !iree_string_view_equal(max_allocation_capacity_str, IREE_SV("*"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_parse_device_size(max_allocation_capacity_str,
                                     &pool_params->max_allocation_capacity),
          "parsing max_allocation_capacity");
    }
    max_free_allocation_count_str =
        iree_string_view_trim(max_free_allocation_count_str);
    if (!iree_string_view_is_empty(max_free_allocation_count_str) &&
        !iree_string_view_equal(max_free_allocation_count_str, IREE_SV("*"))) {
      uint32_t max_free_allocation_count = 0;
      if (!iree_string_view_atoi_uint32(max_free_allocation_count_str,
                                        &max_free_allocation_count)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid count '%.*s'",
                                (int)max_free_allocation_count_str.size,
                                max_free_allocation_count_str.data);
      }
      pool_params->max_free_allocation_count = max_free_allocation_count;
    }
  } while (!iree_string_view_is_empty(config_pairs));
  return iree_hal_caching_allocator_create_with_pools(
      pool_count, pool_params_storage, base_allocator,
      iree_hal_allocator_host_allocator(base_allocator), out_wrapped_allocator);
}

// Parses a single flag and wraps |base_allocator|.
// Flag values are specifications and may include configuration values.
// Examples:
//   some_allocator
//   some_allocator:key=value
//   some_allocator:key=value,key=value
static iree_status_t iree_hal_configure_allocator_from_spec(
    iree_string_view_t spec, iree_hal_device_t* device,
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_t** out_wrapped_allocator) {
  iree_string_view_t allocator_name = iree_string_view_empty();
  iree_string_view_t config_pairs = iree_string_view_empty();
  iree_string_view_split(spec, ':', &allocator_name, &config_pairs);
  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(allocator_name, IREE_SV("caching"))) {
    status = iree_hal_configure_caching_allocator(
        config_pairs, device, base_allocator, out_wrapped_allocator);
  } else if (iree_string_view_equal(allocator_name, IREE_SV("debug"))) {
    status = iree_hal_debug_allocator_create(
        device, base_allocator,
        iree_hal_allocator_host_allocator(base_allocator),
        out_wrapped_allocator);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized allocator '%.*s'",
                            (int)allocator_name.size, allocator_name.data);
  }
  if (iree_status_is_ok(status)) {
    // New wrapping allocator has taken ownership of the base allocator.
    iree_hal_allocator_release(base_allocator);
  }
  return status;
}

// Configures the |device| allocator based on the --device_allocator= flag.
// This will wrap the underlying device allocator in zero or more configurable
// allocator shims.
//
// WARNING: not thread-safe and must only be called immediately after device
// creation.
static iree_status_t iree_hal_configure_allocator_from_flags(
    iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_flag_string_list_t list = FLAG_device_allocator_list();

  // The current device allocator should be the base one registered or created
  // with the device. If no allocator flags were provided this may be no-op and
  // we'll just pass it right back in.
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  iree_hal_allocator_retain(device_allocator);

  // Walk the specs provided and wrap in order from base to last specified.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    status = iree_hal_configure_allocator_from_spec(
        list.values[i], device, device_allocator, &device_allocator);
    if (!iree_status_is_ok(status)) break;
  }

  // Swap the allocator on the device - this is only safe because we know no
  // allocations have been made yet.
  if (iree_status_is_ok(status)) {
    iree_hal_device_replace_allocator(device, device_allocator);
  }
  iree_hal_allocator_release(device_allocator);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Device selection
//===----------------------------------------------------------------------===//

IREE_FLAG_LIST(
    string, device,
    "Specifies one or more HAL devices to use for execution.\n"
    "Use --list_devices/--dump_devices to see available devices and their\n"
    "canonical URI used with this flag.");

// TODO(#5724): remove this and replace with an iree_hal_device_set_t.
void iree_hal_get_devices_flag_list(iree_host_size_t* out_count,
                                    const iree_string_view_t** out_list) {
  *out_count = FLAG_device_list().count;
  *out_list = FLAG_device_list().values;
}

iree_status_t iree_hal_create_device_from_flags(
    iree_string_view_t default_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_string_view_t device_uri = default_device;
  const iree_flag_string_list_t list = FLAG_device_list();
  if (list.count == 0) {
    // No devices specified. Use default if provided.
    if (iree_string_view_is_empty(default_device)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "no device specified; use --list_devices to see the "
          "available devices and specify one with --device=");
    }
  } else if (list.count > 1) {
    // Too many devices for the single device creation function.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "too many devices specified; only one --device= "
                            "flag may be provided with this API");
  } else {
    // Exactly one device specified.
    device_uri = list.values[0];
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Create the device, which may be slow and dynamically load big dependencies
  // (CUDA, Vulkan, etc).
  iree_hal_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_create_device(iree_hal_available_driver_registry(),
                                 device_uri, host_allocator, &device));

  // Optionally wrap the base device allocator with caching/pooling.
  // Doing this here satisfies the requirement that no buffers have been
  // allocated yet - if we returned the device without doing this the caller can
  // more easily break the rules.
  iree_status_t status = iree_hal_configure_allocator_from_flags(device);

  if (iree_status_is_ok(status)) {
    *out_device = device;
  } else {
    iree_hal_device_release(device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

IREE_FLAG(
    string, device_profiling_mode, "",
    "HAL device profiling mode (one of ['queue', 'dispatch', 'executable']) "
    "or empty to disable profiling. HAL implementations may require additional "
    "flags in order to configure profiling support on "
    "their devices.");
IREE_FLAG(
    string, device_profiling_file, "",
    "Optional file path/prefix for profiling file output. Some implementations "
    "may require a file name in order to capture profiling information.");

iree_status_t iree_hal_begin_profiling_from_flags(iree_hal_device_t* device) {
  if (!device) return iree_ok_status();

  // Today we treat these as exclusive. When we have more implementations we
  // can figure out how best to combine them.
  iree_hal_device_profiling_options_t options = {0};
  if (strlen(FLAG_device_profiling_mode) == 0) {
    return iree_ok_status();
  } else if (strcmp(FLAG_device_profiling_mode, "queue") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_QUEUE_OPERATIONS;
  } else if (strcmp(FLAG_device_profiling_mode, "dispatch") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_DISPATCH_COUNTERS;
  } else if (strcmp(FLAG_device_profiling_mode, "executable") == 0) {
    options.mode |= IREE_HAL_DEVICE_PROFILING_MODE_EXECUTABLE_COUNTERS;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profiling mode '%s'",
                            FLAG_device_profiling_mode);
  }

  // We don't validate the file path as each tool has their own style.
  options.file_path = FLAG_device_profiling_file;

  return iree_hal_device_profiling_begin(device, &options);
}

iree_status_t iree_hal_end_profiling_from_flags(iree_hal_device_t* device) {
  if (!device) return iree_ok_status();
  if (strlen(FLAG_device_profiling_mode) == 0) return iree_ok_status();
  return iree_hal_device_profiling_end(device);
}
