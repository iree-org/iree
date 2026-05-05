// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/driver.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/logical_device.h"
#include "iree/hal/drivers/vulkan/physical_device.h"
#include "iree/hal/drivers/vulkan/syms.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_driver_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_vulkan_driver_options_initialize(
    iree_hal_vulkan_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  iree_hal_vulkan_device_options_initialize(&out_options->device_options);
}

static iree_status_t iree_hal_vulkan_driver_options_verify(
    const iree_hal_vulkan_driver_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (options->libvulkan_search_paths.count &&
      !options->libvulkan_search_paths.values) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "Vulkan libvulkan search path list has count %" PRIhsz
                         " but no value storage",
                         options->libvulkan_search_paths.count);
  }
  for (iree_host_size_t i = 0;
       i < options->libvulkan_search_paths.count && iree_status_is_ok(status);
       ++i) {
    const iree_string_view_t search_path =
        options->libvulkan_search_paths.values[i];
    if (search_path.size && !search_path.data) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Vulkan libvulkan search path %" PRIhsz
                                " has a nonzero length and no storage",
                                i);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_driver_t {
  // Base HAL driver resource.
  iree_hal_resource_t resource;
  // Allocator used for driver-owned host allocations.
  iree_allocator_t host_allocator;
  // Canonical driver identifier.
  iree_string_view_t identifier;

  // Driver options with retained string views pointing into trailing storage.
  iree_hal_vulkan_driver_options_t options;

  // Retained Vulkan loader used for enumeration and device creation.
  iree_hal_vulkan_libvulkan_t libvulkan;

  // + trailing libvulkan_search_paths table, identifier, and search path
  // strings.
} iree_hal_vulkan_driver_t;

static const iree_hal_driver_vtable_t iree_hal_vulkan_driver_vtable;

static iree_hal_vulkan_driver_t* iree_hal_vulkan_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_driver_vtable);
  return (iree_hal_vulkan_driver_t*)base_value;
}

static iree_status_t iree_hal_vulkan_driver_initialize_libvulkan(
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* syms, iree_allocator_t host_allocator,
    iree_hal_vulkan_libvulkan_t* out_libvulkan) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_libvulkan);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (syms) {
    status = iree_hal_vulkan_libvulkan_copy(&syms->libvulkan, out_libvulkan);
  } else {
    status = iree_hal_vulkan_libvulkan_initialize(
        IREE_HAL_VULKAN_LIBVULKAN_FLAG_NONE, options->libvulkan_search_paths,
        host_allocator, out_libvulkan);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_driver_create(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* syms, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_driver_options_verify(options));

  iree_host_size_t search_path_storage_size = 0;
  for (iree_host_size_t i = 0; i < options->libvulkan_search_paths.count; ++i) {
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            search_path_storage_size,
            options->libvulkan_search_paths.values[i].size,
            &search_path_storage_size))) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "Vulkan libvulkan search path storage overflow"));
    }
  }

  iree_host_size_t search_paths_offset = 0;
  iree_host_size_t identifier_offset = 0;
  iree_host_size_t search_path_storage_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              iree_sizeof_struct(iree_hal_vulkan_driver_t), &total_size,
              IREE_STRUCT_FIELD(options->libvulkan_search_paths.count,
                                iree_string_view_t, &search_paths_offset),
              IREE_STRUCT_FIELD(identifier.size, char, &identifier_offset),
              IREE_STRUCT_FIELD(search_path_storage_size, char,
                                &search_path_storage_offset)));

  iree_hal_vulkan_driver_t* driver = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  memset(driver, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_vulkan_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(identifier, &driver->identifier,
                                    (char*)driver + identifier_offset);
  memcpy(&driver->options, options, sizeof(*options));
  if (options->libvulkan_search_paths.count) {
    iree_string_view_t* search_paths =
        (iree_string_view_t*)((uint8_t*)driver + search_paths_offset);
    char* search_path_storage = (char*)driver + search_path_storage_offset;
    for (iree_host_size_t i = 0; i < options->libvulkan_search_paths.count;
         ++i) {
      const iree_string_view_t source_path =
          options->libvulkan_search_paths.values[i];
      iree_string_view_append_to_buffer(source_path, &search_paths[i],
                                        search_path_storage);
      search_path_storage += source_path.size;
    }
    driver->options.libvulkan_search_paths = (iree_string_view_list_t){
        .count = options->libvulkan_search_paths.count, .values = search_paths};
  } else {
    driver->options.libvulkan_search_paths = iree_string_view_list_empty();
  }

  iree_status_t status = iree_hal_vulkan_driver_initialize_libvulkan(
      &driver->options, syms, host_allocator, &driver->libvulkan);

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_driver_create_using_instance(
    iree_string_view_t identifier,
    const iree_hal_vulkan_driver_options_t* options,
    iree_hal_vulkan_syms_t* instance_syms, VkInstance instance,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(instance_syms);
  IREE_ASSERT_ARGUMENT(instance);
  (void)instance_syms;
  (void)instance;
  return iree_hal_vulkan_driver_create(identifier, options, instance_syms,
                                       host_allocator, out_driver);
}

static void iree_hal_vulkan_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_vulkan_driver_t* driver = iree_hal_vulkan_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_libvulkan_deinitialize(&driver->libvulkan);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  iree_hal_vulkan_driver_t* driver = iree_hal_vulkan_driver_cast(base_driver);
  return iree_hal_vulkan_query_available_physical_devices(
      &driver->libvulkan, &driver->options, host_allocator,
      out_device_info_count, out_device_infos);
}

static iree_status_t iree_hal_vulkan_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_vulkan_driver_t* driver = iree_hal_vulkan_driver_cast(base_driver);
  return iree_hal_vulkan_dump_physical_device_info(
      &driver->libvulkan, &driver->options, device_id, driver->host_allocator,
      builder);
}

static iree_status_t iree_hal_vulkan_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  iree_hal_vulkan_driver_t* driver = iree_hal_vulkan_driver_cast(base_driver);
  *out_device = NULL;
  return iree_hal_vulkan_logical_device_create_by_id(
      driver->identifier, &driver->options, &driver->libvulkan, device_id,
      param_count, params, create_params, host_allocator, out_device);
}

static iree_status_t iree_hal_vulkan_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  iree_hal_vulkan_driver_t* driver = iree_hal_vulkan_driver_cast(base_driver);
  *out_device = NULL;

  (void)driver_name;
  return iree_hal_vulkan_logical_device_create_by_path(
      driver->identifier, &driver->options, &driver->libvulkan, device_path,
      param_count, params, create_params, host_allocator, out_device);
}

static const iree_hal_driver_vtable_t iree_hal_vulkan_driver_vtable = {
    .destroy = iree_hal_vulkan_driver_destroy,
    .query_available_devices = iree_hal_vulkan_driver_query_available_devices,
    .dump_device_info = iree_hal_vulkan_driver_dump_device_info,
    .create_device_by_id = iree_hal_vulkan_driver_create_device_by_id,
    .create_device_by_path = iree_hal_vulkan_driver_create_device_by_path,
};
