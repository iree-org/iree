// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/driver.h"

#include "iree/hal/drivers/amdgpu/api.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_driver_options_t
//===----------------------------------------------------------------------===//

// Indicates that all visible devices should be used.
#define IREE_HAL_AMDGPU_DEVICE_ID_DEFAULT 0

IREE_API_EXPORT void iree_hal_amdgpu_driver_options_initialize(
    iree_hal_amdgpu_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));

  // TODO(benvanik): set defaults based on compiler configuration. Flags should
  // not be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.

  iree_hal_amdgpu_logical_device_options_initialize(
      &out_options->default_device_options);
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_options_parse(
    iree_hal_amdgpu_driver_options_t* options, iree_string_pair_list_t params) {
  IREE_ASSERT_ARGUMENT(options);
  if (!params.count) return iree_ok_status();  // no-op
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): parameters.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_options_verify(
    const iree_hal_amdgpu_driver_options_t* options) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): verify that the parameters are within expected ranges and
  // any requested features are supported.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_device_info_t
//===----------------------------------------------------------------------===//

// Captures a reference to the string from the prior |intern_offset| to the new
// string builder tail and updates the |intern_offset| to the tail.
//
// Usage:
//  iree_host_size_t intern_offset = iree_string_builder_size(builder);
//  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "foo"));
//  iree_string_builder_intern(builder, &intern_offset, &foo_view);
//  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "bar"));
//  iree_string_builder_intern(builder, &intern_offset, &bar_view);
static void iree_string_builder_intern(iree_string_builder_t* builder,
                                       iree_host_size_t* intern_offset,
                                       iree_string_view_t* out_view) {
  const iree_host_size_t old_offset = *intern_offset;
  const iree_host_size_t new_offset = iree_string_builder_size(builder);
  const iree_host_size_t length = new_offset - old_offset;
  *intern_offset = new_offset;
  if (out_view != NULL) {
    *out_view = iree_make_string_view(
        iree_string_builder_buffer(builder) + old_offset, length);
  }
}

static iree_status_t iree_hal_amdgpu_driver_append_pseudo_device_infos(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_host_size_t* device_info_count, iree_hal_device_info_t* device_infos,
    iree_string_builder_t* builder) {
  iree_host_size_t intern_offset = iree_string_builder_size(builder);

  // Add `` default device (all visible devices).
  iree_hal_device_info_t* default_info =
      device_infos ? &device_infos[*device_info_count] : NULL;
  if (default_info) {
    default_info->device_id = IREE_HAL_AMDGPU_DEVICE_ID_DEFAULT;
    default_info->path = iree_string_view_empty();
  }
  if (topology->gpu_agent_count == 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "ROCR_VISIBLE_DEVICES: 1 Visible GPU Agent (Node "));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "ROCR_VISIBLE_DEVICES: %" PRIhsz " Visible GPU Agents (Nodes ",
        topology->gpu_agent_count));
  }
  for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
    uint32_t node = 0;
    IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa),
                                                 topology->gpu_agents[i],
                                                 HSA_AGENT_INFO_NODE, &node));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, i > 0 ? ", %u" : "%u", node));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV(")")));
  iree_string_builder_intern(builder, &intern_offset,
                             default_info ? &default_info->name : NULL);
  *device_info_count = *device_info_count + 1;

  // TODO(benvanik): use link information to represent clusters of devices.

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_populate_physical_device_info(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent,
    iree_host_size_t gpu_ordinal, iree_hal_device_info_t* device_info,
    iree_string_builder_t* builder) {
  iree_host_size_t intern_offset = iree_string_builder_size(builder);

  // Device IDs are bitfields indicating which GPUs are used.
  if (device_info) {
    uint64_t device_id = 1ull << gpu_ordinal;
    device_info->device_id = (iree_hal_device_id_t)device_id;
  }

  // Path is the device string ("GPU-0e12865a3bf5b7ab").
  // We could support a form without the GPU- but the ROCR tools display it
  // and this means allows easier copy/pasting.
  char agent_uuid[32];
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), gpu_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID,
      agent_uuid));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, agent_uuid));
  iree_string_builder_intern(builder, &intern_offset,
                             device_info ? &device_info->path : NULL);

  // Name is the product name plus the agent node.
  // At least on my device the name has a trailing space so we trim whitespace
  // before formatting for consistency.
  char product_name[64];
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(
      IREE_LIBHSA(libhsa), gpu_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, product_name));
  const iree_string_view_t trimmed_name =
      iree_string_view_trim(iree_make_cstring_view(product_name));
  uint32_t node = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), gpu_agent,
                                               HSA_AGENT_INFO_NODE, &node));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, trimmed_name));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, " (Node %u)", node));
  iree_string_builder_intern(builder, &intern_offset,
                             device_info ? &device_info->name : NULL);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_append_topology_device_infos(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology,
    iree_host_size_t* device_info_count, iree_hal_device_info_t* device_infos,
    iree_string_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // Add pseudo devices based on the topology.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_driver_append_pseudo_device_infos(
        libhsa, topology, device_info_count, device_infos, builder);
  }

  // Add one entry for every physical device.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
      status = iree_hal_amdgpu_driver_populate_physical_device_info(
          libhsa, topology->gpu_agents[i], i,
          device_infos ? &device_infos[*device_info_count] : NULL, builder);
      if (!iree_status_is_ok(status)) break;
      *device_info_count = *device_info_count + 1;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_driver_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_amdgpu_driver_options_t options;

  iree_hal_amdgpu_libhsa_t libhsa;

  // + trailing identifier string storage
} iree_hal_amdgpu_driver_t;

static const iree_hal_driver_vtable_t iree_hal_amdgpu_driver_vtable;

static iree_hal_amdgpu_driver_t* iree_hal_amdgpu_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_driver_vtable);
  return (iree_hal_amdgpu_driver_t*)base_value;
}

// Loads HSA using the provided driver options.
static iree_status_t iree_hal_amdgpu_driver_load_libhsa(
    const iree_hal_amdgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_amdgpu_libhsa_t* out_libhsa) {
  IREE_ASSERT_ARGUMENT(out_libhsa);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_libhsa_flags_t libhsa_flags =
      IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE;

  iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
      libhsa_flags, options->libhsa_search_paths, host_allocator, out_libhsa);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_driver_create(
    iree_string_view_t identifier,
    const iree_hal_amdgpu_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): verify options; this may be moved after any libraries are
  // loaded so the verification can use underlying implementation queries.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_driver_options_verify(options));

  iree_hal_amdgpu_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_amdgpu_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);

  // TODO(benvanik): if there are any string fields then they will need to be
  // retained as well (similar to the identifier they can be tagged on to the
  // end of the driver struct).
  memcpy(&driver->options, options, sizeof(*options));

  // Load HSA. The HSA runtime shared library may already be loaded and we
  // retain a copy during creation to ensure it doesn't get unloaded.
  iree_status_t status = iree_hal_amdgpu_driver_load_libhsa(
      &driver->options, host_allocator, &driver->libhsa);

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_amdgpu_driver_t* driver = iree_hal_amdgpu_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release HSA; it may remain loaded if there are other users or if live
  // devices have been created from it.
  iree_hal_amdgpu_libhsa_deinitialize(&driver->libhsa);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_amdgpu_driver_t* driver = iree_hal_amdgpu_driver_cast(base_driver);
  *out_device_info_count = 0;
  *out_device_infos = NULL;

  // Query available devices based on the default configuration.
  iree_hal_amdgpu_topology_t topology;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_topology_initialize_with_defaults(
      &driver->libhsa, &topology));
  if (topology.gpu_agent_count == 0) {
    return iree_ok_status();  // no devices
  }

  // Run the string builder in size calculation mode.
  // We'll format the devices once to figure out how much string storage is
  // required and then allocate + populate.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_null(), &builder);

  // Calculate the total device info count and calculate the size of the string
  // buffer required with a null string builder.
  iree_host_size_t device_info_count = 0;
  iree_status_t status = iree_hal_amdgpu_driver_append_topology_device_infos(
      &driver->libhsa, &topology, &device_info_count, NULL, &builder);

  // Calculate required size for the device info and its strings and allocate
  // the memory for it.
  iree_hal_device_info_t* device_infos = NULL;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t string_table_size =
        iree_string_builder_size(&builder) + /*NUL*/ 1;
    const iree_host_size_t total_size =
        device_info_count * sizeof(iree_hal_device_info_t) + string_table_size;
    status = iree_allocator_malloc(host_allocator, total_size,
                                   (void**)&device_infos);

    // Point the string builder at the allocated storage.
    // The population routines will take pointers into the storage.
    iree_string_builder_initialize_with_storage(
        (char*)device_infos + device_info_count * sizeof(device_infos[0]),
        string_table_size, &builder);
  }

  // Build the full set of device infos and populate the string table in-place.
  if (iree_status_is_ok(status)) {
    device_info_count = 0;  // reset
    status = iree_hal_amdgpu_driver_append_topology_device_infos(
        &driver->libhsa, &topology, &device_info_count, device_infos, &builder);
  }

  iree_string_builder_deinitialize(&builder);

  if (iree_status_is_ok(status)) {
    *out_device_info_count = device_info_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_amdgpu_driver_t* driver = iree_hal_amdgpu_driver_cast(base_driver);

  // TODO(benvanik): include list of available device library archs and indicate
  // which was selected. Could just have a string builder method in
  // device_library.h for something like `[amdgcn-blah-blah,
  // **amdgcn-blah-blah**, ...]`.

  // TODO(benvanik): query everything like rocminfo (so we don't have to ship
  // it).
  (void)driver;

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("\n")));

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_amdgpu_driver_t* driver = iree_hal_amdgpu_driver_cast(base_driver);

  // Use the provided params to overwrite the default options.
  // The format of the params is implementation-defined. The params strings can
  // be directly referenced if needed as the device creation is only allowed to
  // access them during the create call below.
  iree_hal_amdgpu_logical_device_options_t options =
      driver->options.default_device_options;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_options_parse(
      &options, (iree_string_pair_list_t){
                    .count = param_count,
                    .pairs = params,
                }));

  // Initialize the topology based on the device ID.
  // The ID is a bitfield of device ordinals defined by ROCR_VISIBLE_DEVICES.
  // If IREE_HAL_AMDGPU_DEVICE_ID_DEFAULT (0) then all visible devices will be
  // included.
  iree_hal_amdgpu_topology_t topology;
  iree_status_t status =
      iree_hal_amdgpu_topology_initialize_from_gpu_agent_mask(
          &driver->libhsa, (uint64_t)device_id, &topology);

  // Create the logical device composed of all physical devices specified.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_create(driver->identifier, &options,
                                                   &driver->libhsa, &topology,
                                                   host_allocator, out_device);
  }

  iree_hal_amdgpu_topology_deinitialize(&topology);
  return status;
}

static iree_status_t iree_hal_amdgpu_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_amdgpu_driver_t* driver = iree_hal_amdgpu_driver_cast(base_driver);

  // Use the provided params to overwrite the default options.
  // The format of the params is implementation-defined. The params strings can
  // be directly referenced if needed as the device creation is only allowed to
  // access them during the create call below.
  iree_hal_amdgpu_logical_device_options_t options =
      driver->options.default_device_options;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_options_parse(
      &options, (iree_string_pair_list_t){
                    .count = param_count,
                    .pairs = params,
                }));

  // Load HSA. HSA may already be loaded and we retain a copy during creation to
  // ensure it doesn't get unloaded.
  iree_hal_amdgpu_libhsa_t libhsa;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_driver_load_libhsa(
      &driver->options, host_allocator, &libhsa));

  // Initialize the topology with the given path. It may indicate multiple
  // devices and use different schemes to determine which devices are included.
  iree_hal_amdgpu_topology_t topology;
  iree_status_t status = iree_hal_amdgpu_topology_initialize_from_path(
      &libhsa, device_path, &topology);

  // Create the logical device composed of all physical devices specified.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_logical_device_create(driver->identifier, &options,
                                                   &libhsa, &topology,
                                                   host_allocator, out_device);
  }

  iree_hal_amdgpu_topology_deinitialize(&topology);
  iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  return status;
}

static const iree_hal_driver_vtable_t iree_hal_amdgpu_driver_vtable = {
    .destroy = iree_hal_amdgpu_driver_destroy,
    .query_available_devices = iree_hal_amdgpu_driver_query_available_devices,
    .dump_device_info = iree_hal_amdgpu_driver_dump_device_info,
    .create_device_by_id = iree_hal_amdgpu_driver_create_device_by_id,
    .create_device_by_path = iree_hal_amdgpu_driver_create_device_by_path,
};
