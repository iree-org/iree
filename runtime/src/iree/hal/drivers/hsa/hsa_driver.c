// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/assert.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hsa/api.h"
#include "iree/hal/drivers/hsa/dynamic_symbols.h"
#include "iree/hal/drivers/hsa/hsa_device.h"
#include "iree/hal/drivers/hsa/status_util.h"

// Maximum device name length supported by the HSA HAL driver.
#define IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH 128

// Maximum number of GPU agents we support.
#define IREE_HAL_HSA_MAX_AGENTS 16

typedef struct iree_hal_hsa_driver_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  // Identifier used for registering the driver in the IREE driver registry.
  iree_string_view_t identifier;
  // HSA runtime API dynamic symbols to interact with the HSA system.
  iree_hal_hsa_dynamic_symbols_t hsa_symbols;

  // The default parameters for creating devices using this driver.
  iree_hal_hsa_device_params_t device_params;

  // The index of the default HSA device to use if multiple ones are available.
  int default_device_index;

  // Cached GPU agents.
  hsa_agent_t gpu_agents[IREE_HAL_HSA_MAX_AGENTS];
  iree_host_size_t gpu_agent_count;

  // Cached CPU agent (for memory operations).
  hsa_agent_t cpu_agent;
  bool has_cpu_agent;
} iree_hal_hsa_driver_t;

static const iree_hal_driver_vtable_t iree_hal_hsa_driver_vtable;

static iree_hal_hsa_driver_t* iree_hal_hsa_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_driver_vtable);
  return (iree_hal_hsa_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hsa_driver_options_initialize(
    iree_hal_hsa_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

// Callback for iterating agents to find GPU agents.
static hsa_status_t iree_hal_hsa_find_gpu_agents_callback(hsa_agent_t agent,
                                                          void* data) {
  iree_hal_hsa_driver_t* driver = (iree_hal_hsa_driver_t*)data;

  hsa_device_type_t device_type;
  hsa_status_t status =
      driver->hsa_symbols.hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    if (driver->gpu_agent_count < IREE_HAL_HSA_MAX_AGENTS) {
      driver->gpu_agents[driver->gpu_agent_count++] = agent;
    }
  } else if (device_type == HSA_DEVICE_TYPE_CPU && !driver->has_cpu_agent) {
    driver->cpu_agent = agent;
    driver->has_cpu_agent = true;
  }

  return HSA_STATUS_SUCCESS;
}

// Initializes the HSA system and enumerates agents.
static iree_status_t iree_hal_hsa_init(iree_hal_hsa_driver_t* driver) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = IREE_HSA_CALL_TO_STATUS(
      &driver->hsa_symbols, hsa_init(), "hsa_init");

  if (iree_status_is_ok(status)) {
    // Iterate agents to find GPU and CPU agents.
    driver->gpu_agent_count = 0;
    driver->has_cpu_agent = false;
    status = IREE_HSA_CALL_TO_STATUS(
        &driver->hsa_symbols,
        hsa_iterate_agents(iree_hal_hsa_find_gpu_agents_callback, driver),
        "hsa_iterate_agents");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_create_internal(
    iree_string_view_t identifier, const iree_hal_hsa_driver_options_t* options,
    const iree_hal_hsa_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_hsa_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_hsa_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  driver->default_device_index = options->default_device_index;

  iree_status_t status = iree_hal_hsa_dynamic_symbols_initialize(
      host_allocator, options->hsa_lib_search_path_count,
      options->hsa_lib_search_paths, &driver->hsa_symbols);

  memcpy(&driver->device_params, device_params, sizeof(driver->device_params));

  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_init(driver);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_hsa_driver_create(
    iree_string_view_t identifier, const iree_hal_hsa_driver_options_t* options,
    const iree_hal_hsa_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(device_params);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hsa_driver_create_internal(
      identifier, options, device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hsa_driver_destroy(iree_hal_driver_t* base_driver) {
  IREE_ASSERT_ARGUMENT(base_driver);

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Shutdown HSA runtime.
  if (driver->hsa_symbols.hsa_shut_down) {
    IREE_HSA_IGNORE_ERROR(&driver->hsa_symbols, hsa_shut_down());
  }

  iree_hal_hsa_dynamic_symbols_deinitialize(&driver->hsa_symbols);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

// Populates device information from the given HSA GPU agent.
static iree_status_t iree_hal_hsa_populate_device_info(
    hsa_agent_t agent, iree_hal_hsa_dynamic_symbols_t* syms,
    uint8_t* buffer_ptr, uint8_t** out_buffer_ptr,
    iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  char device_name[IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH];
  hsa_status_t status =
      syms->hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, device_name);
  if (status != HSA_STATUS_SUCCESS) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to get HSA agent name");
  }

  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = (iree_hal_device_id_t)agent.handle;

  // Create a device path from the agent handle.
  char device_path_str[32] = {0};
  snprintf(device_path_str, sizeof(device_path_str), "GPU-%" PRIx64,
           agent.handle);
  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_cstring_view(device_path_str), &out_device_info->path,
      (char*)buffer_ptr);

  iree_string_view_t device_name_str =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_str, &out_device_info->name, (char*)buffer_ptr);

  *out_buffer_ptr = buffer_ptr;
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t device_count = driver->gpu_agent_count;

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size =
      device_count * (sizeof(iree_hal_device_info_t) +
                      IREE_HAL_HSA_MAX_DEVICE_NAME_LENGTH * sizeof(char));
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);

  iree_host_size_t valid_device_count = 0;
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(*device_infos);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      status = iree_hal_hsa_populate_device_info(
          driver->gpu_agents[i], &driver->hsa_symbols, buffer_ptr, &buffer_ptr,
          &device_infos[valid_device_count]);
      if (!iree_status_is_ok(status)) break;
      ++valid_device_count;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = valid_device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);

  // Report path to the runtime library.
  iree_string_builder_t path_builder;
  iree_string_builder_initialize(builder->allocator, &path_builder);
  iree_status_t status = iree_hal_hsa_dynamic_symbols_append_path_to_builder(
      &driver->hsa_symbols, &path_builder);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        builder, "\n- hsa_runtime_dylib_path: %s", path_builder.buffer);
    iree_string_builder_deinitialize(&path_builder);
    IREE_RETURN_IF_ERROR(status);
  }

  // Find the agent for this device_id.
  hsa_agent_t agent;
  agent.handle = (uint64_t)device_id;

  // Get agent info.
  char agent_name[64] = {0};
  status = IREE_HSA_RESULT_TO_STATUS(
      &driver->hsa_symbols,
      driver->hsa_symbols.hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME,
                                             agent_name),
      "hsa_agent_get_info(NAME)");
  if (iree_status_is_ok(status)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "\n- agent-name: %s", agent_name));
  }

  // Get ISA name via AMD extension.
  char isa_name[128] = {0};
  status = IREE_HSA_RESULT_TO_STATUS(
      &driver->hsa_symbols,
      driver->hsa_symbols.hsa_agent_get_info(
          agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, isa_name),
      "hsa_agent_get_info(PRODUCT_NAME)");
  if (iree_status_is_ok(status)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "\n- product-name: %s", isa_name));
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_driver_select_default_device(
    iree_hal_driver_t* base_driver, iree_hal_hsa_dynamic_symbols_t* syms,
    int default_device_index, iree_allocator_t host_allocator,
    hsa_agent_t* out_agent) {
  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);

  if (driver->gpu_agent_count == 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no compatible HSA GPU devices were found");
  }
  if (default_device_index >= (int)driver->gpu_agent_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "default device %d not found (of %" PRIhsz
                            " enumerated)",
                            default_device_index, driver->gpu_agent_count);
  }
  *out_agent = driver->gpu_agents[default_device_index];
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  hsa_agent_t agent;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hsa_driver_select_default_device(
                base_driver, &driver->hsa_symbols, driver->default_device_index,
                host_allocator, &agent));
  } else {
    agent.handle = (uint64_t)device_id;
  }

  iree_string_view_t device_name = iree_make_cstring_view("hsa");

  // Attempt to create the device now.
  iree_status_t status = iree_hal_hsa_device_create(
      base_driver, device_name, &driver->device_params, &driver->hsa_symbols,
      agent, driver->cpu_agent, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_hsa_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  // Try to parse as an index.
  int32_t device_index = 0;
  if (iree_string_view_atoi_int32(device_path, &device_index)) {
    iree_hal_hsa_driver_t* driver = iree_hal_hsa_driver_cast(base_driver);
    if (device_index >= 0 && device_index < (int32_t)driver->gpu_agent_count) {
      hsa_agent_t agent = driver->gpu_agents[device_index];
      return iree_hal_hsa_driver_create_device_by_id(
          base_driver, (iree_hal_device_id_t)agent.handle, param_count, params,
          host_allocator, out_device);
    }
  }

  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "device path '%.*s' not found or invalid",
                          (int)device_path.size, device_path.data);
}

static const iree_hal_driver_vtable_t iree_hal_hsa_driver_vtable = {
    .destroy = iree_hal_hsa_driver_destroy,
    .query_available_devices = iree_hal_hsa_driver_query_available_devices,
    .dump_device_info = iree_hal_hsa_driver_dump_device_info,
    .create_device_by_id = iree_hal_hsa_driver_create_device_by_id,
    .create_device_by_path = iree_hal_hsa_driver_create_device_by_path,
};

