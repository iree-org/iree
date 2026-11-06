// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hexagon/device.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/hal/drivers/hexagon/allocator.h"
#include "iree/hal/drivers/hexagon/rpc_types.h"
#include "iree/hal/drivers/hexagon/utils.h"

// Hexagon SDK headers.
#include "AEEStdErr.h"
#include "remote.h"

// Generated from interface/hexagon_dsp.idl.
#include "hexagon_dsp.h"

//===----------------------------------------------------------------------===//
// Hexagon DSP domains
//===----------------------------------------------------------------------===//

// LOOKUP_HEXAGON_DOMAIN(domain ID, domain name, FastRPC URI domain suffix)
#define LOOKUP_HEXAGON_DOMAINS                               \
  LOOKUP_HEXAGON_DOMAIN(ADSP_DOMAIN_ID, "ADSP", ADSP_DOMAIN) \
  LOOKUP_HEXAGON_DOMAIN(MDSP_DOMAIN_ID, "MDSP", MDSP_DOMAIN) \
  LOOKUP_HEXAGON_DOMAIN(SDSP_DOMAIN_ID, "SDSP", SDSP_DOMAIN) \
  LOOKUP_HEXAGON_DOMAIN(CDSP_DOMAIN_ID, "CDSP", CDSP_DOMAIN) \
  LOOKUP_HEXAGON_DOMAIN(CDSP1_DOMAIN_ID, "CDSP1", CDSP1_DOMAIN)

bool iree_hal_hexagon_get_domain_name(iree_hal_hexagon_domain_id_t domain_id,
                                      const char** out_name) {
  switch (domain_id) {
#define LOOKUP_HEXAGON_DOMAIN(id, name, uri_suffix) \
  case id:                                          \
    *out_name = name;                               \
    return true;
    LOOKUP_HEXAGON_DOMAINS
#undef LOOKUP_HEXAGON_DOMAIN
    default:
      return false;
  }
}

bool iree_hal_hexagon_get_domain_id(
    iree_string_view_t name, iree_hal_hexagon_domain_id_t* out_domain_id) {
#define LOOKUP_HEXAGON_DOMAIN(id, check_name, uri_suffix)  \
  if (iree_string_view_equal(name, IREE_SV(check_name))) { \
    *out_domain_id = id;                                   \
    return true;                                           \
  }
  LOOKUP_HEXAGON_DOMAINS
#undef LOOKUP_HEXAGON_DOMAIN
  return false;
}

// Returns the FastRPC URI suffix selecting the DSP domain with |domain_id|
// (e.g. "&_dom=cdsp") in |out_uri_suffix|. Returns false if the domain is
// unknown.
static bool iree_hal_hexagon_get_domain_uri_suffix(
    iree_hal_hexagon_domain_id_t domain_id, const char** out_uri_suffix) {
  switch (domain_id) {
#define LOOKUP_HEXAGON_DOMAIN(id, name, uri_suffix) \
  case id:                                          \
    *out_uri_suffix = uri_suffix;                   \
    return true;
    LOOKUP_HEXAGON_DOMAINS
#undef LOOKUP_HEXAGON_DOMAIN
    default:
      return false;
  }
}

void iree_hal_hexagon_fill_device_info(
    iree_hal_hexagon_domain_id_t domain_id,
    iree_hal_device_info_t* out_device_info) {
  const char* name = "unknown";
  iree_hal_hexagon_get_domain_name(domain_id, &name);
  out_device_info->device_id =
      iree_hal_hexagon_device_id_from_domain_id(domain_id);
  out_device_info->name = iree_make_cstring_view(name);
  out_device_info->path = out_device_info->name;
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_options_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_hal_hexagon_device_options_initialize(
    iree_hal_hexagon_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

static iree_status_t iree_hal_hexagon_device_options_verify(
    const iree_hal_hexagon_device_options_t* options) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hexagon_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hexagon_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // DSP domain the device runs on.
  iree_hal_hexagon_domain_id_t domain_id;

  // Options the device was created with.
  iree_hal_hexagon_device_options_t options;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // FastRPC session with the DSP or 0 if not opened.
  rpc_session_handle_t rpc_session_handle;

  // Shared frontier tracker for cross-device causal ordering. Retained after
  // topology assignment and released during device destruction.
  iree_async_frontier_tracker_t* frontier_tracker;

  // This device's axis for frontier tracking.
  iree_async_axis_t axis;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Topology information if this device is part of a multi-device topology.
  iree_hal_device_topology_info_t topology_info;

  // + trailing identifier string storage
} iree_hal_hexagon_device_t;

static const iree_hal_device_vtable_t iree_hal_hexagon_device_vtable;

static iree_hal_hexagon_device_t* iree_hal_hexagon_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hexagon_device_vtable);
  return (iree_hal_hexagon_device_t*)base_value;
}

// Prints DSP process domain status changes to stderr.
static int iree_hal_hexagon_device_status_notification(
    void* context, int domain_id, int session,
    remote_rpc_status_flags_t status) {
  const char* status_str = "unknown status";
  switch (status) {
    case FASTRPC_USER_PD_UP:
      status_str = "PD is up";
      break;
    case FASTRPC_USER_PD_EXIT:
      status_str = "PD closed";
      break;
    case FASTRPC_USER_PD_FORCE_KILL:
      status_str = "PD force kill";
      break;
    case FASTRPC_USER_PD_EXCEPTION:
      status_str = "PD exception";
      break;
    case FASTRPC_DSP_SSR:
      status_str = "DSP subsystem restart";
      break;
    case FASTRPC_USERPD_TIMEOUT:
      status_str = "PD timeout";
      break;
    default:
      break;
  }
  const char* domain_name = "unknown";
  iree_hal_hexagon_get_domain_name(domain_id, &domain_name);
  fprintf(stderr, "hexagon: DSP %s status: %s\n", domain_name, status_str);
  return AEE_SUCCESS;
}

// Registers iree_hal_hexagon_device_status_notification for the DSP domain.
static iree_status_t iree_hal_hexagon_device_request_status_notifications(
    iree_hal_hexagon_domain_id_t domain_id) {
  struct remote_dsp_capability capability = {
      .domain = (uint32_t)domain_id,
      .attribute_ID = STATUS_NOTIFICATION_SUPPORT,
      .capability = 0,
  };
  int dsp_err = remote_handle_control(DSPRPC_GET_DSP_INFO, &capability,
                                      sizeof(capability));
  if ((dsp_err & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "FastRPC capability API is not supported");
  } else if (dsp_err != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "querying DSP status notification support failed");
  } else if (capability.capability != 1) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "DSP status notifications are not supported");
  }

  // The device is not passed as context as notifications may arrive after it
  // has been destroyed (e.g. when the session is closed).
  struct remote_rpc_notif_register notification = {
      .context = NULL,
      .domain = domain_id,
      .notifier_fn = iree_hal_hexagon_device_status_notification,
  };
  dsp_err = remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS,
                                   &notification, sizeof(notification));
  if (dsp_err != AEE_SUCCESS) {
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "enabling DSP status notifications failed");
  }
  return iree_ok_status();
}

// Returns the directory of the running executable including a trailing slash
// in |out_binary_dir|. The caller must free it with |host_allocator|.
static iree_status_t iree_hal_hexagon_get_binary_dir(
    iree_allocator_t host_allocator, char** out_binary_dir) {
  *out_binary_dir = NULL;

  // Resolve the executable symlink, growing the buffer until the path fits.
  char* path = NULL;
  iree_host_size_t capacity = 256;
  ssize_t length = 0;
  while (true) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        host_allocator, capacity, (void**)&path));
    length = readlink("/proc/self/exe", path, capacity);
    if (length < 0) {
      iree_status_t status = IREE_HAL_HEXAGON_MAKE_STATUS_FROM_ERRNO(
          "resolving the path of the executable failed");
      iree_allocator_free(host_allocator, path);
      return status;
    }
    // readlink() silently truncates, so a path filling the whole buffer may
    // have been truncated.
    if ((iree_host_size_t)length < capacity) break;
    iree_allocator_free(host_allocator, path);
    if (capacity >= 64 * 1024) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "path of the executable exceeds 64KiB");
    }
    capacity *= 2;
  }
  path[length] = 0;

  // Strip the executable name and keep the trailing slash.
  char* last_slash = strrchr(path, '/');
  if (!last_slash) {
    iree_status_t status =
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "no directory in the executable path '%s'", path);
    iree_allocator_free(host_allocator, path);
    return status;
  }
  last_slash[1] = 0;

  *out_binary_dir = path;
  return iree_ok_status();
}

// Appends the FastRPC URI opening a session on the DSP domain with |domain_id|
// to |builder|.
//
// The generated hexagon_dsp_URI has the format
//   "file:///" <library> "?" <skel invoke function> "&_modver=" <version>
//   "&_idlver=" <version>
// and is completed by appending the domain suffix (e.g. "&_dom=cdsp"). FastRPC
// requires the library to be given as absolute path (or "./<library>"), so the
// directory <executable dir>/../lib/hexagon/ is prepended to it. This allows
// running without setting DSP_LIBRARY_PATH.
static iree_status_t iree_hal_hexagon_append_session_uri(
    iree_hal_hexagon_domain_id_t domain_id, iree_allocator_t host_allocator,
    iree_string_builder_t* builder) {
  iree_string_view_t prefix = IREE_SV("file:///");
  iree_string_view_t library_uri = iree_make_cstring_view(hexagon_dsp_URI);
  if (!iree_string_view_consume_prefix(&library_uri, prefix)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "DSP URI '%s' does not start with '%.*s'",
                            hexagon_dsp_URI, (int)prefix.size, prefix.data);
  }

  const char* domain_uri_suffix = NULL;
  if (!iree_hal_hexagon_get_domain_uri_suffix(domain_id, &domain_uri_suffix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown DSP domain ID %d", domain_id);
  }

  char* binary_dir = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_hexagon_get_binary_dir(host_allocator, &binary_dir));
  iree_status_t status = iree_string_builder_append_format(
      builder, "%.*s%s../lib/hexagon/%.*s%s", (int)prefix.size, prefix.data,
      binary_dir, (int)library_uri.size, library_uri.data, domain_uri_suffix);
  iree_allocator_free(host_allocator, binary_dir);
  return status;
}

// Opens the FastRPC session with the DSP, which loads the device library.
static iree_status_t iree_hal_hexagon_device_open_session(
    iree_hal_hexagon_device_t* device) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (device->options.dsp_status_notify) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hexagon_device_request_status_notifications(
                device->domain_id));
  }

  // The device library is not signed, so it has to run in an unsigned PD.
  struct remote_rpc_control_unsigned_module unsigned_module = {
      .domain = device->domain_id,
      .enable = 1,
  };
  int dsp_err =
      remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_module,
                             sizeof(unsigned_module));
  if (dsp_err != AEE_SUCCESS) {
    IREE_TRACE_ZONE_END(z0);
    return IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
        dsp_err, "enabling unsigned modules on the DSP failed");
  }

  iree_string_builder_t uri_builder;
  iree_string_builder_initialize(device->host_allocator, &uri_builder);
  iree_status_t status = iree_hal_hexagon_append_session_uri(
      device->domain_id, device->host_allocator, &uri_builder);
  if (iree_status_is_ok(status)) {
    const char* uri = iree_string_builder_buffer(&uri_builder);
    if (device->options.verbose) {
      fprintf(stderr, "hexagon: opening DSP session '%s'\n", uri);
    }
    remote_handle64 rpc_session_handle = 0;
    dsp_err = hexagon_dsp_open(uri, &rpc_session_handle);
    if (dsp_err == AEE_SUCCESS) {
      device->rpc_session_handle = rpc_session_handle;
    } else {
      status = IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(
          dsp_err, "opening the DSP session failed");
    }
  }
  iree_string_builder_deinitialize(&uri_builder);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_hexagon_device_create(
    iree_string_view_t identifier, iree_hal_hexagon_domain_id_t domain_id,
    const iree_hal_hexagon_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(create_params);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_device = NULL;

  // Verify the parameters prior to creating resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hexagon_device_options_verify(options));

  iree_hal_hexagon_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  iree_hal_resource_initialize(&iree_hal_hexagon_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + total_size - identifier.size);
  device->domain_id = domain_id;
  device->options = *options;
  device->host_allocator = host_allocator;

  iree_status_t status = iree_hal_hexagon_allocator_create(
      host_allocator, &device->device_allocator);
  if (iree_status_is_ok(status)) {
    status = iree_hal_hexagon_device_open_session(device);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hexagon_device_clear_topology_info(
    iree_hal_hexagon_device_t* device) {
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        device->frontier_tracker, device->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof(device->topology_info));
}

static void iree_hal_hexagon_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (device->rpc_session_handle) {
    int dsp_err = hexagon_dsp_close(device->rpc_session_handle);
    if (device->options.verbose) {
      fprintf(stderr, "hexagon: closed DSP session (result 0x%x)\n", dsp_err);
    }
  }

  iree_hal_hexagon_device_clear_topology_info(device);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_hexagon_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_hexagon_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_hexagon_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_hexagon_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hexagon_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_hexagon_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_hexagon_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    // NOTE: this is a fuzzy match and can allow a program to work with multiple
    // device implementations.
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    // TODO(hexagon): report the supported executable formats once executables
    // are implemented.
    *out_value = 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = 1;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hexagon_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t*
iree_hal_hexagon_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_hexagon_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_hexagon_device_t* device = iree_hal_hexagon_device_cast(base_device);
  if (!topology_info) {
    iree_hal_hexagon_device_clear_topology_info(device);
    return iree_ok_status();
  }
  iree_async_frontier_tracker_t* frontier_tracker =
      topology_info->frontier.tracker;
  iree_async_axis_t axis = topology_info->frontier.base_axis;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  device->topology_info = *topology_info;
  device->frontier_tracker = frontier_tracker;
  device->axis = axis;
  iree_async_frontier_tracker_retain(device->frontier_tracker);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Unimplemented device functionality
//===----------------------------------------------------------------------===//

// TODO(hexagon): implement resources and queue operations. Until then the
// device can only be created and queried.

static iree_status_t iree_hal_hexagon_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  *out_channel = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon channels not implemented");
}

static iree_status_t iree_hal_hexagon_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon command buffers not implemented");
}

static iree_status_t iree_hal_hexagon_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  *out_event = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon events not implemented");
}

static iree_status_t iree_hal_hexagon_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  *out_executable_cache = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon executables not implemented");
}

static iree_status_t iree_hal_hexagon_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  *out_file = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon files not implemented");
}

static iree_status_t iree_hal_hexagon_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  *out_semaphore = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon semaphores not implemented");
}

static iree_hal_semaphore_compatibility_t
iree_hal_hexagon_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;
}

static iree_status_t iree_hal_hexagon_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue pools not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue alloca not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue dealloca not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue fill not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue update not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue copy not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue read not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue write not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue host call not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_function_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue dispatch not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon queue execute not implemented");
}

static iree_status_t iree_hal_hexagon_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Nothing is ever queued.
  return iree_ok_status();
}

static iree_status_t iree_hal_hexagon_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon device profiling not implemented");
}

static iree_status_t iree_hal_hexagon_device_profiling_flush(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon device profiling not implemented");
}

static iree_status_t iree_hal_hexagon_device_profiling_end(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon device profiling not implemented");
}

static iree_status_t iree_hal_hexagon_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon external capture not implemented");
}

static iree_status_t iree_hal_hexagon_device_external_capture_end(
    iree_hal_device_t* base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "hexagon external capture not implemented");
}

static const iree_hal_device_vtable_t iree_hal_hexagon_device_vtable = {
    .destroy = iree_hal_hexagon_device_destroy,
    .id = iree_hal_hexagon_device_id,
    .host_allocator = iree_hal_hexagon_device_host_allocator,
    .device_allocator = iree_hal_hexagon_device_allocator,
    .replace_device_allocator = iree_hal_hexagon_replace_device_allocator,
    .replace_channel_provider = iree_hal_hexagon_replace_channel_provider,
    .trim = iree_hal_hexagon_device_trim,
    .query_i64 = iree_hal_hexagon_device_query_i64,
    .query_capabilities = iree_hal_hexagon_device_query_capabilities,
    .topology_info = iree_hal_hexagon_device_topology_info,
    .refine_topology_edge = iree_hal_hexagon_device_refine_topology_edge,
    .assign_topology_info = iree_hal_hexagon_device_assign_topology_info,
    .create_channel = iree_hal_hexagon_device_create_channel,
    .create_command_buffer = iree_hal_hexagon_device_create_command_buffer,
    .create_event = iree_hal_hexagon_device_create_event,
    .create_executable_cache = iree_hal_hexagon_device_create_executable_cache,
    .import_file = iree_hal_hexagon_device_import_file,
    .create_semaphore = iree_hal_hexagon_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hexagon_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_hexagon_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_hexagon_device_queue_alloca,
    .queue_dealloca = iree_hal_hexagon_device_queue_dealloca,
    .queue_fill = iree_hal_hexagon_device_queue_fill,
    .queue_update = iree_hal_hexagon_device_queue_update,
    .queue_copy = iree_hal_hexagon_device_queue_copy,
    .queue_read = iree_hal_hexagon_device_queue_read,
    .queue_write = iree_hal_hexagon_device_queue_write,
    .queue_host_call = iree_hal_hexagon_device_queue_host_call,
    .queue_dispatch = iree_hal_hexagon_device_queue_dispatch,
    .queue_execute = iree_hal_hexagon_device_queue_execute,
    .queue_flush = iree_hal_hexagon_device_queue_flush,
    .profiling_begin = iree_hal_hexagon_device_profiling_begin,
    .profiling_flush = iree_hal_hexagon_device_profiling_flush,
    .profiling_end = iree_hal_hexagon_device_profiling_end,
    .external_capture_begin = iree_hal_hexagon_device_external_capture_begin,
    .external_capture_end = iree_hal_hexagon_device_external_capture_end,
};
