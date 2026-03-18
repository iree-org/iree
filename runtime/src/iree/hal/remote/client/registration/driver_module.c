// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/hal/remote/client/api.h"
#include "iree/net/transport_factory.h"

#if defined(IREE_HAVE_NET_TCP_TRANSPORT)
#include "iree/net/carrier/tcp/factory.h"
#endif  // IREE_HAVE_NET_TCP_TRANSPORT

#if defined(IREE_HAVE_NET_SHM_TRANSPORT)
#include "iree/net/carrier/shm/factory.h"
#endif  // IREE_HAVE_NET_SHM_TRANSPORT

// Remote HAL drivers organized by transport name.
// Each transport uses the same URI scheme (host:port or path) but different
// underlying transport mechanisms. This allows:
//   --device=remote-tcp://server:5000   (TCP sockets)
//   --device=remote-shm:///dev/shm/iree (Shared memory for testing)
//
// The transport name after "remote-" determines which factory is created.
// Available transports depend on what has been compiled in via the
// IREE_HAVE_NET_*_TRANSPORT defines, controlled by the enabled_transports
// build flag.

static const iree_string_view_t IREE_HAL_REMOTE_DRIVER_PREFIX =
    iree_string_view_literal("remote-");

static const iree_hal_driver_info_t iree_hal_remote_driver_infos[] = {
#if defined(IREE_HAVE_NET_TCP_TRANSPORT)
    {
        .driver_name = IREE_SVL("remote-tcp"),
        .full_name = IREE_SVL("Remote HAL Client (TCP)"),
    },
#endif  // IREE_HAVE_NET_TCP_TRANSPORT
#if defined(IREE_HAVE_NET_SHM_TRANSPORT)
    {
        .driver_name = IREE_SVL("remote-shm"),
        .full_name = IREE_SVL("Remote HAL Client (Shared Memory)"),
    },
#endif  // IREE_HAVE_NET_SHM_TRANSPORT
};

static iree_status_t iree_hal_remote_client_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  *out_driver_info_count = IREE_ARRAYSIZE(iree_hal_remote_driver_infos);
  *out_driver_infos = iree_hal_remote_driver_infos;
  return iree_ok_status();
}

// Extracts the transport name from a driver name like "remote-tcp" -> "tcp".
// Returns UNAVAILABLE if the driver name does not have the "remote-" prefix.
static iree_status_t iree_hal_remote_client_parse_transport_name(
    iree_string_view_t driver_name, iree_string_view_t* out_transport_name) {
  if (!iree_string_view_starts_with(driver_name,
                                    IREE_HAL_REMOTE_DRIVER_PREFIX)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  *out_transport_name = iree_string_view_substr(
      driver_name, IREE_HAL_REMOTE_DRIVER_PREFIX.size,
      driver_name.size - IREE_HAL_REMOTE_DRIVER_PREFIX.size);
  if (iree_string_view_is_empty(*out_transport_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "driver name '%.*s' has no transport suffix after 'remote-'",
        (int)driver_name.size, driver_name.data);
  }
  return iree_ok_status();
}

// Creates a transport factory for the given transport name.
// Returns UNAVAILABLE if the transport is not compiled in.
static iree_status_t iree_hal_remote_client_create_transport_factory(
    iree_string_view_t transport_name, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  *out_factory = NULL;

#if defined(IREE_HAVE_NET_TCP_TRANSPORT)
  if (iree_string_view_equal(transport_name, IREE_SV("tcp"))) {
    iree_net_tcp_carrier_options_t tcp_options =
        iree_net_tcp_carrier_options_default();
    // HAL remote uses multiple endpoints per connection (control channel +
    // queue channels).
    tcp_options.max_endpoint_count = 4;
    return iree_net_tcp_factory_create(tcp_options, host_allocator,
                                       out_factory);
  }
#endif  // IREE_HAVE_NET_TCP_TRANSPORT

#if defined(IREE_HAVE_NET_SHM_TRANSPORT)
  if (iree_string_view_equal(transport_name, IREE_SV("shm"))) {
    return iree_net_shm_factory_create(iree_net_shm_carrier_options_default(),
                                       host_allocator, out_factory);
  }
#endif  // IREE_HAVE_NET_SHM_TRANSPORT

  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "transport '%.*s' not compiled in",
                          (int)transport_name.size, transport_name.data);
}

static iree_status_t iree_hal_remote_client_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  // Extract the transport name from the driver name suffix.
  iree_string_view_t transport_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_parse_transport_name(
      driver_name, &transport_name));

  // Create the transport factory for this transport type.
  iree_net_transport_factory_t* factory = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_create_transport_factory(
      transport_name, host_allocator, &factory));

  // Set up driver options with the factory. The proactor, recv_pool, and
  // frontier_tracker are left NULL here — they are resolved from
  // create_params->proactor_pool during create_device_by_path.
  iree_hal_remote_client_driver_options_t options;
  iree_hal_remote_client_driver_options_initialize(&options);
  options.transport_factory = factory;
  options.default_device_options.transport_factory = factory;

  iree_status_t status = iree_hal_remote_client_driver_create(
      driver_name, &options, host_allocator, out_driver);
  iree_net_transport_factory_release(factory);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_remote_client_driver_factory_enumerate,
      .try_create = iree_hal_remote_client_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
