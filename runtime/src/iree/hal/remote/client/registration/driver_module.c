// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/hal/remote/client/api.h"

// Remote HAL drivers organized by carrier transport.
// Each carrier uses the same URI scheme (host:port or path) but different
// underlying transport mechanisms. This allows:
//   --device=remote-tcp://server:5000   (TCP sockets)
//   --device=remote-quic://server:5000  (QUIC/UDP)
//   --device=remote-ws://server:5000    (WebSockets)
//   --device=remote-shm:///dev/shm/iree (Shared memory for testing)

static iree_status_t iree_hal_remote_client_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  // Enumerate all carrier variants. The driver factory will report all
  // compiled-in carriers; try_create will fail for disabled ones.
  static const iree_hal_driver_info_t driver_infos[] = {
      {
          .driver_name = IREE_SVL("remote-tcp"),
          .full_name = IREE_SVL("Remote HAL Client (TCP)"),
      },
      {
          .driver_name = IREE_SVL("remote-quic"),
          .full_name = IREE_SVL("Remote HAL Client (QUIC)"),
      },
      {
          .driver_name = IREE_SVL("remote-ws"),
          .full_name = IREE_SVL("Remote HAL Client (WebSocket)"),
      },
      {
          .driver_name = IREE_SVL("remote-shm"),
          .full_name = IREE_SVL("Remote HAL Client (Shared Memory)"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

// Parses a driver name like "remote-tcp" and returns the carrier type.
static iree_status_t iree_hal_remote_client_parse_carrier(
    iree_string_view_t driver_name,
    iree_hal_remote_client_carrier_t* out_carrier) {
  if (iree_string_view_equal(driver_name, IREE_SV("remote-tcp"))) {
    *out_carrier = IREE_HAL_REMOTE_CLIENT_CARRIER_TCP;
    return iree_ok_status();
  }
  if (iree_string_view_equal(driver_name, IREE_SV("remote-quic"))) {
    *out_carrier = IREE_HAL_REMOTE_CLIENT_CARRIER_QUIC;
    return iree_ok_status();
  }
  if (iree_string_view_equal(driver_name, IREE_SV("remote-ws"))) {
    *out_carrier = IREE_HAL_REMOTE_CLIENT_CARRIER_WEBSOCKET;
    return iree_ok_status();
  }
  if (iree_string_view_equal(driver_name, IREE_SV("remote-shm"))) {
    *out_carrier = IREE_HAL_REMOTE_CLIENT_CARRIER_SHM;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "no driver '%.*s' is provided by this factory",
                          (int)driver_name.size, driver_name.data);
}

static iree_status_t iree_hal_remote_client_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  // Parse the carrier from the driver name.
  iree_hal_remote_client_carrier_t carrier;
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_parse_carrier(driver_name, &carrier));

  // Check if the requested carrier is supported.
  // Today only TCP and SHM are implemented.
  switch (carrier) {
    case IREE_HAL_REMOTE_CLIENT_CARRIER_TCP:
    case IREE_HAL_REMOTE_CLIENT_CARRIER_SHM:
      // Supported carriers.
      break;
    case IREE_HAL_REMOTE_CLIENT_CARRIER_QUIC:
    case IREE_HAL_REMOTE_CLIENT_CARRIER_WEBSOCKET:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "carrier '%.*s' is not yet implemented",
                              (int)driver_name.size, driver_name.data);
  }

  iree_hal_remote_client_driver_options_t options;
  iree_hal_remote_client_driver_options_initialize(&options);
  options.carrier = carrier;

  // Driver options can be populated from flags here for native tools.
  // Programmatic creation bypasses this and uses the options struct directly.

  iree_status_t status = iree_hal_remote_client_driver_create(
      driver_name, &options, host_allocator, out_driver);

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
