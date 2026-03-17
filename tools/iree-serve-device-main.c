// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-serve-device: Exposes local HAL devices to remote clients.
//
// Usage:
//   iree-serve-device --device=hip://3 --bind=tcp://0.0.0.0:5000
//   iree-serve-device --device=cuda://0 --bind=tcp://[::]:5000
//
// This tool creates a server that wraps a local HAL device and allows remote
// clients to execute operations on it. Clients connect using the remote HAL
// driver:
//
//   iree-run-module --device=remote://server:5000 --module=model.vmfb
//
// Multiple clients can connect to a single server. Each client gets an
// independent session with its own resource namespace.

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/tooling/device_util.h"

// Note: We use the standard --device flag from tooling/device_util.h.
// Users specify the device as: iree-serve-device --device=hip://0

IREE_FLAG(string, bind, "tcp://0.0.0.0:5000",
          "Address to bind the server to.\n"
          "Transport prefixes:\n"
          "  tcp://host:port       TCP sockets (default)\n"
          "  shm:///path           Shared memory (testing)");

IREE_FLAG(int32_t, max_connections, 16,
          "Maximum number of concurrent client connections.");

IREE_FLAG(bool, rdma, false, "Enable RDMA for bulk transfers when available.");

IREE_FLAG(bool, trace, false, "Enable server operation tracing for debugging.");

// Parses the transport prefix from a bind URI and returns the address portion.
static iree_status_t iree_serve_device_parse_bind_uri(
    iree_string_view_t bind_uri, iree_string_view_t* out_transport,
    iree_string_view_t* out_address) {
  // Find "://" separator.
  iree_string_view_t prefix = iree_string_view_empty();
  iree_string_view_t remainder = bind_uri;
  if (iree_string_view_consume_prefix(&remainder, IREE_SV("tcp://"))) {
    prefix = IREE_SV("tcp");
  } else if (iree_string_view_consume_prefix(&remainder, IREE_SV("shm://"))) {
    prefix = IREE_SV("shm");
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bind URI must have a transport prefix (tcp://, shm://), got: '%.*s'",
        (int)bind_uri.size, bind_uri.data);
  }
  *out_transport = prefix;
  *out_address = remainder;
  return iree_ok_status();
}

static iree_status_t iree_serve_device_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();

  // Create the local device to serve.
  // Uses the --device flag from tooling/device_util.h.
  iree_hal_device_t* device = NULL;
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  IREE_RETURN_IF_ERROR(iree_hal_create_device_from_flags(
      iree_hal_available_driver_registry(),
      /*default_device=*/iree_string_view_empty(), &create_params,
      host_allocator, &device));

  iree_string_view_t device_id = iree_hal_device_id(device);
  fprintf(stdout, "Created device: %.*s\n", (int)device_id.size,
          device_id.data);

  // Parse transport and address from --bind flag.
  iree_string_view_t transport_name = iree_string_view_empty();
  iree_string_view_t bind_address = iree_string_view_empty();
  iree_status_t status = iree_serve_device_parse_bind_uri(
      iree_make_cstring_view(FLAG_bind), &transport_name, &bind_address);

  // Configure server options.
  iree_hal_remote_server_t* server = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_remote_server_options_t options;
    iree_hal_remote_server_options_initialize(&options);
    options.bind_address = bind_address;
    options.max_connections = (uint32_t)FLAG_max_connections;
    if (FLAG_rdma) {
      options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
    }
    if (FLAG_trace) {
      options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS;
    }

    // Transport factory creation will be wired up when the server
    // implementation is complete. For now, server_create will fail because
    // transport_factory is NULL (the options_verify check catches this).
    fprintf(stdout, "Creating server: transport=%.*s address=%.*s\n",
            (int)transport_name.size, transport_name.data,
            (int)bind_address.size, bind_address.data);
    status = iree_hal_remote_server_create(&options, device, host_allocator,
                                           &server);
  }

  // Run the server event loop.
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Server starting...\n");
    status = iree_hal_remote_server_run(server);
  }

  // Cleanup.
  iree_hal_remote_server_release(server);
  iree_hal_device_release(device);

  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-serve-device",
      "Exposes local HAL devices to remote clients over the network.\n"
      "\n"
      "Examples:\n"
      "  # Serve a HIP device on port 5000 over TCP\n"
      "  iree-serve-device --device=hip://0 --bind=tcp://0.0.0.0:5000\n"
      "\n"
      "  # Serve over shared memory (for local testing)\n"
      "  iree-serve-device --device=hip://0 --bind=shm:///dev/shm/iree-gpu\n"
      "\n"
      "  # Connect from another machine (client uses matching carrier)\n"
      "  iree-run-module --device=remote-tcp://server:5000 "
      "--module=model.vmfb\n"
      "\n"
      "  # Connect via shared memory\n"
      "  iree-run-module --device=remote-shm:///dev/shm/iree-gpu "
      "--module=model.vmfb\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_serve_device_run();

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
