// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP carrier CTS backend registration.
//
// Registers the TCP carrier as a CTS backend with appropriate capability tags.
// The factory creates a connected TCP socket pair via localhost loopback,
// with separate buffer pools for each endpoint.

#include <atomic>
#include <cstring>
#include <string>

#include "iree/async/address.h"
#include "iree/async/buffer_pool.h"
#include "iree/async/operations/net.h"
#include "iree/async/platform/posix/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/async/socket.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/tcp/carrier.h"
#include "iree/net/carrier/tcp/factory.h"
#include "iree/net/transport_factory.h"

namespace iree::net::carrier::cts {
namespace {

//===----------------------------------------------------------------------===//
// Context for TCP carrier pair resources
//===----------------------------------------------------------------------===//

// Holds all resources needed for a TCP carrier pair (except the proactor).
// The proactor is owned by the test fixture or fuzz harness, not by this
// context. Cleanup releases in reverse order of creation.
struct TcpPairContext {
  iree_async_socket_t* listener = nullptr;

  // Client side resources.
  iree_async_slab_t* client_slab = nullptr;
  iree_async_region_t* client_region = nullptr;
  iree_async_buffer_pool_t* client_pool = nullptr;

  // Server side resources.
  iree_async_slab_t* server_slab = nullptr;
  iree_async_region_t* server_region = nullptr;
  iree_async_buffer_pool_t* server_pool = nullptr;

  ~TcpPairContext() {
    iree_async_buffer_pool_free(client_pool);
    iree_async_region_release(client_region);
    iree_async_slab_release(client_slab);

    iree_async_buffer_pool_free(server_pool);
    iree_async_region_release(server_region);
    iree_async_slab_release(server_slab);

    iree_async_socket_release(listener);
  }
};

void TcpPairCleanup(void* ctx_ptr) {
  delete static_cast<TcpPairContext*>(ctx_ptr);
}

//===----------------------------------------------------------------------===//
// Helper: Create slab/region/pool for one endpoint
//===----------------------------------------------------------------------===//

static iree_status_t CreateBufferPool(iree_async_proactor_t* proactor,
                                      iree_async_slab_t** out_slab,
                                      iree_async_region_t** out_region,
                                      iree_async_buffer_pool_t** out_pool) {
  iree_async_slab_options_t slab_options = {0};
  slab_options.buffer_size = 4096;
  slab_options.buffer_count = 16;
  IREE_RETURN_IF_ERROR(
      iree_async_slab_create(slab_options, iree_allocator_system(), out_slab));

  iree_status_t status = iree_async_proactor_register_slab(
      proactor, *out_slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, out_region);
  if (!iree_status_is_ok(status)) {
    iree_async_slab_release(*out_slab);
    *out_slab = nullptr;
    return status;
  }

  status = iree_async_buffer_pool_allocate(*out_region, iree_allocator_system(),
                                           out_pool);
  if (!iree_status_is_ok(status)) {
    iree_async_region_release(*out_region);
    *out_region = nullptr;
    iree_async_slab_release(*out_slab);
    *out_slab = nullptr;
    return status;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Factory function
//===----------------------------------------------------------------------===//

// Callbacks for connect/accept completion.
struct ConnectionState {
  std::atomic<int> completions{0};
  iree_async_socket_t* accepted_socket = nullptr;
  iree_status_t accept_status = iree_ok_status();
  iree_status_t connect_status = iree_ok_status();
};

static void AcceptCallback(void* user_data, iree_async_operation_t* operation,
                           iree_status_t status,
                           iree_async_completion_flags_t flags) {
  auto* state = static_cast<ConnectionState*>(user_data);
  auto* accept_op = (iree_async_socket_accept_operation_t*)operation;
  state->accept_status = status;
  if (iree_status_is_ok(status)) {
    state->accepted_socket = accept_op->accepted_socket;
  }
  state->completions.fetch_add(1, std::memory_order_release);
}

static void ConnectCallback(void* user_data, iree_async_operation_t* operation,
                            iree_status_t status,
                            iree_async_completion_flags_t flags) {
  auto* state = static_cast<ConnectionState*>(user_data);
  state->connect_status = status;
  state->completions.fetch_add(1, std::memory_order_release);
}

// Implementation: creates a TCP carrier pair on a given proactor.
// Callers should use CreateTcpCarrierPair() which handles optional proactor
// creation.
static iree::StatusOr<CarrierPair> CreateTcpCarrierPairImpl(
    iree_async_proactor_t* proactor) {
  auto ctx = std::make_unique<TcpPairContext>();

  // Create listener on localhost ephemeral port.
  IREE_RETURN_IF_ERROR(iree_async_socket_create(
      proactor, IREE_ASYNC_SOCKET_TYPE_TCP, IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
      &ctx->listener));

  iree_async_address_t bind_address;
  IREE_RETURN_IF_ERROR(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &bind_address));
  IREE_RETURN_IF_ERROR(iree_async_socket_bind(ctx->listener, &bind_address));
  IREE_RETURN_IF_ERROR(iree_async_socket_listen(ctx->listener, 16));

  iree_async_address_t listen_address;
  IREE_RETURN_IF_ERROR(
      iree_async_socket_query_local_address(ctx->listener, &listen_address));

  // Submit accept operation.
  ConnectionState conn_state;
  iree_async_socket_accept_operation_t accept_op;
  memset(&accept_op, 0, sizeof(accept_op));
  iree_async_operation_initialize(
      &accept_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
      IREE_ASYNC_OPERATION_FLAG_NONE, AcceptCallback, &conn_state);
  accept_op.listen_socket = ctx->listener;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_submit_one(proactor, &accept_op.base));

  // Create and connect client socket.
  iree_async_socket_t* client_socket = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_socket_create(
      proactor, IREE_ASYNC_SOCKET_TYPE_TCP, IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
      &client_socket));

  iree_async_socket_connect_operation_t connect_op;
  memset(&connect_op, 0, sizeof(connect_op));
  iree_async_operation_initialize(
      &connect_op.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
      IREE_ASYNC_OPERATION_FLAG_NONE, ConnectCallback, &conn_state);
  connect_op.socket = client_socket;
  connect_op.address = listen_address;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_submit_one(proactor, &connect_op.base));

  // Poll until both connect and accept complete.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
  while (conn_state.completions.load(std::memory_order_acquire) < 2) {
    if (iree_time_now() >= deadline) {
      iree_async_socket_release(client_socket);
      return iree::Status(iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                                           "TCP connection timed out"));
    }
    iree_host_size_t count = 0;
    iree_status_t status =
        iree_async_proactor_poll(proactor, iree_make_timeout_ms(100), &count);
    if (!iree_status_is_ok(status) &&
        !iree_status_is_deadline_exceeded(status)) {
      iree_async_socket_release(client_socket);
      return iree::Status(std::move(status));
    }
    iree_status_ignore(status);
  }

  // Check connection results.
  if (!iree_status_is_ok(conn_state.accept_status)) {
    iree_async_socket_release(client_socket);
    return iree::Status(std::move(conn_state.accept_status));
  }
  if (!iree_status_is_ok(conn_state.connect_status)) {
    iree_async_socket_release(client_socket);
    if (conn_state.accepted_socket) {
      iree_async_socket_release(conn_state.accepted_socket);
    }
    return iree::Status(std::move(conn_state.connect_status));
  }

  iree_async_socket_t* server_socket = conn_state.accepted_socket;

  // Create buffer pools for both endpoints.
  IREE_RETURN_IF_ERROR(CreateBufferPool(
      proactor, &ctx->client_slab, &ctx->client_region, &ctx->client_pool));
  IREE_RETURN_IF_ERROR(CreateBufferPool(
      proactor, &ctx->server_slab, &ctx->server_region, &ctx->server_pool));

  // Create TCP carriers.
  // Carriers take ownership of the sockets on success.
  iree_net_carrier_t* client_carrier = nullptr;
  iree_net_carrier_t* server_carrier = nullptr;
  iree_net_carrier_callback_t callback = {nullptr, nullptr};

  iree_status_t status = iree_net_tcp_carrier_allocate(
      proactor, client_socket, ctx->client_pool,
      iree_net_tcp_carrier_options_default(), callback, iree_allocator_system(),
      &client_carrier);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(client_socket);
    iree_async_socket_release(server_socket);
    return iree::Status(std::move(status));
  }

  status = iree_net_tcp_carrier_allocate(
      proactor, server_socket, ctx->server_pool,
      iree_net_tcp_carrier_options_default(), callback, iree_allocator_system(),
      &server_carrier);
  if (!iree_status_is_ok(status)) {
    iree_net_carrier_release(client_carrier);
    iree_async_socket_release(server_socket);
    return iree::Status(std::move(status));
  }

  // Build result.
  CarrierPair pair;
  pair.client = client_carrier;
  pair.server = server_carrier;
  pair.proactor = proactor;
  pair.context = ctx.release();  // Transfer ownership.
  pair.cleanup = TcpPairCleanup;

  return pair;
}

// Creates a TCP carrier pair. When |proactor| is NULL a new proactor is
// created and returned via pair.proactor; otherwise the caller's proactor is
// used (the production pattern for connection pools and proactor reuse).
static iree::StatusOr<CarrierPair> CreateTcpCarrierPair(
    iree_async_proactor_t* proactor) {
  bool created_proactor = (proactor == nullptr);
  if (created_proactor) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
  }
  auto result = CreateTcpCarrierPairImpl(proactor);
  if (!result.ok() && created_proactor) {
    iree_async_proactor_release(proactor);
  }
  return result;
}

// Creates a TCP carrier pair using a POSIX proactor with the specified event
// backend. Returns UNAVAILABLE when the POSIX proactor or its event backend
// is not supported on this platform.
static iree::StatusOr<CarrierPair> CreateTcpCarrierPairPosix(
    iree_async_proactor_t* proactor,
    iree_async_posix_event_backend_t event_backend) {
  bool created_proactor = (proactor == nullptr);
  if (created_proactor) {
    iree_status_t status = iree_async_proactor_create_posix_with_backend(
        iree_async_proactor_options_default(), event_backend,
        iree_allocator_system(), &proactor);
    if (!iree_status_is_ok(status)) return iree::Status(std::move(status));
  }
  auto result = CreateTcpCarrierPairImpl(proactor);
  if (!result.ok() && created_proactor) {
    iree_async_proactor_release(proactor);
  }
  return result;
}

static iree::StatusOr<CarrierPair> CreateTcpCarrierPairEpoll(
    iree_async_proactor_t* proactor) {
  return CreateTcpCarrierPairPosix(proactor,
                                   IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL);
}

static iree::StatusOr<CarrierPair> CreateTcpCarrierPairPoll(
    iree_async_proactor_t* proactor) {
  return CreateTcpCarrierPairPosix(proactor,
                                   IREE_ASYNC_POSIX_EVENT_BACKEND_POLL);
}

//===----------------------------------------------------------------------===//
// Factory-level CTS support
//===----------------------------------------------------------------------===//

static iree_status_t CreateTcpFactory(
    iree_allocator_t allocator, iree_net_transport_factory_t** out_factory) {
  return iree_net_tcp_factory_create(iree_net_tcp_carrier_options_default(),
                                     allocator, out_factory);
}

static std::string MakeTcpBindAddress() {
  // TCP always binds to localhost ephemeral port.
  return "127.0.0.1:0";
}

static std::string ResolveTcpConnectAddress(const std::string& /*bind_address*/,
                                            iree_net_listener_t* listener) {
  // Query the listener for the actual bound address (ephemeral port assigned).
  char address_buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t bound_address;
  IREE_CHECK_OK(iree_net_listener_query_bound_address(
      listener, sizeof(address_buffer), address_buffer, &bound_address));
  return std::string(bound_address.data, bound_address.size);
}

static std::string MakeTcpUnreachableAddress(iree_async_proactor_t* proactor) {
  // Bind+close a socket to get a port that is definitely not being listened on.
  iree_async_socket_t* temp_socket = nullptr;
  IREE_CHECK_OK(iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
                                         IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR,
                                         &temp_socket));
  iree_async_address_t temp_address;
  IREE_CHECK_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &temp_address));
  IREE_CHECK_OK(iree_async_socket_bind(temp_socket, &temp_address));
  iree_async_address_t bound_address;
  IREE_CHECK_OK(
      iree_async_socket_query_local_address(temp_socket, &bound_address));
  char address_buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t address_str;
  IREE_CHECK_OK(iree_async_address_format(
      &bound_address, sizeof(address_buffer), address_buffer, &address_str));
  iree_async_socket_release(temp_socket);
  return std::string(address_str.data, address_str.size);
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// TCP on io_uring: reliable, ordered, supports zero-copy TX.
static bool tcp_registered =
    (CtsRegistry::RegisterBackend(
         {"tcp",
          {"tcp", CreateTcpCarrierPair, CreateTcpFactory, MakeTcpBindAddress,
           ResolveTcpConnectAddress, MakeTcpUnreachableAddress},
          {"reliable", "ordered", "zerocopy_tx", "factory"}}),
     true);

// TCP on POSIX epoll: reliable, ordered, no zero-copy.
static bool tcp_epoll_registered =
    (CtsRegistry::RegisterBackend({
         "tcp_epoll",
         {"tcp_epoll", CreateTcpCarrierPairEpoll},
         {"reliable", "ordered"},
     }),
     true);

// TCP on POSIX poll: reliable, ordered, no zero-copy.
static bool tcp_poll_registered = (CtsRegistry::RegisterBackend({
                                       "tcp_poll",
                                       {"tcp_poll", CreateTcpCarrierPairPoll},
                                       {"reliable", "ordered"},
                                   }),
                                   true);

}  // namespace
}  // namespace iree::net::carrier::cts
