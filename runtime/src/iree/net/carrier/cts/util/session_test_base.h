// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for session CTS tests.
//
// Extends FactoryTestBase with frontier tracker initialization for both client
// and server sides, session callback tracking, and helpers to establish a
// bootstrapped session pair across any transport backend.
//
// The frontier tracker uses the initialize/deinitialize pattern with
// pre-allocated axis table entries. Each side gets its own tracker because in
// production, client and server are separate machines with separate trackers.
// The session registers the peer's axes (as proxy semaphores) in its local
// tracker.
//
// Session tests require the "factory" tag because they need the full
// factory -> listener -> connection -> session pipeline.

#ifndef IREE_NET_CARRIER_CTS_UTIL_SESSION_TEST_BASE_H_
#define IREE_NET_CARRIER_CTS_UTIL_SESSION_TEST_BASE_H_

#include <cstring>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/net/carrier/cts/util/factory_test_base.h"
#include "iree/net/session.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Session callback tracking
//===----------------------------------------------------------------------===//

// Captures all session callback invocations for test assertions.
//
// All callbacks store their arguments and set a fired flag. The on_ready
// callback copies remote topology data (which is only valid for the duration
// of the callback) into vectors for later inspection.
struct SessionCallbackTracker {
  // on_ready results.
  bool ready_fired = false;
  uint32_t remote_axis_count = 0;
  uint8_t remote_machine_index = 0;
  uint8_t remote_session_epoch = 0;
  std::vector<uint64_t> remote_axes;
  std::vector<uint64_t> remote_epochs;

  // on_goaway results.
  bool goaway_fired = false;
  uint32_t goaway_reason_code = 0;
  std::string goaway_message;

  // on_error results.
  bool error_fired = false;
  iree_status_code_t error_code = IREE_STATUS_OK;

  // on_control_data results.
  bool control_data_fired = false;
  std::vector<uint8_t> control_data;
  iree_net_control_frame_flags_t control_data_flags = 0;

  // Builds a callbacks struct that routes all callbacks to this tracker.
  iree_net_session_callbacks_t MakeCallbacks() {
    iree_net_session_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_ready = OnReady;
    callbacks.on_goaway = OnGoaway;
    callbacks.on_error = OnError;
    callbacks.on_control_data = OnControlData;
    callbacks.user_data = this;
    return callbacks;
  }

  static void OnReady(void* user_data, iree_net_session_t* session,
                      const iree_net_session_topology_t* remote_topology) {
    auto* self = static_cast<SessionCallbackTracker*>(user_data);
    self->ready_fired = true;
    self->remote_axis_count = remote_topology->axis_count;
    self->remote_machine_index = remote_topology->machine_index;
    self->remote_session_epoch = remote_topology->session_epoch;
    for (uint32_t i = 0; i < remote_topology->axis_count; ++i) {
      self->remote_axes.push_back((uint64_t)remote_topology->axes[i]);
      self->remote_epochs.push_back(remote_topology->current_epochs[i]);
    }
  }

  static void OnGoaway(void* user_data, iree_net_session_t* session,
                       uint32_t reason_code, iree_string_view_t message) {
    auto* self = static_cast<SessionCallbackTracker*>(user_data);
    self->goaway_fired = true;
    self->goaway_reason_code = reason_code;
    self->goaway_message = std::string(message.data, message.size);
  }

  static void OnError(void* user_data, iree_net_session_t* session,
                      iree_status_t status) {
    auto* self = static_cast<SessionCallbackTracker*>(user_data);
    self->error_fired = true;
    self->error_code = iree_status_code(status);
    iree_status_ignore(status);
  }

  static iree_status_t OnControlData(void* user_data,
                                     iree_net_control_frame_flags_t flags,
                                     iree_const_byte_span_t payload,
                                     iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<SessionCallbackTracker*>(user_data);
    self->control_data_fired = true;
    self->control_data_flags = flags;
    self->control_data.insert(self->control_data.end(), payload.data,
                              payload.data + payload.data_length);
    return iree_ok_status();
  }
};

//===----------------------------------------------------------------------===//
// Session test base fixture
//===----------------------------------------------------------------------===//

class SessionTestBase : public FactoryTestBase {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    FactoryTestBase::SetUp();

    memset(client_axis_entries_, 0, sizeof(client_axis_entries_));
    memset(server_axis_entries_, 0, sizeof(server_axis_entries_));

    IREE_ASSERT_OK(iree_async_frontier_tracker_initialize(
        &client_tracker_, client_axis_entries_, kAxisTableCapacity,
        iree_allocator_system()));
    IREE_ASSERT_OK(iree_async_frontier_tracker_initialize(
        &server_tracker_, server_axis_entries_, kAxisTableCapacity,
        iree_allocator_system()));
  }

  void TearDown() override {
    // Sessions in BOOTSTRAPPING have pending async callbacks that reference
    // the session as user_data. Poll until bootstrap completes or fails
    // before releasing, otherwise the pending callbacks cause UAF.
    PollUntil(
        [&]() {
          bool client_done =
              !client_session_ || iree_net_session_state(client_session_) !=
                                      IREE_NET_SESSION_STATE_BOOTSTRAPPING;
          bool server_done =
              !server_session_ || iree_net_session_state(server_session_) !=
                                      IREE_NET_SESSION_STATE_BOOTSTRAPPING;
          return client_done && server_done;
        },
        iree_make_duration_ms(2000));

    // Drain pending proactor operations before releasing sessions. Sessions
    // in DRAINING or OPERATIONAL state may have in-flight carrier NOPs (from
    // GOAWAY sends, control data, etc.) that reference the carrier as
    // user_data. Releasing the session destroys the carrier; if any NOPs are
    // still in the io_uring CQ, the proactor will UAF on them. A brief poll
    // window lets those NOPs complete while the carrier is still alive.
    PollUntil([&]() { return false; }, iree_make_duration_ms(100));

    if (server_session_) {
      iree_net_session_release(server_session_);
      server_session_ = nullptr;
    }
    if (client_session_) {
      iree_net_session_release(client_session_);
      client_session_ = nullptr;
    }

    // Second drain: process async cleanup from session/connection teardown.
    // Releasing sessions may trigger deactivation callbacks, listener stop
    // notifications, etc. that need a poll cycle to fire.
    PollUntil([&]() { return false; }, iree_make_duration_ms(100));

    if (listener_) {
      StopAndWait(listener_);
      iree_net_listener_free(listener_);
      listener_ = nullptr;
    }

    iree_async_frontier_tracker_deinitialize(&server_tracker_);
    iree_async_frontier_tracker_deinitialize(&client_tracker_);

    FactoryTestBase::TearDown();
  }

  //===--------------------------------------------------------------------===//
  // Session pair establishment
  //===--------------------------------------------------------------------===//

  // Creates a connected and bootstrapped session pair.
  //
  // On success, client_session_ and server_session_ are both OPERATIONAL,
  // listener_ is valid, and both callback trackers have captured their
  // on_ready invocations.
  //
  // The accept callback calls session_accept() directly so that the server's
  // endpoint NOP is submitted before the client's connect callback cascades
  // to HELLO. This guarantees the server carrier is ACTIVE when the HELLO
  // delivery NOP fires, which matters for zero-latency transports (loopback)
  // where the entire connect → endpoint → HELLO → delivery chain can
  // complete within a single proactor poll cycle.
  void EstablishSessionPair(const iree_net_session_topology_t& client_topology,
                            const iree_net_session_topology_t& server_topology,
                            uint64_t server_session_id = 42) {
    std::string bind_str = MakeBindAddress();
    iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

    // Context passed to the accept callback so it can create the server
    // session inline, before the connect callback fires.
    struct AcceptCtx {
      SessionTestBase* test = nullptr;
      iree_net_session_topology_t server_topology = {};
      uint64_t server_session_id = 0;
      bool fired = false;
    } accept_ctx;
    accept_ctx.test = this;
    accept_ctx.server_topology = server_topology;
    accept_ctx.server_session_id = server_session_id;

    IREE_CHECK_OK(iree_net_transport_factory_create_listener(
        factory_, bind_addr, proactor_, recv_pool_,
        [](void* user_data, iree_status_t status,
           iree_net_connection_t* connection) {
          auto* ctx = static_cast<AcceptCtx*>(user_data);
          IREE_CHECK_OK(status);

          iree_net_session_options_t server_options =
              iree_net_session_options_default();
          server_options.local_topology = ctx->server_topology;
          server_options.session_id = ctx->server_session_id;

          IREE_CHECK_OK(iree_net_session_accept(
              connection, ctx->test->proactor_, &ctx->test->server_tracker_,
              &server_options, ctx->test->server_callbacks_.MakeCallbacks(),
              iree_allocator_system(), &ctx->test->server_session_));

          // Release the accept callback's reference (the session retains it).
          iree_net_connection_release(connection);
          ctx->fired = true;
        },
        &accept_ctx, iree_allocator_system(), &listener_));

    std::string connect_str = ResolveConnectAddress(bind_str, listener_);

    // Start client session (async connect + bootstrap).
    iree_net_session_options_t client_options =
        iree_net_session_options_default();
    client_options.local_topology = client_topology;

    IREE_CHECK_OK(iree_net_session_connect(
        factory_,
        iree_make_string_view(connect_str.c_str(), connect_str.size()),
        proactor_, recv_pool_, &client_tracker_, &client_options,
        client_callbacks_.MakeCallbacks(), iree_allocator_system(),
        &client_session_));

    // Poll until both sessions complete bootstrap.
    ASSERT_TRUE(PollUntil([&]() {
      return client_callbacks_.ready_fired && server_callbacks_.ready_fired;
    })) << "Session bootstrap timed out";
  }

  // Establishes a session pair with simple single-axis topologies.
  // Client: axis 0x0100 at epoch 0, machine_index=0, session_epoch=1.
  // Server: axis 0x0200 at epoch 0, machine_index=1, session_epoch=1.
  void EstablishDefaultSessionPair() {
    iree_async_axis_t client_axes[] = {0x0100};
    uint64_t client_epochs[] = {0};
    iree_net_session_topology_t client_topo = {};
    client_topo.axes = client_axes;
    client_topo.current_epochs = client_epochs;
    client_topo.axis_count = 1;
    client_topo.machine_index = 0;
    client_topo.session_epoch = 1;

    iree_async_axis_t server_axes[] = {0x0200};
    uint64_t server_epochs[] = {0};
    iree_net_session_topology_t server_topo = {};
    server_topo.axes = server_axes;
    server_topo.current_epochs = server_epochs;
    server_topo.axis_count = 1;
    server_topo.machine_index = 1;
    server_topo.session_epoch = 1;

    EstablishSessionPair(client_topo, server_topo);
  }

  //===--------------------------------------------------------------------===//
  // Test state
  //===--------------------------------------------------------------------===//

  iree_async_axis_table_entry_t client_axis_entries_[kAxisTableCapacity];
  iree_async_frontier_tracker_t client_tracker_;

  iree_async_axis_table_entry_t server_axis_entries_[kAxisTableCapacity];
  iree_async_frontier_tracker_t server_tracker_;

  SessionCallbackTracker client_callbacks_;
  SessionCallbackTracker server_callbacks_;

  iree_net_session_t* client_session_ = nullptr;
  iree_net_session_t* server_session_ = nullptr;
  iree_net_listener_t* listener_ = nullptr;
};

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_SESSION_TEST_BASE_H_
