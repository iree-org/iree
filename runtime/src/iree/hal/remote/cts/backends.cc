// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the remote HAL driver.
//
// Registers a "remote_local_task" backend that creates a remote client device
// connected to a server wrapping a local-task device via loopback transport.
// The factory is parameterizable: adding other server devices (HIP, CUDA) or
// transports (TCP, SHM) requires registering additional backends with
// different factory functions.

#include <atomic>
#include <thread>

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/net/session.h"

namespace iree::hal::cts {
namespace {

static constexpr uint32_t kAxisTableCapacity = 16;

// Supporting infrastructure that must outlive the remote client device.
// The CTS caches one device per backend (global GTest environment), so
// there's at most one context per backend per process.
struct RemoteBackendContext {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  // Shared recv_pool used by both client and server (loopback: same process).
  iree_hal_remote_recv_pool_t* recv_pool = nullptr;
  iree_net_transport_factory_t* factory = nullptr;
  iree_async_axis_table_entry_t client_axis_entries[kAxisTableCapacity] = {};
  iree_async_frontier_tracker_t client_tracker = {};
  iree_async_axis_table_entry_t server_axis_entries[kAxisTableCapacity] = {};
  iree_async_frontier_tracker_t server_tracker = {};
  iree_hal_driver_t* server_driver = nullptr;
  iree_hal_device_t* server_device = nullptr;
  iree_hal_remote_server_t* server = nullptr;
  bool initialized = false;

  ~RemoteBackendContext() { Teardown(); }

  struct TeardownState {
    iree_hal_remote_server_t* server;
    std::atomic<int32_t> phase;
  };

  void Teardown() {
    if (!initialized) return;
    // The CTS releases the client device before calling this. That triggers
    // fire-and-forget RESOURCE_RELEASE_BATCH messages. Release the server on
    // the proactor thread so all pending messages, session teardown, and
    // carrier deactivation happen in the same context without races.
    // Release server and server-side resources.
    iree_hal_remote_server_release(server);
    server = nullptr;
    iree_hal_device_release(server_device);
    server_device = nullptr;
    iree_hal_driver_release(server_driver);
    server_driver = nullptr;
    iree_net_transport_factory_release(factory);
    factory = nullptr;
    iree_async_frontier_tracker_deinitialize(&server_tracker);
    iree_async_frontier_tracker_deinitialize(&client_tracker);
    iree_hal_remote_recv_pool_release(recv_pool);
    recv_pool = nullptr;
    // Proactor pool owns proactors and their threads — release stops them.
    iree_async_proactor_pool_release(proactor_pool);
    proactor_pool = nullptr;
    initialized = false;
  }
};

// GTest environment that tears down the remote backend context at program exit.
class RemoteBackendEnvironment : public ::testing::Environment {
 public:
  RemoteBackendContext* context() { return &context_; }
  void TearDown() override { context_.Teardown(); }

 private:
  RemoteBackendContext context_;
};

static RemoteBackendEnvironment* GetEnvironment() {
  static RemoteBackendEnvironment* env = [] {
    auto* e = new RemoteBackendEnvironment();
    ::testing::AddGlobalTestEnvironment(e);
    return e;
  }();
  return env;
}

// Creates the server-side local-task device.
static iree_status_t CreateLocalTaskServerDevice(
    iree_async_proactor_pool_t* proactor_pool, iree_hal_driver_t** out_driver,
    iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        driver, &create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

// Creates a remote client device connected to a server via loopback.
// |create_server_device| creates the server-side device+driver pair using
// the shared proactor pool.
static iree_status_t CreateRemoteDevice(
    iree_status_t (*create_server_device)(iree_async_proactor_pool_t*,
                                          iree_hal_driver_t**,
                                          iree_hal_device_t**),
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  RemoteBackendContext* ctx = GetEnvironment()->context();
  *out_driver = nullptr;
  *out_device = nullptr;

  // Create proactor pool (provides proactors with driving threads on demand).
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &ctx->proactor_pool);

  // Create shared recv_pool (used by both client and server in loopback mode).
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_recv_pool_create(
        ctx->proactor_pool, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        iree_allocator_system(), &ctx->recv_pool);
  }

  // Get the proactor from the recv_pool (needed for server creation).
  iree_async_proactor_t* proactor = nullptr;
  if (iree_status_is_ok(status)) {
    proactor = iree_hal_remote_recv_pool_proactor(ctx->recv_pool);
  }

  // Create frontier trackers.
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_initialize(
        &ctx->client_tracker, ctx->client_axis_entries, kAxisTableCapacity,
        iree_allocator_system());
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_initialize(
        &ctx->server_tracker, ctx->server_axis_entries, kAxisTableCapacity,
        iree_allocator_system());
  }

  // Create loopback transport.
  if (iree_status_is_ok(status)) {
    status = iree_net_loopback_factory_create(
        iree_net_loopback_factory_options_default(), iree_allocator_system(),
        &ctx->factory);
  }

  // Create the server-side device.
  if (iree_status_is_ok(status)) {
    status = create_server_device(ctx->proactor_pool, &ctx->server_driver,
                                  &ctx->server_device);
  }

  // Create and start the server.
  if (iree_status_is_ok(status)) {
    iree_async_axis_t server_axes[] = {0x0200};
    uint64_t server_epochs[] = {0};
    iree_net_session_topology_t server_topology = {};
    server_topology.axes = server_axes;
    server_topology.current_epochs = server_epochs;
    server_topology.axis_count = 1;
    server_topology.machine_index = 1;
    server_topology.session_epoch = 1;

    iree_hal_remote_server_options_t server_options;
    iree_hal_remote_server_options_initialize(&server_options);
    server_options.transport_factory = ctx->factory;
    server_options.bind_address = IREE_SV("cts-server");
    server_options.local_topology = &server_topology;
    server_options.max_connections = 1;

    iree_hal_device_t* devices[] = {ctx->server_device};
    status = iree_hal_remote_server_create(
        &server_options, devices, 1, proactor, &ctx->server_tracker,
        iree_hal_remote_recv_pool_buffer_pool(ctx->recv_pool),
        iree_allocator_system(), &ctx->server);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_start(ctx->server);
  }

  // Create the remote client device.
  iree_hal_device_t* client_device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = ctx->factory;
    client_options.server_address = IREE_SV("cts-server");

    iree_hal_device_create_params_t client_create_params =
        iree_hal_device_create_params_default();
    client_create_params.proactor_pool = ctx->proactor_pool;
    client_create_params.frontier.tracker = &ctx->client_tracker;

    status = iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, &client_create_params,
        ctx->recv_pool, iree_allocator_system(), &client_device);
  }

  // Connect and wait.
  if (iree_status_is_ok(status)) {
    // Synchronous connect: use a notification to block until callback fires.
    struct ConnectState {
      std::atomic<bool> fired{false};
      std::atomic<iree_status_code_t> code{IREE_STATUS_OK};
    };
    ConnectState connect_state;

    iree_hal_remote_client_device_connected_callback_t callback;
    callback.fn = [](void* user_data, iree_status_t status) {
      auto* state = static_cast<ConnectState*>(user_data);
      state->code = iree_status_code(status);
      iree_status_ignore(status);
      state->fired = true;
    };
    callback.user_data = &connect_state;

    status = iree_hal_remote_client_device_connect(client_device, callback);
    if (iree_status_is_ok(status)) {
      // Spin-wait for connect callback (proactor thread processes it).
      iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
      while (!connect_state.fired.load()) {
        if (iree_time_now() >= deadline) {
          status = iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                                    "CTS backend connect timed out");
          break;
        }
        std::this_thread::yield();
      }
      if (iree_status_is_ok(status) && connect_state.code != IREE_STATUS_OK) {
        status = iree_status_from_code(connect_state.code);
      }
    }
  }

  if (iree_status_is_ok(status)) {
    ctx->initialized = true;
    *out_driver = nullptr;  // Remote device has no driver.
    *out_device = client_device;
  } else {
    iree_hal_device_release(client_device);
    ctx->Teardown();
  }
  return status;
}

static iree_status_t CreateRemoteLocalTask(iree_hal_driver_t** out_driver,
                                           iree_hal_device_t** out_device) {
  return CreateRemoteDevice(CreateLocalTaskServerDevice, out_driver,
                            out_device);
}

static bool remote_local_task_registered_ = [] {
  BackendInfo info;
  info.name = "remote_local_task";
  info.factory = CreateRemoteLocalTask;
  info.unsupported_tests = {
      {"DriverTest.*",
       "remote devices are created directly, not through driver enumeration"},
      {"EventTest.*", "events not implemented"},
      {"ExecutableCacheTest.*",
       "remote client returns true for all formats (server validates)"},
      {"ExecutableTest.*",
       "export info queries require EXECUTABLE_QUERY_EXPORT RPC"},
      {"FileTest.*", "file I/O not implemented"},
      {"QueueHostCallTest.*", "host calls not implemented"},
  };
  CtsRegistry::RegisterBackend({
      "remote_local_task",
      std::move(info),
      {"mapping"},
  });
  return true;
}();

}  // namespace
}  // namespace iree::hal::cts
