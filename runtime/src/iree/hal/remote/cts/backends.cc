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
#include "iree/async/util/proactor_thread.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/net/session.h"

namespace iree::hal::cts {
namespace {

static constexpr uint32_t kAxisTableCapacity = 16;

// Supporting infrastructure that must outlive the remote client device.
// The CTS caches one device per backend (global GTest environment), so
// there's at most one context per backend per process.
struct RemoteBackendContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_proactor_thread_t* proactor_thread = nullptr;
  iree_async_slab_t* slab = nullptr;
  iree_async_region_t* region = nullptr;
  iree_async_buffer_pool_t* recv_pool = nullptr;
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
    if (proactor_thread && server) {
      TeardownState state;
      state.server = server;
      state.phase.store(0, std::memory_order_relaxed);
      server = nullptr;

      iree_async_proactor_message_callback_t msg_callback;
      msg_callback.fn = [](iree_async_proactor_t*, uint64_t message_data,
                           void* ud) {
        auto* s = static_cast<TeardownState*>(ud);
        if (message_data == 1) {
          iree_hal_remote_server_release(s->server);
          s->server = nullptr;
        }
        s->phase.store(static_cast<int32_t>(message_data),
                       std::memory_order_release);
      };
      msg_callback.user_data = &state;
      iree_async_proactor_set_message_callback(proactor, msg_callback);

      // Phase 1: release server (processes pending fire-and-forget messages,
      // session teardown, carrier deactivation).
      iree_status_ignore(iree_async_proactor_send_message(proactor, 1));
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (state.phase.load(std::memory_order_acquire) < 1) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::yield();
      }
      // Phase 2: flush cascading work (disconnect notifications, etc.).
      iree_status_ignore(iree_async_proactor_send_message(proactor, 2));
      while (state.phase.load(std::memory_order_acquire) < 2) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::yield();
      }
    }
    if (server) {
      iree_hal_remote_server_release(server);
      server = nullptr;
    }
    if (server_device) {
      iree_hal_device_release(server_device);
      server_device = nullptr;
    }
    if (server_driver) {
      iree_hal_driver_release(server_driver);
      server_driver = nullptr;
    }
    if (factory) {
      iree_net_transport_factory_release(factory);
      factory = nullptr;
    }
    iree_async_frontier_tracker_deinitialize(&server_tracker);
    iree_async_frontier_tracker_deinitialize(&client_tracker);
    // Stop the proactor thread after all session/carrier teardown is complete
    // but before releasing the proactor and its registered buffers.
    if (proactor_thread) {
      iree_async_proactor_thread_request_stop(proactor_thread);
      iree_status_ignore(iree_async_proactor_thread_join(
          proactor_thread, IREE_DURATION_INFINITE));
      iree_status_ignore(
          iree_async_proactor_thread_consume_status(proactor_thread));
      iree_async_proactor_thread_release(proactor_thread);
      proactor_thread = nullptr;
    }
    if (recv_pool) {
      iree_async_buffer_pool_free(recv_pool);
      recv_pool = nullptr;
    }
    if (region) {
      iree_async_region_release(region);
      region = nullptr;
    }
    if (slab) {
      iree_async_slab_release(slab);
      slab = nullptr;
    }
    if (proactor) {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    }
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
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
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

  iree_async_proactor_pool_t* proactor_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_driver_create_default_device(
        driver, &create_params, iree_allocator_system(), &device);
  }

  iree_async_proactor_pool_release(proactor_pool);

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
// |create_server_device| creates the server-side device+driver pair.
static iree_status_t CreateRemoteDevice(
    iree_status_t (*create_server_device)(iree_hal_driver_t**,
                                          iree_hal_device_t**),
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  RemoteBackendContext* ctx = GetEnvironment()->context();
  *out_driver = nullptr;
  *out_device = nullptr;

  // Create proactor.
  iree_status_t status = iree_ok_status();
  iree_async_proactor_options_t proactor_options =
      iree_async_proactor_options_default();
  status = iree_async_proactor_create_platform(
      proactor_options, iree_allocator_system(), &ctx->proactor);

  // Create slab/region/recv_pool.
  if (iree_status_is_ok(status)) {
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    status = iree_async_slab_create(slab_options, iree_allocator_system(),
                                    &ctx->slab);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_slab(
        ctx->proactor, ctx->slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
        &ctx->region);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_buffer_pool_allocate(
        ctx->region, iree_allocator_system(), &ctx->recv_pool);
  }

  // Start proactor thread (needed for blocking control RPCs).
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_thread_create(
        ctx->proactor, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &ctx->proactor_thread);
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
    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    status = iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &ctx->factory);
  }

  // Create the server-side device.
  if (iree_status_is_ok(status)) {
    status = create_server_device(&ctx->server_driver, &ctx->server_device);
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
        &server_options, devices, 1, ctx->proactor, &ctx->server_tracker,
        ctx->recv_pool, iree_allocator_system(), &ctx->server);
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

    status = iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, /*create_params=*/nullptr,
        ctx->proactor, &ctx->client_tracker, ctx->recv_pool,
        iree_allocator_system(), &client_device);
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
