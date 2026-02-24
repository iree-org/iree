// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for socket operations.
//
// Measures the performance of TCP socket operations including roundtrip
// latency, throughput, and accept rate.

#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/buffer_pool.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/operations/net.h"
#include "iree/async/proactor.h"
#include "iree/async/slab.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Zero-copy mode for throughput comparison
//===----------------------------------------------------------------------===//

// Zero-copy send modes for benchmarking different paths.
// Zero-copy is controlled at socket creation via
// IREE_ASYNC_SOCKET_OPTION_ZERO_COPY, not per-send. These modes determine how
// the client socket is configured.
enum class ZeroCopyMode {
  // Regular send (no ZERO_COPY socket option).
  kCopy,
  // ZERO_COPY socket option with unregistered buffer (ad-hoc page pinning).
  kAdHoc,
  // ZERO_COPY socket option with registered buffer (fixed buffer, no per-op
  // pinning).
  kFixed,
};

// Returns a string representation of the ZeroCopyMode for benchmark names.
static const char* ZeroCopyModeName(ZeroCopyMode mode) {
  switch (mode) {
    case ZeroCopyMode::kCopy:
      return "Copy";
    case ZeroCopyMode::kAdHoc:
      return "AdHoc";
    case ZeroCopyMode::kFixed:
      return "Fixed";
  }
  return "Unknown";
}

//===----------------------------------------------------------------------===//
// Benchmark context
//===----------------------------------------------------------------------===//

// Holds resources for a loopback benchmark session.
struct LoopbackContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_socket_t* listener = nullptr;
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_address_t address = {};

  // Completion tracking for async operations.
  bool completed = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_host_size_t bytes_transferred = 0;

  // Zero-copy achievement tracking.
  iree_host_size_t zc_achieved_count = 0;

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<LoopbackContext*>(user_data);
    ctx->completed = true;
    ctx->status_code = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void RecvCallback(void* user_data, iree_async_operation_t* operation,
                           iree_status_t status,
                           iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<LoopbackContext*>(user_data);
    ctx->completed = true;
    ctx->status_code = iree_status_code(status);
    if (iree_status_is_ok(status)) {
      auto* recv_op =
          reinterpret_cast<iree_async_socket_recv_operation_t*>(operation);
      ctx->bytes_transferred = recv_op->bytes_received;
    }
    iree_status_ignore(status);
  }

  static void SendCallback(void* user_data, iree_async_operation_t* operation,
                           iree_status_t status,
                           iree_async_completion_flags_t flags) {
    auto* ctx = static_cast<LoopbackContext*>(user_data);
    ctx->completed = true;
    ctx->status_code = iree_status_code(status);
    if (iree_status_is_ok(status)) {
      auto* send_op =
          reinterpret_cast<iree_async_socket_send_operation_t*>(operation);
      ctx->bytes_transferred = send_op->bytes_sent;
      // Track if zero-copy was actually achieved (vs falling back to copy).
      if (flags & IREE_ASYNC_COMPLETION_FLAG_ZERO_COPY_ACHIEVED) {
        ++ctx->zc_achieved_count;
      }
    }
    iree_status_ignore(status);
  }

  void Reset() {
    completed = false;
    status_code = IREE_STATUS_OK;
    bytes_transferred = 0;
    // Note: zc_achieved_count is NOT reset here - it accumulates across the
    // benchmark run. Reset it explicitly when starting a new benchmark.
  }

  void ResetZCTracking() { zc_achieved_count = 0; }

  // Polls until completed or timeout.
  bool PollUntilComplete(iree_duration_t budget_ns = 5000000000LL) {
    iree_time_t deadline = iree_time_now() + budget_ns;
    while (!completed) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t count = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor, iree_make_timeout_ms(100), &count);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        return false;
      }
      iree_status_ignore(status);
    }
    return status_code == IREE_STATUS_OK;
  }
};

// Creates a loopback context with connected client/server sockets.
// Returns nullptr on failure (and sets state error).
// |client_options| are applied to the client socket at creation. Use
// IREE_ASYNC_SOCKET_OPTION_ZERO_COPY to enable zero-copy sends.
static LoopbackContext* CreateLoopbackContext(
    const ProactorFactory& factory, ::benchmark::State& state,
    iree_async_socket_options_t client_options =
        IREE_ASYNC_SOCKET_OPTION_NONE) {
  auto* ctx = new LoopbackContext();

  // Create proactor.
  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create listener.
  iree_status_t status = iree_async_socket_create(
      ctx->proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
      IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR | IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
      &ctx->listener);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Listener creation failed");
    iree_status_ignore(status);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Bind to loopback and listen.
  iree_async_address_t bind_addr;
  iree_async_address_from_ipv4(IREE_SV("127.0.0.1"), 0, &bind_addr);
  status = iree_async_socket_bind(ctx->listener, &bind_addr);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(ctx->listener, 16);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_async_socket_query_local_address(ctx->listener, &ctx->address);
  }
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Bind/listen failed");
    iree_status_ignore(status);
    iree_async_socket_release(ctx->listener);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Create client socket.
  status = iree_async_socket_create(
      ctx->proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | client_options, &ctx->client);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Client creation failed");
    iree_status_ignore(status);
    iree_async_socket_release(ctx->listener);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Submit accept.
  iree_async_socket_accept_operation_t accept_op;
  memset(&accept_op, 0, sizeof(accept_op));
  accept_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT;
  accept_op.base.completion_fn = LoopbackContext::Callback;
  accept_op.base.user_data = ctx;
  accept_op.listen_socket = ctx->listener;

  status = iree_async_proactor_submit_one(ctx->proactor, &accept_op.base);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Accept submit failed");
    iree_status_ignore(status);
    iree_async_socket_release(ctx->client);
    iree_async_socket_release(ctx->listener);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Submit connect.
  ctx->Reset();
  iree_async_socket_connect_operation_t connect_op;
  memset(&connect_op, 0, sizeof(connect_op));
  connect_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT;
  connect_op.base.completion_fn = LoopbackContext::Callback;
  connect_op.base.user_data = ctx;
  connect_op.socket = ctx->client;
  connect_op.address = ctx->address;

  status = iree_async_proactor_submit_one(ctx->proactor, &connect_op.base);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Connect submit failed");
    iree_status_ignore(status);
    iree_async_socket_release(ctx->client);
    iree_async_socket_release(ctx->listener);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Wait for both to complete.
  int completions = 0;
  iree_time_t deadline = iree_time_now() + 5000000000LL;
  while (completions < 2) {
    if (iree_time_now() >= deadline) {
      state.SkipWithError("Connect/accept timeout");
      iree_async_socket_release(ctx->client);
      iree_async_socket_release(ctx->listener);
      iree_async_proactor_release(ctx->proactor);
      delete ctx;
      return nullptr;
    }
    ctx->completed = false;
    iree_host_size_t count = 0;
    status = iree_async_proactor_poll(ctx->proactor, iree_make_timeout_ms(100),
                                      &count);
    if (!iree_status_is_ok(status) &&
        !iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    }
    iree_status_ignore(status);
    completions += (int)count;
  }

  ctx->server = accept_op.accepted_socket;
  if (!ctx->server) {
    state.SkipWithError("Accept failed - no server socket");
    iree_async_socket_release(ctx->client);
    iree_async_socket_release(ctx->listener);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  return ctx;
}

// Destroys a loopback context.
static void DestroyLoopbackContext(LoopbackContext* ctx) {
  if (!ctx) return;
  iree_async_socket_release(ctx->server);
  iree_async_socket_release(ctx->client);
  iree_async_socket_release(ctx->listener);
  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

//===----------------------------------------------------------------------===//
// Benchmark implementations
//===----------------------------------------------------------------------===//

// Roundtrip latency: send N bytes client->server, recv, send server->client,
// recv.
static void BM_Roundtrip(::benchmark::State& state,
                         const ProactorFactory& factory, size_t message_size) {
  auto* ctx = CreateLoopbackContext(factory, state);
  if (!ctx) return;

  // Allocate buffers.
  std::vector<uint8_t> send_buffer(message_size, 0xAB);
  std::vector<uint8_t> recv_buffer(message_size);
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_buffer.data(), message_size);
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer.data(), message_size);

  for (auto _ : state) {
    // Client -> Server.
    iree_async_socket_send_operation_t send_op;
    memset(&send_op, 0, sizeof(send_op));
    send_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
    send_op.base.completion_fn = LoopbackContext::SendCallback;
    send_op.base.user_data = ctx;
    send_op.socket = ctx->client;
    send_op.buffers.values = &send_span;
    send_op.buffers.count = 1;

    iree_async_socket_recv_operation_t recv_op;
    memset(&recv_op, 0, sizeof(recv_op));
    recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
    recv_op.base.completion_fn = LoopbackContext::RecvCallback;
    recv_op.base.user_data = ctx;
    recv_op.socket = ctx->server;
    recv_op.buffers.values = &recv_span;
    recv_op.buffers.count = 1;

    ctx->Reset();
    iree_async_proactor_submit_one(ctx->proactor, &send_op.base);
    ctx->PollUntilComplete();

    ctx->Reset();
    iree_async_proactor_submit_one(ctx->proactor, &recv_op.base);
    ctx->PollUntilComplete();

    // Server -> Client.
    send_op.socket = ctx->server;
    recv_op.socket = ctx->client;

    ctx->Reset();
    iree_async_proactor_submit_one(ctx->proactor, &send_op.base);
    ctx->PollUntilComplete();

    ctx->Reset();
    iree_async_proactor_submit_one(ctx->proactor, &recv_op.base);
    ctx->PollUntilComplete();
  }

  state.SetBytesProcessed(state.iterations() * message_size * 2);
  DestroyLoopbackContext(ctx);
}

// Unidirectional throughput: stream data client->server as fast as possible.
static void BM_Throughput(::benchmark::State& state,
                          const ProactorFactory& factory, size_t buffer_size) {
  auto* ctx = CreateLoopbackContext(factory, state);
  if (!ctx) return;

  std::vector<uint8_t> send_buffer(buffer_size, 0xCD);
  std::vector<uint8_t> recv_buffer(buffer_size);
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_buffer.data(), buffer_size);
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer.data(), buffer_size);

  for (auto _ : state) {
    iree_async_socket_send_operation_t send_op;
    memset(&send_op, 0, sizeof(send_op));
    send_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
    send_op.base.completion_fn = LoopbackContext::SendCallback;
    send_op.base.user_data = ctx;
    send_op.socket = ctx->client;
    send_op.buffers.values = &send_span;
    send_op.buffers.count = 1;

    iree_async_socket_recv_operation_t recv_op;
    memset(&recv_op, 0, sizeof(recv_op));
    recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
    recv_op.base.completion_fn = LoopbackContext::RecvCallback;
    recv_op.base.user_data = ctx;
    recv_op.socket = ctx->server;
    recv_op.buffers.values = &recv_span;
    recv_op.buffers.count = 1;

    // Submit both, wait for both.
    ctx->Reset();
    iree_async_proactor_submit_one(ctx->proactor, &send_op.base);
    iree_async_proactor_submit_one(ctx->proactor, &recv_op.base);

    // Wait for recv to complete.
    int completions = 0;
    while (completions < 2) {
      iree_host_size_t count = 0;
      iree_status_t status = iree_async_proactor_poll(
          ctx->proactor, iree_make_timeout_ms(1000), &count);
      iree_status_ignore(status);
      completions += (int)count;
    }
  }

  state.SetBytesProcessed(state.iterations() * buffer_size);
  DestroyLoopbackContext(ctx);
}

// Accept rate: how fast can we accept new connections?
static void BM_AcceptRate(::benchmark::State& state,
                          const ProactorFactory& factory) {
  // Create proactor and listener only.
  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    return;
  }
  iree_async_proactor_t* proactor = result.value();

  iree_async_socket_t* listener = nullptr;
  iree_status_t status =
      iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
                               IREE_ASYNC_SOCKET_OPTION_REUSE_ADDR, &listener);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Listener creation failed");
    iree_status_ignore(status);
    iree_async_proactor_release(proactor);
    return;
  }

  iree_async_address_t bind_addr;
  iree_async_address_from_ipv4(IREE_SV("127.0.0.1"), 0, &bind_addr);
  status = iree_async_socket_bind(listener, &bind_addr);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(listener, 128);
  }
  iree_async_address_t listen_addr;
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_query_local_address(listener, &listen_addr);
  }
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Bind/listen failed");
    iree_status_ignore(status);
    iree_async_socket_release(listener);
    iree_async_proactor_release(proactor);
    return;
  }

  LoopbackContext ctx_data;
  ctx_data.proactor = proactor;

  for (auto _ : state) {
    // Create client, connect, accept, close both.
    iree_async_socket_t* client = nullptr;
    status = iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_TCP,
                                      IREE_ASYNC_SOCKET_OPTION_NONE, &client);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    iree_async_socket_accept_operation_t accept_op;
    memset(&accept_op, 0, sizeof(accept_op));
    accept_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT;
    accept_op.base.completion_fn = LoopbackContext::Callback;
    accept_op.base.user_data = &ctx_data;
    accept_op.listen_socket = listener;

    iree_async_socket_connect_operation_t connect_op;
    memset(&connect_op, 0, sizeof(connect_op));
    connect_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT;
    connect_op.base.completion_fn = LoopbackContext::Callback;
    connect_op.base.user_data = &ctx_data;
    connect_op.socket = client;
    connect_op.address = listen_addr;

    ctx_data.Reset();
    iree_async_proactor_submit_one(proactor, &accept_op.base);
    iree_async_proactor_submit_one(proactor, &connect_op.base);

    // Wait for both.
    int completions = 0;
    while (completions < 2) {
      iree_host_size_t count = 0;
      status = iree_async_proactor_poll(proactor, iree_make_timeout_ms(1000),
                                        &count);
      iree_status_ignore(status);
      completions += (int)count;
    }

    // Clean up.
    iree_async_socket_release(accept_op.accepted_socket);
    iree_async_socket_release(client);
  }

  state.SetItemsProcessed(state.iterations());
  iree_async_socket_release(listener);
  iree_async_proactor_release(proactor);
}

// Zero-copy throughput comparison: measures Copy vs AdHoc vs Fixed ZC modes.
// This helps quantify the benefit of registered buffers over ad-hoc page
// pinning.
static void BM_ThroughputZC(::benchmark::State& state,
                            const ProactorFactory& factory, size_t buffer_size,
                            ZeroCopyMode mode) {
  // Create client socket with ZERO_COPY option for ZC modes.
  iree_async_socket_options_t client_options = IREE_ASYNC_SOCKET_OPTION_NONE;
  if (mode == ZeroCopyMode::kAdHoc || mode == ZeroCopyMode::kFixed) {
    client_options = IREE_ASYNC_SOCKET_OPTION_ZERO_COPY;
  }
  auto* ctx = CreateLoopbackContext(factory, state, client_options);
  if (!ctx) return;

  // Resources for Fixed mode (registered buffers).
  iree_async_slab_t* slab = nullptr;
  iree_async_region_t* region = nullptr;
  iree_async_buffer_pool_t* pool = nullptr;
  iree_async_buffer_lease_t lease = {};
  bool using_fixed = false;

  // Heap buffer for Copy and AdHoc modes.
  std::vector<uint8_t> heap_buffer;

  iree_async_span_t send_span;

  if (mode == ZeroCopyMode::kFixed) {
    // Setup registered buffers for Fixed mode.
    iree_async_slab_options_t slab_options = {};
    slab_options.buffer_size = buffer_size;
    // Use 16 buffers (power-of-2 for io_uring).
    slab_options.buffer_count = 16;

    iree_status_t status =
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Slab allocation failed");
      iree_status_ignore(status);
      DestroyLoopbackContext(ctx);
      return;
    }

    status = iree_async_proactor_register_slab(
        ctx->proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region);
    if (iree_status_is_failed_precondition(status)) {
      // Backend doesn't support additional READ regions.
      state.SkipWithError("Cannot register READ region on this backend");
      iree_status_ignore(status);
      iree_async_slab_release(slab);
      DestroyLoopbackContext(ctx);
      return;
    }
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Slab registration failed");
      iree_status_ignore(status);
      iree_async_slab_release(slab);
      DestroyLoopbackContext(ctx);
      return;
    }

    status =
        iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Pool allocation failed");
      iree_status_ignore(status);
      iree_async_region_release(region);
      iree_async_slab_release(slab);
      DestroyLoopbackContext(ctx);
      return;
    }

    // Acquire one buffer for the benchmark.
    status = iree_async_buffer_pool_acquire(pool, &lease);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Buffer acquire failed");
      iree_status_ignore(status);
      iree_async_buffer_pool_free(pool);
      iree_async_region_release(region);
      iree_async_slab_release(slab);
      DestroyLoopbackContext(ctx);
      return;
    }

    send_span = lease.span;
    using_fixed = true;
  } else {
    // Heap buffer for Copy and AdHoc modes.
    heap_buffer.resize(buffer_size, 0xEF);
    send_span = iree_async_span_from_ptr(heap_buffer.data(), buffer_size);
  }

  // Fill send buffer with pattern.
  memset(iree_async_span_ptr(send_span), 0xEF, buffer_size);

  // Recv buffer.
  std::vector<uint8_t> recv_buffer(buffer_size);
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer.data(), buffer_size);

  // Reset ZC tracking for this benchmark run.
  ctx->ResetZCTracking();

  for (auto _ : state) {
    iree_async_socket_send_operation_t send_op;
    memset(&send_op, 0, sizeof(send_op));
    send_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
    send_op.base.completion_fn = LoopbackContext::SendCallback;
    send_op.base.user_data = ctx;
    send_op.socket = ctx->client;
    send_op.buffers.values = &send_span;
    send_op.buffers.count = 1;
    send_op.send_flags = IREE_ASYNC_SOCKET_SEND_FLAG_NONE;

    iree_async_socket_recv_operation_t recv_op;
    memset(&recv_op, 0, sizeof(recv_op));
    recv_op.base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
    recv_op.base.completion_fn = LoopbackContext::RecvCallback;
    recv_op.base.user_data = ctx;
    recv_op.socket = ctx->server;
    recv_op.buffers.values = &recv_span;
    recv_op.buffers.count = 1;

    // Submit both, wait for both.
    ctx->Reset();
    iree_async_proactor_submit_one(ctx->proactor, &send_op.base);
    iree_async_proactor_submit_one(ctx->proactor, &recv_op.base);

    // Wait for both completions.
    int completions = 0;
    while (completions < 2) {
      iree_host_size_t count = 0;
      iree_status_t status = iree_async_proactor_poll(
          ctx->proactor, iree_make_timeout_ms(1000), &count);
      iree_status_ignore(status);
      completions += (int)count;
    }
  }

  state.SetBytesProcessed(state.iterations() * buffer_size);

  // Report ZC achievement rate for modes that request zero-copy.
  if (mode != ZeroCopyMode::kCopy) {
    double zc_rate = state.iterations() > 0
                         ? 100.0 * ctx->zc_achieved_count / state.iterations()
                         : 0.0;
    state.counters["ZC_Achieved_%"] =
        ::benchmark::Counter(zc_rate, ::benchmark::Counter::kDefaults);
  }

  // Cleanup.
  if (using_fixed) {
    iree_async_buffer_lease_release(&lease);
    iree_async_buffer_pool_free(pool);
    iree_async_region_release(region);
    iree_async_slab_release(slab);
  }
  DestroyLoopbackContext(ctx);
}

//===----------------------------------------------------------------------===//
// Benchmark suite class
//===----------------------------------------------------------------------===//

class SocketBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Roundtrip latency at various message sizes.
    for (size_t size : {64, 256, 1024, 4096, 16384, 65536}) {
      std::string name =
          std::string(prefix) + "/SocketRoundtrip/" + std::to_string(size);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, size](::benchmark::State& state) {
            BM_Roundtrip(state, factory, size);
          })
          ->Unit(::benchmark::kMicrosecond);
    }

    // Throughput at various buffer sizes.
    for (size_t size : {4096, 16384, 65536, 262144}) {
      std::string name =
          std::string(prefix) + "/SocketThroughput/" + std::to_string(size);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, size](::benchmark::State& state) {
            BM_Throughput(state, factory, size);
          })
          ->Unit(::benchmark::kMicrosecond);
    }

    // Accept rate.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/SocketAcceptRate").c_str(),
        [factory](::benchmark::State& state) { BM_AcceptRate(state, factory); })
        ->Unit(::benchmark::kMicrosecond);

    // Zero-copy throughput comparison at 64KB (good size for ZC benefit).
    // Compares Copy vs AdHoc (page pinning) vs Fixed (registered buffers).
    for (ZeroCopyMode mode :
         {ZeroCopyMode::kCopy, ZeroCopyMode::kAdHoc, ZeroCopyMode::kFixed}) {
      std::string name = std::string(prefix) + "/SocketZCThroughput/" +
                         ZeroCopyModeName(mode) + "/65536";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, mode](::benchmark::State& state) {
            BM_ThroughputZC(state, factory, 65536, mode);
          })
          ->Unit(::benchmark::kMicrosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(SocketBenchmarks);

}  // namespace iree::async::cts
