// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base infrastructure for carrier benchmarks.
//
// Provides BenchmarkContext for managing carrier pair lifecycle and
// completion tracking during benchmarks.

#ifndef IREE_NET_CARRIER_CTS_UTIL_BENCHMARK_BASE_H_
#define IREE_NET_CARRIER_CTS_UTIL_BENCHMARK_BASE_H_

#include <atomic>
#include <chrono>

#include "benchmark/benchmark.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {

// Default timeout for poll operations (5 seconds).
static constexpr iree_duration_t kDefaultPollBudget = 5000 * 1000000LL;

// Base context for carrier benchmarks.
// Manages carrier pair lifetime and provides completion tracking.
struct BenchmarkContext {
  CarrierPair pair = {};
  iree_net_carrier_t* client = nullptr;
  iree_net_carrier_t* server = nullptr;
  iree_async_proactor_t* proactor = nullptr;
  iree_net_carrier_capabilities_t capabilities = 0;

  std::atomic<int> send_completions{0};
  std::atomic<int> recv_completions{0};
  iree_status_code_t last_status_code = IREE_STATUS_OK;

  // Send completion callback.
  static void SendCallback(void* callback_user_data,
                           uint64_t operation_user_data, iree_status_t status,
                           iree_host_size_t bytes_transferred,
                           iree_async_buffer_lease_t* recv_lease) {
    auto* ctx = static_cast<BenchmarkContext*>(callback_user_data);
    ctx->last_status_code = iree_status_code(status);
    ctx->send_completions.fetch_add(1, std::memory_order_release);
    iree_status_ignore(status);
  }

  // Recv handler that counts received messages.
  static iree_status_t RecvHandler(void* user_data, iree_async_span_t data,
                                   iree_async_buffer_lease_t* lease) {
    auto* ctx = static_cast<BenchmarkContext*>(user_data);
    ctx->recv_completions.fetch_add(1, std::memory_order_release);
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  void Reset() {
    send_completions.store(0, std::memory_order_release);
    recv_completions.store(0, std::memory_order_release);
    last_status_code = IREE_STATUS_OK;
  }

  // Spin poll until expected completions received.
  bool SpinPollUntilComplete(int expected_sends, int expected_recvs,
                             iree_duration_t budget_ns = kDefaultPollBudget) {
    iree_time_t deadline = iree_time_now() + budget_ns;
    while (send_completions.load(std::memory_order_acquire) < expected_sends ||
           recv_completions.load(std::memory_order_acquire) < expected_recvs) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t count = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor, iree_immediate_timeout(), &count);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        return false;
      }
      iree_status_ignore(status);
    }
    return last_status_code == IREE_STATUS_OK;
  }
};

// Creates a benchmark context from the factory. Returns nullptr on failure.
inline BenchmarkContext* CreateBenchmarkContext(
    const CarrierPairFactory& factory, ::benchmark::State& state) {
  auto* ctx = new BenchmarkContext();

  auto result = factory(/*proactor=*/nullptr);
  if (!result.ok()) {
    if (result.status().code() == iree::StatusCode::kUnavailable) {
      state.SkipWithError("Backend unavailable on this system");
    } else {
      state.SkipWithError("Carrier pair creation failed");
    }
    delete ctx;
    return nullptr;
  }
  ctx->pair = result.value();
  ctx->client = ctx->pair.client;
  ctx->server = ctx->pair.server;
  ctx->proactor = ctx->pair.proactor;
  ctx->capabilities = iree_net_carrier_capabilities(ctx->client);

  return ctx;
}

// Destroys a benchmark context. Handles deactivation and cleanup.
inline void DestroyBenchmarkContext(BenchmarkContext* ctx) {
  if (!ctx) return;

  // Replace recv handlers with null handlers before draining to prevent
  // callbacks into the benchmark context during drain.
  if (ctx->client) {
    iree_net_carrier_set_recv_handler(ctx->client, MakeNullRecvHandler());
  }
  if (ctx->server) {
    iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
  }

  // Deactivate carriers and wait for async cleanup to complete.
  if (ctx->client) {
    DeactivateAndDrain(ctx->client, ctx->proactor);
    iree_net_carrier_release(ctx->client);
  }
  if (ctx->server) {
    DeactivateAndDrain(ctx->server, ctx->proactor);
    iree_net_carrier_release(ctx->server);
  }
  if (ctx->pair.cleanup) {
    ctx->pair.cleanup(ctx->pair.context);
  }
  if (ctx->proactor) {
    iree_async_proactor_release(ctx->proactor);
  }
  delete ctx;
}

// Skips benchmark if capability is missing.
inline bool RequireCapability(BenchmarkContext* ctx,
                              iree_net_carrier_capabilities_t required,
                              ::benchmark::State& state) {
  if (!(ctx->capabilities & required)) {
    state.SkipWithError("Backend lacks required capability");
    return false;
  }
  return true;
}

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_BENCHMARK_BASE_H_
