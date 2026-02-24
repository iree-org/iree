// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for relay operations.
//
// Measures the performance of the relay mechanism - declarative source-to-sink
// event propagation. Relays trigger a sink (signal notification, write to fd)
// when a source (notification signaled, fd readable) is activated.
//
// Benchmarks:
//   - Throughput: Signals per second through a relay chain
//   - Latency: Time from source signal to sink signal
//   - Scalability: Performance with multiple concurrent relays

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/relay.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Relay benchmark context
//===----------------------------------------------------------------------===//

// Context for relay benchmarks.
//
// Architecture:
//   The benchmark thread owns all resources and calls poll(). Completion is
//   detected by querying the sink notification's epoch directly using
//   iree_async_notification_query_epoch(). This avoids race conditions that
//   would occur with a separate waiter thread.
//
// Flow per iteration:
//   1. Capture sink epoch (observed)
//   2. Signal source notification
//   3. Poll until sink epoch advances past observed
struct RelayContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_notification_t* source = nullptr;
  iree_async_notification_t* sink = nullptr;
  iree_async_relay_t* relay = nullptr;
};

static RelayContext* CreateRelayContext(const ProactorFactory& factory,
                                        ::benchmark::State& state) {
  auto* ctx = new RelayContext();

  // Create proactor.
  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create source and sink notifications.
  iree_status_t status = iree_async_notification_create(
      ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->source);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Source notification creation failed");
    iree_status_ignore(status);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  status = iree_async_notification_create(
      ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->sink);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Sink notification creation failed");
    iree_status_ignore(status);
    iree_async_notification_release(ctx->source);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Register relay: source notification -> sink notification.
  iree_async_relay_source_t source_desc =
      iree_async_relay_source_from_notification(ctx->source);
  iree_async_relay_sink_t sink_desc =
      iree_async_relay_sink_signal_notification(ctx->sink, INT32_MAX);
  status = iree_async_proactor_register_relay(
      ctx->proactor, source_desc, sink_desc, IREE_ASYNC_RELAY_FLAG_PERSISTENT,
      iree_async_relay_error_callback_none(), &ctx->relay);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Relay registration failed");
    iree_status_ignore(status);
    iree_async_notification_release(ctx->sink);
    iree_async_notification_release(ctx->source);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }

  // Initial poll to ensure the relay is armed before we start signaling.
  iree_host_size_t initial_completed = 0;
  status = iree_async_proactor_poll(ctx->proactor, iree_make_timeout_ms(10),
                                    &initial_completed);
  if (!iree_status_is_ok(status) && !iree_status_is_deadline_exceeded(status)) {
    state.SkipWithError("Initial poll failed");
    iree_status_ignore(status);
    iree_async_proactor_unregister_relay(ctx->proactor, ctx->relay);
    iree_async_notification_release(ctx->sink);
    iree_async_notification_release(ctx->source);
    iree_async_proactor_release(ctx->proactor);
    delete ctx;
    return nullptr;
  }
  iree_status_ignore(status);

  return ctx;
}

static void DestroyRelayContext(RelayContext* ctx) {
  if (!ctx) return;

  // Cleanup resources.
  iree_async_proactor_unregister_relay(ctx->proactor, ctx->relay);
  iree_async_notification_release(ctx->sink);
  iree_async_notification_release(ctx->source);
  iree_async_proactor_release(ctx->proactor);

  delete ctx;
}

//===----------------------------------------------------------------------===//
// Benchmark implementations
//===----------------------------------------------------------------------===//

// Benchmark relay throughput: signals per second through a notification relay.
//
// Measures the full round-trip:
//   signal(source) -> poll() fires relay -> signal(sink)
//
// This is the primary throughput metric for relay performance.
static void BM_Throughput(::benchmark::State& state,
                          const ProactorFactory& factory) {
  auto* ctx = CreateRelayContext(factory, state);
  if (!ctx) return;

  for (auto _ : state) {
    // Capture sink epoch before signaling.
    uint32_t observed = iree_async_notification_query_epoch(ctx->sink);
    iree_async_notification_signal(ctx->source, 1);

    // Poll until the sink notification epoch advances.
    while (iree_async_notification_query_epoch(ctx->sink) == observed) {
      iree_status_t status = iree_async_proactor_poll(
          ctx->proactor, iree_make_timeout_ms(100), nullptr);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        state.SkipWithError("Poll failed during benchmark");
        DestroyRelayContext(ctx);
        return;
      }
      iree_status_ignore(status);
    }
  }

  state.SetItemsProcessed(state.iterations());
  DestroyRelayContext(ctx);
}

// Benchmark relay latency with manual timing.
//
// Uses benchmark::State::SetIterationTime() to report the actual
// source-to-sink latency rather than the loop overhead.
static void BM_Latency(::benchmark::State& state,
                       const ProactorFactory& factory) {
  auto* ctx = CreateRelayContext(factory, state);
  if (!ctx) return;

  for (auto _ : state) {
    // Capture sink epoch before signaling.
    uint32_t observed = iree_async_notification_query_epoch(ctx->sink);
    auto start = std::chrono::high_resolution_clock::now();
    iree_async_notification_signal(ctx->source, 1);

    // Poll until the sink notification epoch advances.
    while (iree_async_notification_query_epoch(ctx->sink) == observed) {
      iree_status_t status = iree_async_proactor_poll(
          ctx->proactor, iree_make_timeout_ms(100), nullptr);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        state.SkipWithError("Poll failed during benchmark");
        DestroyRelayContext(ctx);
        return;
      }
      iree_status_ignore(status);
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  DestroyRelayContext(ctx);
}

//===----------------------------------------------------------------------===//
// Scalability benchmark context (multiple relay channels)
//===----------------------------------------------------------------------===//

struct RelayChannel {
  iree_async_notification_t* source = nullptr;
  iree_async_notification_t* sink = nullptr;
  iree_async_relay_t* relay = nullptr;
};

struct ScalabilityContext {
  iree_async_proactor_t* proactor = nullptr;
  std::vector<std::unique_ptr<RelayChannel>> channels;
};

// Creates a relay channel with source/sink notifications, relay, and waiter
// thread. Returns an error status on failure; the channel is fully initialized
// on success.
static StatusOr<std::unique_ptr<RelayChannel>> CreateRelayChannel(
    iree_async_proactor_t* proactor) {
  auto channel = std::make_unique<RelayChannel>();

  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &channel->source);
  if (!iree_status_is_ok(status)) {
    return iree::Status(static_cast<iree::StatusCode>(iree_status_code(status)),
                        "Source notification creation failed");
  }

  status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &channel->sink);
  if (!iree_status_is_ok(status)) {
    iree_async_notification_release(channel->source);
    channel->source = nullptr;
    return iree::Status(static_cast<iree::StatusCode>(iree_status_code(status)),
                        "Sink notification creation failed");
  }

  iree_async_relay_source_t source_desc =
      iree_async_relay_source_from_notification(channel->source);
  iree_async_relay_sink_t sink_desc =
      iree_async_relay_sink_signal_notification(channel->sink, INT32_MAX);
  status = iree_async_proactor_register_relay(
      proactor, source_desc, sink_desc, IREE_ASYNC_RELAY_FLAG_PERSISTENT,
      iree_async_relay_error_callback_none(), &channel->relay);
  if (!iree_status_is_ok(status)) {
    iree_async_notification_release(channel->sink);
    iree_async_notification_release(channel->source);
    channel->sink = nullptr;
    channel->source = nullptr;
    return iree::Status(static_cast<iree::StatusCode>(iree_status_code(status)),
                        "Relay registration failed");
  }

  return channel;
}

// Forward declaration for use in CreateScalabilityContext error path.
static void DestroyScalabilityContext(ScalabilityContext* ctx);

static ScalabilityContext* CreateScalabilityContext(
    const ProactorFactory& factory, size_t channel_count,
    ::benchmark::State& state) {
  auto* ctx = new ScalabilityContext();

  // Create proactor.
  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create channels.
  ctx->channels.reserve(channel_count);
  for (size_t i = 0; i < channel_count; ++i) {
    auto channel_result = CreateRelayChannel(ctx->proactor);
    if (!channel_result.ok()) {
      state.SkipWithError(channel_result.status().ToString().c_str());
      DestroyScalabilityContext(ctx);
      return nullptr;
    }
    ctx->channels.push_back(std::move(channel_result).value());
  }

  return ctx;
}

static void DestroyScalabilityContext(ScalabilityContext* ctx) {
  if (!ctx) return;

  // Cleanup resources.
  for (auto& channel : ctx->channels) {
    if (channel->relay) {
      iree_async_proactor_unregister_relay(ctx->proactor, channel->relay);
    }
    iree_async_notification_release(channel->sink);
    iree_async_notification_release(channel->source);
  }

  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

// Benchmark relay scalability: N concurrent relay channels.
//
// Each iteration signals all N sources and waits for all N sinks.
// This measures how performance scales with concurrent relay activity.
static void BM_Scalability(::benchmark::State& state,
                           const ProactorFactory& factory,
                           size_t channel_count) {
  auto* ctx = CreateScalabilityContext(factory, channel_count, state);
  if (!ctx) return;

  // Pre-allocate observed epochs array to avoid per-iteration allocation.
  std::vector<uint32_t> observed(channel_count);

  for (auto _ : state) {
    // Snapshot all sink epochs.
    for (size_t i = 0; i < channel_count; ++i) {
      observed[i] = iree_async_notification_query_epoch(ctx->channels[i]->sink);
    }

    // Signal all sources.
    for (size_t i = 0; i < channel_count; ++i) {
      iree_async_notification_signal(ctx->channels[i]->source, 1);
    }

    // Poll until all sink epochs advance.
    size_t completed = 0;
    while (completed < channel_count) {
      iree_status_t status = iree_async_proactor_poll(
          ctx->proactor, iree_make_timeout_ms(100), nullptr);
      if (!iree_status_is_ok(status) &&
          !iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
        state.SkipWithError("Poll failed during scalability benchmark");
        DestroyScalabilityContext(ctx);
        return;
      }
      iree_status_ignore(status);

      // Count completed channels.
      completed = 0;
      for (size_t i = 0; i < channel_count; ++i) {
        if (iree_async_notification_query_epoch(ctx->channels[i]->sink) !=
            observed[i]) {
          ++completed;
        }
      }
    }
  }

  state.SetItemsProcessed(state.iterations() * channel_count);
  DestroyScalabilityContext(ctx);
}

//===----------------------------------------------------------------------===//
// Benchmark suite class
//===----------------------------------------------------------------------===//

class RelayBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Throughput (signals per second).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/RelayThroughput").c_str(),
        [factory](::benchmark::State& state) { BM_Throughput(state, factory); })
        ->Unit(::benchmark::kMicrosecond);

    // Latency (source -> sink time).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/RelayLatency").c_str(),
        [factory](::benchmark::State& state) { BM_Latency(state, factory); })
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();

    // Scalability at various channel counts.
    for (size_t channels : {1, 4, 16, 64, 256, 1024}) {
      std::string name =
          std::string(prefix) + "/RelayScalability/" + std::to_string(channels);
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [factory, channels](::benchmark::State& state) {
            BM_Scalability(state, factory, channels);
          })
          ->Unit(::benchmark::kMicrosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(RelayBenchmarks);

}  // namespace iree::async::cts
