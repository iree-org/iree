// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS benchmarks for dispatch scalability.
//
// Measures the cost of dispatching a single ready fd among varying numbers of
// idle handlers in the proactor. The key property being validated is O(1)
// dispatch regardless of how many handlers are registered: if the proactor uses
// a hash table for fd-to-handler lookup (rather than linear scan), dispatch
// time should be constant regardless of N.
//
// Benchmarks:
//   - DispatchAmongIdleHandlers: N idle event sources, 1 active relay channel.
//     Measures dispatch time as a function of idle handler count.
//   - RelayDispatchScalability: N notification-relay channels, signal one.
//     Measures per-relay dispatch time with N total relays registered.
//   - NotificationRelayFanOut: 1 source notification, N relay sinks.
//     Measures fan-out throughput (O(N) by necessity, should be tight).

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/cts/util/benchmark_base.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/event.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/relay.h"
#include "iree/base/api.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Polls the proactor until |sink|'s epoch advances past |observed_epoch|.
// Returns false on timeout or poll error.
static bool PollUntilEpochAdvances(iree_async_proactor_t* proactor,
                                   iree_async_notification_t* sink,
                                   uint32_t observed_epoch) {
  iree_time_t deadline = iree_time_now() + 5000 * 1000000LL;
  while (iree_async_notification_query_epoch(sink) == observed_epoch) {
    if (iree_time_now() >= deadline) return false;
    iree_status_t status =
        iree_async_proactor_poll(proactor, iree_make_timeout_ms(100), nullptr);
    if (!iree_status_is_ok(status) &&
        !iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
      return false;
    }
    iree_status_ignore(status);
  }
  return true;
}

// No-op callback for idle event sources. These sources are never signaled, so
// this callback should never fire during normal benchmark operation.
static void IdleEventSourceCallback(void* /*user_data*/,
                                    iree_async_event_source_t* /*source*/,
                                    iree_async_poll_events_t /*events*/) {}

//===----------------------------------------------------------------------===//
// BM_DispatchAmongIdleHandlers
//===----------------------------------------------------------------------===//

// Registers N event sources (using events that are never signaled) as idle
// entries in the proactor's fd_map, plus one active notification relay. Each
// benchmark iteration signals the active source and polls until the active
// sink fires. The N idle event source fds in the fd_map must not affect the
// dispatch time of the active notification fd.
//
// With O(n) linear scan: dispatch time grows linearly with N.
// With O(1) hash lookup: dispatch time is constant regardless of N.
struct IdleHandlerContext {
  iree_async_proactor_t* proactor = nullptr;
  std::vector<iree_async_event_t*> idle_events;
  std::vector<iree_async_event_source_t*> idle_sources;
  iree_async_notification_t* active_source = nullptr;
  iree_async_notification_t* active_sink = nullptr;
  iree_async_relay_t* active_relay = nullptr;
};

static void DestroyIdleHandlerContext(IdleHandlerContext* ctx);

static IdleHandlerContext* CreateIdleHandlerContext(
    const ProactorFactory& factory, size_t idle_count,
    ::benchmark::State& state) {
  auto* ctx = new IdleHandlerContext();

  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create N idle event sources. Each event creates a pollable fd (eventfd on
  // Linux, pipe on macOS). The fd is registered in the proactor's fd_map as an
  // EVENT_SOURCE handler. Since the event is never signaled, the fd is never
  // readable and the callback never fires.
  ctx->idle_events.resize(idle_count, nullptr);
  ctx->idle_sources.resize(idle_count, nullptr);
  for (size_t i = 0; i < idle_count; ++i) {
    iree_status_t status =
        iree_async_event_create(ctx->proactor, &ctx->idle_events[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Event creation failed");
      iree_status_ignore(status);
      DestroyIdleHandlerContext(ctx);
      return nullptr;
    }

    iree_async_event_source_callback_t callback;
    callback.fn = IdleEventSourceCallback;
    callback.user_data = nullptr;
    status = iree_async_proactor_register_event_source(
        ctx->proactor, ctx->idle_events[i]->primitive, callback,
        &ctx->idle_sources[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Event source registration failed");
      iree_status_ignore(status);
      DestroyIdleHandlerContext(ctx);
      return nullptr;
    }
  }

  // Create active relay channel: notification -> notification.
  iree_status_t status = iree_async_notification_create(
      ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->active_source);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Active source notification creation failed");
    iree_status_ignore(status);
    DestroyIdleHandlerContext(ctx);
    return nullptr;
  }

  status = iree_async_notification_create(
      ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->active_sink);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Active sink notification creation failed");
    iree_status_ignore(status);
    DestroyIdleHandlerContext(ctx);
    return nullptr;
  }

  iree_async_relay_source_t source_desc =
      iree_async_relay_source_from_notification(ctx->active_source);
  iree_async_relay_sink_t sink_desc =
      iree_async_relay_sink_signal_notification(ctx->active_sink, INT32_MAX);
  status = iree_async_proactor_register_relay(
      ctx->proactor, source_desc, sink_desc, IREE_ASYNC_RELAY_FLAG_PERSISTENT,
      iree_async_relay_error_callback_none(), &ctx->active_relay);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Active relay registration failed");
    iree_status_ignore(status);
    DestroyIdleHandlerContext(ctx);
    return nullptr;
  }

  // Initial poll to ensure the relay is armed.
  status = iree_async_proactor_poll(ctx->proactor, iree_make_timeout_ms(10),
                                    nullptr);
  iree_status_ignore(status);

  return ctx;
}

static void DestroyIdleHandlerContext(IdleHandlerContext* ctx) {
  if (!ctx) return;
  if (ctx->active_relay) {
    iree_async_proactor_unregister_relay(ctx->proactor, ctx->active_relay);
  }
  iree_async_notification_release(ctx->active_sink);
  iree_async_notification_release(ctx->active_source);
  for (size_t i = 0; i < ctx->idle_sources.size(); ++i) {
    if (ctx->idle_sources[i]) {
      iree_async_proactor_unregister_event_source(ctx->proactor,
                                                  ctx->idle_sources[i]);
    }
    iree_async_event_release(ctx->idle_events[i]);
  }
  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

static void BM_DispatchAmongIdleHandlers(::benchmark::State& state,
                                         const ProactorFactory& factory) {
  size_t idle_count = static_cast<size_t>(state.range(0));
  auto* ctx = CreateIdleHandlerContext(factory, idle_count, state);
  if (!ctx) return;

  for (auto _ : state) {
    uint32_t observed = iree_async_notification_query_epoch(ctx->active_sink);
    auto start = std::chrono::high_resolution_clock::now();
    iree_async_notification_signal(ctx->active_source, 1);

    if (!PollUntilEpochAdvances(ctx->proactor, ctx->active_sink, observed)) {
      state.SkipWithError("Active relay dispatch timed out");
      DestroyIdleHandlerContext(ctx);
      return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  DestroyIdleHandlerContext(ctx);
}

//===----------------------------------------------------------------------===//
// BM_RelayDispatchScalability
//===----------------------------------------------------------------------===//

// N notification-source relays, each with a unique source notification and all
// pointing to the same sink notification. One specific relay is signaled per
// iteration. The N-1 other notification fds in the fd_map must not slow down
// dispatch.
//
// With O(n) linear scan: dispatch time grows linearly with N.
// With O(1) hash lookup: dispatch time is constant regardless of N.
struct RelayScalabilityContext {
  iree_async_proactor_t* proactor = nullptr;
  std::vector<iree_async_notification_t*> sources;
  std::vector<iree_async_relay_t*> relays;
  iree_async_notification_t* sink = nullptr;
};

static void DestroyRelayScalabilityContext(RelayScalabilityContext* ctx);

static RelayScalabilityContext* CreateRelayScalabilityContext(
    const ProactorFactory& factory, size_t relay_count,
    ::benchmark::State& state) {
  auto* ctx = new RelayScalabilityContext();

  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create shared sink notification.
  iree_status_t status = iree_async_notification_create(
      ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->sink);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Sink notification creation failed");
    iree_status_ignore(status);
    iree_async_proactor_release(ctx->proactor);
    ctx->proactor = nullptr;
    delete ctx;
    return nullptr;
  }

  // Create N source notifications with relays to the shared sink.
  ctx->sources.resize(relay_count, nullptr);
  ctx->relays.resize(relay_count, nullptr);
  for (size_t i = 0; i < relay_count; ++i) {
    status = iree_async_notification_create(
        ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->sources[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Source notification creation failed");
      iree_status_ignore(status);
      DestroyRelayScalabilityContext(ctx);
      return nullptr;
    }

    iree_async_relay_source_t source_desc =
        iree_async_relay_source_from_notification(ctx->sources[i]);
    iree_async_relay_sink_t sink_desc =
        iree_async_relay_sink_signal_notification(ctx->sink, INT32_MAX);
    status = iree_async_proactor_register_relay(
        ctx->proactor, source_desc, sink_desc, IREE_ASYNC_RELAY_FLAG_PERSISTENT,
        iree_async_relay_error_callback_none(), &ctx->relays[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Relay registration failed");
      iree_status_ignore(status);
      DestroyRelayScalabilityContext(ctx);
      return nullptr;
    }
  }

  // Initial poll to arm relays.
  status = iree_async_proactor_poll(ctx->proactor, iree_make_timeout_ms(10),
                                    nullptr);
  iree_status_ignore(status);

  return ctx;
}

static void DestroyRelayScalabilityContext(RelayScalabilityContext* ctx) {
  if (!ctx) return;
  for (size_t i = 0; i < ctx->relays.size(); ++i) {
    if (ctx->relays[i]) {
      iree_async_proactor_unregister_relay(ctx->proactor, ctx->relays[i]);
    }
  }
  for (size_t i = 0; i < ctx->sources.size(); ++i) {
    iree_async_notification_release(ctx->sources[i]);
  }
  iree_async_notification_release(ctx->sink);
  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

static void BM_RelayDispatchScalability(::benchmark::State& state,
                                        const ProactorFactory& factory) {
  size_t relay_count = static_cast<size_t>(state.range(0));
  auto* ctx = CreateRelayScalabilityContext(factory, relay_count, state);
  if (!ctx) return;

  // Signal the last source each iteration. With linear scan, this would be the
  // worst case (scanning all N-1 other entries first). With hash lookup, the
  // position is irrelevant.
  size_t target_index = relay_count - 1;

  for (auto _ : state) {
    uint32_t observed = iree_async_notification_query_epoch(ctx->sink);
    auto start = std::chrono::high_resolution_clock::now();
    iree_async_notification_signal(ctx->sources[target_index], 1);

    if (!PollUntilEpochAdvances(ctx->proactor, ctx->sink, observed)) {
      state.SkipWithError("Relay dispatch timed out");
      DestroyRelayScalabilityContext(ctx);
      return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  DestroyRelayScalabilityContext(ctx);
}

//===----------------------------------------------------------------------===//
// BM_NotificationRelayFanOut
//===----------------------------------------------------------------------===//

// One source notification fans out to N sink notifications via N relays.
// This measures the per-notification relay_list walking performance: the poll
// loop finds the source notification's fd via O(1) hash lookup, then walks the
// relay_list of N entries to fire all sinks.
//
// Time should scale linearly with N (fan-out is inherently O(N)), but the
// constant factor should be small: one pointer chase and one sink signal per
// relay.
struct FanOutContext {
  iree_async_proactor_t* proactor = nullptr;
  iree_async_notification_t* source = nullptr;
  std::vector<iree_async_notification_t*> sinks;
  std::vector<iree_async_relay_t*> relays;
};

static void DestroyFanOutContext(FanOutContext* ctx);

static FanOutContext* CreateFanOutContext(const ProactorFactory& factory,
                                          size_t fan_out,
                                          ::benchmark::State& state) {
  auto* ctx = new FanOutContext();

  auto result = factory();
  if (!result.ok()) {
    state.SkipWithError("Proactor creation failed");
    delete ctx;
    return nullptr;
  }
  ctx->proactor = result.value();

  // Create source notification.
  iree_status_t status = iree_async_notification_create(
      ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->source);
  if (!iree_status_is_ok(status)) {
    state.SkipWithError("Source notification creation failed");
    iree_status_ignore(status);
    iree_async_proactor_release(ctx->proactor);
    ctx->proactor = nullptr;
    delete ctx;
    return nullptr;
  }

  // Create N sink notifications and relays from the shared source.
  ctx->sinks.resize(fan_out, nullptr);
  ctx->relays.resize(fan_out, nullptr);
  for (size_t i = 0; i < fan_out; ++i) {
    status = iree_async_notification_create(
        ctx->proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &ctx->sinks[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Sink notification creation failed");
      iree_status_ignore(status);
      DestroyFanOutContext(ctx);
      return nullptr;
    }

    iree_async_relay_source_t source_desc =
        iree_async_relay_source_from_notification(ctx->source);
    iree_async_relay_sink_t sink_desc =
        iree_async_relay_sink_signal_notification(ctx->sinks[i], INT32_MAX);
    status = iree_async_proactor_register_relay(
        ctx->proactor, source_desc, sink_desc, IREE_ASYNC_RELAY_FLAG_PERSISTENT,
        iree_async_relay_error_callback_none(), &ctx->relays[i]);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Relay registration failed");
      iree_status_ignore(status);
      DestroyFanOutContext(ctx);
      return nullptr;
    }
  }

  // Initial poll to arm relays.
  status = iree_async_proactor_poll(ctx->proactor, iree_make_timeout_ms(10),
                                    nullptr);
  iree_status_ignore(status);

  return ctx;
}

static void DestroyFanOutContext(FanOutContext* ctx) {
  if (!ctx) return;
  for (size_t i = 0; i < ctx->relays.size(); ++i) {
    if (ctx->relays[i]) {
      iree_async_proactor_unregister_relay(ctx->proactor, ctx->relays[i]);
    }
  }
  for (size_t i = 0; i < ctx->sinks.size(); ++i) {
    iree_async_notification_release(ctx->sinks[i]);
  }
  iree_async_notification_release(ctx->source);
  iree_async_proactor_release(ctx->proactor);
  delete ctx;
}

static void BM_NotificationRelayFanOut(::benchmark::State& state,
                                       const ProactorFactory& factory) {
  size_t fan_out = static_cast<size_t>(state.range(0));
  auto* ctx = CreateFanOutContext(factory, fan_out, state);
  if (!ctx) return;

  // Pre-allocate observed epochs to avoid per-iteration allocation.
  std::vector<uint32_t> observed(fan_out);

  for (auto _ : state) {
    // Snapshot all sink epochs.
    for (size_t i = 0; i < fan_out; ++i) {
      observed[i] = iree_async_notification_query_epoch(ctx->sinks[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    iree_async_notification_signal(ctx->source, 1);

    // Poll until all sink epochs advance.
    iree_time_t deadline = iree_time_now() + 5000 * 1000000LL;
    bool all_fired = false;
    while (!all_fired) {
      if (iree_time_now() >= deadline) {
        state.SkipWithError("Fan-out relay dispatch timed out");
        DestroyFanOutContext(ctx);
        return;
      }
      iree_status_t poll_status = iree_async_proactor_poll(
          ctx->proactor, iree_make_timeout_ms(100), nullptr);
      if (!iree_status_is_ok(poll_status) &&
          !iree_status_is_deadline_exceeded(poll_status)) {
        iree_status_ignore(poll_status);
        state.SkipWithError("Poll failed during fan-out benchmark");
        DestroyFanOutContext(ctx);
        return;
      }
      iree_status_ignore(poll_status);

      all_fired = true;
      for (size_t i = 0; i < fan_out; ++i) {
        if (iree_async_notification_query_epoch(ctx->sinks[i]) == observed[i]) {
          all_fired = false;
          break;
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.SetItemsProcessed(state.iterations() * fan_out);
  DestroyFanOutContext(ctx);
}

//===----------------------------------------------------------------------===//
// Benchmark suite registration
//===----------------------------------------------------------------------===//

class DispatchScalabilityBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const ProactorFactory& factory) {
    // Dispatch among idle handlers: measures O(1) fd_map lookup.
    // Time should be constant regardless of N idle handlers.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/DispatchAmongIdleHandlers").c_str(),
        [factory](::benchmark::State& state) {
          BM_DispatchAmongIdleHandlers(state, factory);
        })
        ->Range(1, 4096)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();

    // Relay dispatch scalability: measures O(1) dispatch among N channels.
    // Time should be constant regardless of N registered relays.
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/RelayDispatchScalability").c_str(),
        [factory](::benchmark::State& state) {
          BM_RelayDispatchScalability(state, factory);
        })
        ->Range(1, 4096)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();

    // Notification relay fan-out: measures per-notification relay_list walk.
    // Time should scale linearly with N (inherent in fan-out).
    ::benchmark::RegisterBenchmark(
        (std::string(prefix) + "/NotificationRelayFanOut").c_str(),
        [factory](::benchmark::State& state) {
          BM_NotificationRelayFanOut(state, factory);
        })
        ->Range(1, 256)
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();
  }
};

CTS_REGISTER_BENCHMARK_SUITE(DispatchScalabilityBenchmarks);

}  // namespace iree::async::cts
