// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Carrier latency benchmark: cold vs warm delivery latency.
//
// Measures one-way message delivery latency (send → recv handler invocation)
// in two operating modes by timestamping inside the recv handler itself:
//
//   Cold: Carrier is in notification-driven sleep mode. Each message delivery
//     traverses the kernel notification path: send() signals the peer's
//     notification → proactor wakes from epoll_wait/kqueue/IOCP →
//     notification dispatches drain callback → recv handler fires. Between
//     iterations, 300 immediate-timeout polls drain any adaptive polling state
//     back to notification mode. Expected SHM latency: ~1-5µs.
//
//   Warm: Carrier is in adaptive poll mode. A warmup burst triggers the
//     sleep-to-poll transition. The proactor's progress callback detects new
//     ring data via acquire-load — no kernel notification. The recv handler
//     fires before the kernel poll syscall (epoll_wait/kqueue/IOCP), so the
//     measured latency excludes that overhead entirely. Expected SHM latency:
//     ~100-500ns.
//
// The key timing technique: the recv handler records Clock::now() at the
// instant it fires, and the benchmark reads that timestamp after PollOnce
// returns. This separates the actual delivery latency from the trailing kernel
// poll cost that PollOnce always incurs.
//
// For carriers without adaptive polling, both modes report similar numbers.
//
// Contrast with the roundtrip benchmark: roundtrip measures time around the
// full PollOnce call (including the kernel poll). This benchmark isolates the
// delivery path itself, making the adaptive polling benefit directly visible.

#include <chrono>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/net/carrier/cts/util/benchmark_base.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {
namespace {

using Clock = std::chrono::high_resolution_clock;

// Number of immediate-timeout polls between cold iterations. Must exceed any
// carrier's idle-to-sleep threshold so adaptive carriers transition back to
// notification mode. SHM uses IREE_NET_SHM_IDLE_SPIN_THRESHOLD (256); 300
// provides margin without excessive overhead (~15µs at ~50ns per poll).
static constexpr int kColdDrainPolls = 300;

// Number of warmup messages sent before warm latency measurement begins.
// Must exceed any carrier's poll-mode-threshold (SHM default = 1). 16 messages
// provide robust margin for any threshold configuration.
static constexpr int kWarmupMessages = 16;

// Recv handler that timestamps each delivery. The recv_time field captures
// Clock::now() at the instant the handler fires inside the proactor's dispatch
// path — before the trailing kernel poll syscall. This is the correct point
// for latency measurement.
struct LatencyRecvState {
  Clock::time_point recv_time{};
  std::atomic<int> recv_count{0};

  static iree_status_t Handler(void* user_data, iree_async_span_t data,
                                iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<LatencyRecvState*>(user_data);
    self->recv_time = Clock::now();
    self->recv_count.fetch_add(1, std::memory_order_release);
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsHandler() { return {Handler, this}; }
};

// Polls the proactor once. Returns false on fatal error.
static bool PollOnce(iree_async_proactor_t* proactor,
                     iree_timeout_t timeout) {
  iree_host_size_t completed = 0;
  iree_status_t status =
      iree_async_proactor_poll(proactor, timeout, &completed);
  if (iree_status_is_ok(status) || iree_status_is_deadline_exceeded(status)) {
    iree_status_ignore(status);
    return true;
  }
  iree_status_ignore(status);
  return false;
}

// Cold latency: one-way delivery through the notification/sleep path.
//
// Between iterations, polls the proactor with immediate timeout to drain any
// adaptive polling state back to notification mode. The measurement captures
// the notification-driven delivery path: send() signals the peer's notification
// → proactor dispatches the notification (inside epoll_wait processing) →
// drain callback fires → recv handler fires (timestamp captured here).
static void BM_ColdLatency(::benchmark::State& state,
                            const CarrierPairFactory& factory) {
  int64_t message_size = state.range(0);

  auto* ctx = CreateBenchmarkContext(factory, state);
  if (!ctx) return;
  if (!RequireCapability(ctx, IREE_NET_CARRIER_CAPABILITY_RELIABLE, state)) {
    DestroyBenchmarkContext(ctx);
    return;
  }

  // Use the timestamping recv handler instead of BenchmarkContext's counter.
  LatencyRecvState recv_state;
  iree_net_carrier_set_recv_handler(ctx->server, recv_state.AsHandler());
  iree_net_carrier_set_recv_handler(ctx->client, MakeNullRecvHandler());

  if (!iree_status_is_ok(iree_net_carrier_activate(ctx->client)) ||
      !iree_status_is_ok(iree_net_carrier_activate(ctx->server))) {
    state.SkipWithError("Activation failed");
    iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
    DestroyBenchmarkContext(ctx);
    return;
  }

  std::vector<uint8_t> buffer(message_size, 0xAB);
  iree_async_span_t span =
      iree_async_span_from_ptr(buffer.data(), buffer.size());
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  params.user_data = 0;

  int expected_recv = 0;
  for (auto _ : state) {
    // Drain any adaptive polling state back to notification mode.
    // For SHM carriers: 300 empty polls exceed the 256-iteration idle
    // threshold, forcing the progress callback to transition back to sleep
    // mode and re-post a NOTIFICATION_WAIT.
    // For non-adaptive carriers: 300 immediate-timeout polls are no-ops.
    for (int i = 0; i < kColdDrainPolls; ++i) {
      PollOnce(ctx->proactor, iree_immediate_timeout());
    }

    ++expected_recv;
    auto send_time = Clock::now();

    iree_status_t status = iree_net_carrier_send(ctx->client, &params);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Send failed");
      iree_status_ignore(status);
      break;
    }

    // Spin-poll until the recv handler fires (timestamps inside handler).
    while (recv_state.recv_count.load(std::memory_order_acquire) <
           expected_recv) {
      if (!PollOnce(ctx->proactor, iree_immediate_timeout())) {
        state.SkipWithError("Poll failed");
        goto done;
      }
    }

    // Report send → recv handler timestamp, excluding trailing kernel poll.
    state.SetIterationTime(
        std::chrono::duration<double>(recv_state.recv_time - send_time)
            .count());
  }

done:
  iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
  DestroyBenchmarkContext(ctx);
}

// Warm latency: one-way delivery through the adaptive poll path.
//
// A warmup burst of messages triggers the sleep-to-poll transition. During
// measurement, the proactor's progress callback detects new ring data via
// acquire-load before the kernel poll syscall — the recv handler fires and
// timestamps this early delivery point.
static void BM_WarmLatency(::benchmark::State& state,
                            const CarrierPairFactory& factory) {
  int64_t message_size = state.range(0);

  auto* ctx = CreateBenchmarkContext(factory, state);
  if (!ctx) return;
  if (!RequireCapability(ctx, IREE_NET_CARRIER_CAPABILITY_RELIABLE, state)) {
    DestroyBenchmarkContext(ctx);
    return;
  }

  LatencyRecvState recv_state;
  iree_net_carrier_set_recv_handler(ctx->server, recv_state.AsHandler());
  iree_net_carrier_set_recv_handler(ctx->client, MakeNullRecvHandler());

  if (!iree_status_is_ok(iree_net_carrier_activate(ctx->client)) ||
      !iree_status_is_ok(iree_net_carrier_activate(ctx->server))) {
    state.SkipWithError("Activation failed");
    iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
    DestroyBenchmarkContext(ctx);
    return;
  }

  std::vector<uint8_t> buffer(message_size, 0xAB);
  iree_async_span_t span =
      iree_async_span_from_ptr(buffer.data(), buffer.size());
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  params.user_data = 0;

  // Warmup: send enough messages to trigger adaptive polling.
  // Sequential send-then-drain keeps the ring from filling while ensuring
  // the carrier processes at least one message per drain callback (enough
  // to exceed the default poll_mode_threshold of 1).
  int expected_recv = 0;
  for (int i = 0; i < kWarmupMessages; ++i) {
    ++expected_recv;
    iree_status_t status = iree_net_carrier_send(ctx->client, &params);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Warmup send failed");
      iree_status_ignore(status);
      iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
      DestroyBenchmarkContext(ctx);
      return;
    }
    while (recv_state.recv_count.load(std::memory_order_acquire) <
           expected_recv) {
      PollOnce(ctx->proactor, iree_immediate_timeout());
    }
  }

  // Measurement: carrier is now in poll mode (for adaptive carriers).
  // Each iteration sends one message and captures the recv handler timestamp.
  // The carrier stays warm because iterations are rapid — idle_spin_count
  // never approaches the sleep threshold.
  for (auto _ : state) {
    ++expected_recv;
    auto send_time = Clock::now();

    iree_status_t status = iree_net_carrier_send(ctx->client, &params);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Send failed");
      iree_status_ignore(status);
      break;
    }

    while (recv_state.recv_count.load(std::memory_order_acquire) <
           expected_recv) {
      if (!PollOnce(ctx->proactor, iree_immediate_timeout())) {
        state.SkipWithError("Poll failed");
        goto done;
      }
    }

    // Report send → recv handler timestamp, excluding trailing kernel poll.
    state.SetIterationTime(
        std::chrono::duration<double>(recv_state.recv_time - send_time)
            .count());
  }

done:
  iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
  DestroyBenchmarkContext(ctx);
}

}  // namespace

// Benchmark suite class for CTS registration.
class LatencyBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const CarrierPairFactory& factory) {
    for (int64_t size : {64, 1024, 4096, 65536}) {
      ::benchmark::RegisterBenchmark(
          (std::string(prefix) + "/Latency/Cold").c_str(),
          [factory](::benchmark::State& state) {
            BM_ColdLatency(state, factory);
          })
          ->Args({size})
          ->UseManualTime()
          ->Unit(::benchmark::kNanosecond);

      ::benchmark::RegisterBenchmark(
          (std::string(prefix) + "/Latency/Warm").c_str(),
          [factory](::benchmark::State& state) {
            BM_WarmLatency(state, factory);
          })
          ->Args({size})
          ->UseManualTime()
          ->Unit(::benchmark::kNanosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE_WITH_TAGS(LatencyBenchmarks, {"reliable"}, {});

}  // namespace iree::net::carrier::cts
