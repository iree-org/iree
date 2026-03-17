// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Carrier throughput and IOPS benchmark.
//
// Measures sustained unidirectional throughput by keeping the send pipeline
// full and polling for completions only when backpressured. Reports both
// bytes/sec (throughput) and items/sec (IOPS). At small message sizes the
// per-operation overhead dominates (IOPS benchmark); at large sizes the
// data movement rate dominates (throughput benchmark).

#include <atomic>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/net/carrier/cts/util/benchmark_base.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {
namespace {

// Recv handler that tracks total bytes received. Needed because TCP can
// coalesce multiple sends into fewer recv callbacks — tracking message count
// alone would hang during the drain phase.
struct ThroughputRecvState {
  std::atomic<int64_t> bytes_received{0};

  static iree_status_t Handler(void* user_data, iree_async_span_t data,
                               iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<ThroughputRecvState*>(user_data);
    self->bytes_received.fetch_add(data.length, std::memory_order_release);
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsHandler() { return {Handler, this}; }
};

// Polls the proactor with a short timeout. Returns false if a fatal error
// occurs (not deadline exceeded, which is expected).
static bool PollOnce(iree_async_proactor_t* proactor, iree_timeout_t timeout) {
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

// Sustained unidirectional throughput: client sends, server receives.
static void BM_Throughput(::benchmark::State& state,
                          const CarrierPairFactory& factory) {
  int64_t message_size = state.range(0);

  auto* ctx = CreateBenchmarkContext(factory, state);
  if (!ctx) return;
  if (!RequireCapability(ctx, IREE_NET_CARRIER_CAPABILITY_RELIABLE, state)) {
    DestroyBenchmarkContext(ctx);
    return;
  }

  // Recv handler on server tracks bytes for correct drain.
  ThroughputRecvState recv_state;
  iree_net_carrier_set_recv_handler(ctx->server, recv_state.AsHandler());
  iree_net_carrier_set_recv_handler(ctx->client, MakeNullRecvHandler());

  // Activate both.
  if (!iree_status_is_ok(iree_net_carrier_activate(ctx->client)) ||
      !iree_status_is_ok(iree_net_carrier_activate(ctx->server))) {
    state.SkipWithError("Activation failed");
    iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
    DestroyBenchmarkContext(ctx);
    return;
  }

  // Prepare send buffer.
  std::vector<uint8_t> buffer(message_size, 0xAB);
  iree_async_span_t span =
      iree_async_span_from_ptr(buffer.data(), buffer.size());
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  params.user_data = 0;

  bool send_error = false;
  for (auto _ : state) {
    // If send budget is exhausted, poll until a slot is available. This
    // naturally paces the sender to the receiver's consumption rate once the
    // pipeline fills.
    while (iree_net_carrier_query_send_budget(ctx->client).slots == 0) {
      if (!PollOnce(ctx->proactor, iree_make_timeout_ms(100))) {
        state.SkipWithError("Poll failed during backpressure");
        send_error = true;
        break;
      }
    }
    if (send_error) break;

    // Submit send.
    iree_status_t status = iree_net_carrier_send(ctx->client, &params);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Send failed");
      iree_status_ignore(status);
      break;
    }

    // Non-blocking poll to keep the pipeline moving. Without this, completions
    // only get processed when we hit backpressure above, which would cause
    // bursty behavior instead of smooth pipelining.
    PollOnce(ctx->proactor, iree_immediate_timeout());
  }

  // Drain remaining recv completions. PauseTiming excludes this from the
  // measured throughput — we only want to measure the steady-state pipeline.
  state.PauseTiming();
  {
    int64_t expected_bytes = state.iterations() * message_size;
    iree_time_t deadline = iree_time_now() + kDefaultPollBudget;
    while (recv_state.bytes_received.load(std::memory_order_acquire) <
           expected_bytes) {
      if (iree_time_now() >= deadline) {
        state.SkipWithError("Drain timeout");
        break;
      }
      PollOnce(ctx->proactor, iree_make_timeout_ms(100));
    }
  }
  state.ResumeTiming();

  state.SetBytesProcessed(state.iterations() * message_size);
  state.SetItemsProcessed(state.iterations());

  // Replace recv handler before destroy to prevent callbacks into local state.
  iree_net_carrier_set_recv_handler(ctx->server, MakeNullRecvHandler());
  DestroyBenchmarkContext(ctx);
}

}  // namespace

// Benchmark suite class for CTS registration.
class ThroughputBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const CarrierPairFactory& factory) {
    for (int64_t size : {64, 1024, 4096, 65536}) {
      ::benchmark::RegisterBenchmark(
          (std::string(prefix) + "/Throughput").c_str(),
          [factory](::benchmark::State& state) {
            BM_Throughput(state, factory);
          })
          ->Args({size})
          ->Unit(::benchmark::kMicrosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE_WITH_TAGS(ThroughputBenchmarks, {"reliable"}, {});

}  // namespace iree::net::carrier::cts
