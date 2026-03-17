// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Carrier roundtrip latency benchmark.
//
// Measures the time for a single send->recv->completion cycle at various
// message sizes. This is the baseline latency metric for carrier
// implementations — the minimum time to push data from sender to receiver
// and confirm delivery.

#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/net/carrier/cts/util/benchmark_base.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {

// Single message roundtrip: client sends, server receives.
static void BM_Roundtrip(::benchmark::State& state,
                         const CarrierPairFactory& factory) {
  auto* ctx = CreateBenchmarkContext(factory, state);
  if (!ctx) return;

  // Set up recv handler on server.
  iree_net_carrier_recv_handler_t server_handler = {
      BenchmarkContext::RecvHandler, ctx};
  iree_net_carrier_set_recv_handler(ctx->server, server_handler);

  // Null handler on client (we're not receiving on client).
  iree_net_carrier_set_recv_handler(ctx->client, MakeNullRecvHandler());

  // Activate both.
  if (!iree_status_is_ok(iree_net_carrier_activate(ctx->client)) ||
      !iree_status_is_ok(iree_net_carrier_activate(ctx->server))) {
    state.SkipWithError("Activation failed");
    DestroyBenchmarkContext(ctx);
    return;
  }

  // Prepare send data sized from the benchmark parameter.
  int64_t message_size = state.range(0);
  std::vector<uint8_t> buffer(message_size, 0xAB);
  iree_async_span_t span =
      iree_async_span_from_ptr(buffer.data(), buffer.size());

  for (auto _ : state) {
    ctx->Reset();

    auto start = std::chrono::high_resolution_clock::now();

    // Send from client.
    iree_net_send_params_t params = {};
    params.data.values = &span;
    params.data.count = 1;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    params.user_data = 0;

    iree_status_t status = iree_net_carrier_send(ctx->client, &params);
    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Send failed");
      iree_status_ignore(status);
      break;
    }

    // Wait for recv completion.
    if (!ctx->SpinPollUntilComplete(0, 1)) {
      state.SkipWithError("Poll timeout");
      break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  DestroyBenchmarkContext(ctx);
}

// Benchmark suite class for CTS registration.
class RoundtripBenchmarks {
 public:
  static void RegisterBenchmarks(const char* prefix,
                                 const CarrierPairFactory& factory) {
    for (int64_t size : {64, 1024, 4096, 65536}) {
      ::benchmark::RegisterBenchmark(
          (std::string(prefix) + "/Roundtrip").c_str(),
          [factory](::benchmark::State& state) {
            BM_Roundtrip(state, factory);
          })
          ->Args({size})
          ->UseManualTime()
          ->Unit(::benchmark::kNanosecond);
    }
  }
};

CTS_REGISTER_BENCHMARK_SUITE(RoundtripBenchmarks);

}  // namespace iree::net::carrier::cts
