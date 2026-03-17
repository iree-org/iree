// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Carrier CTS fuzz entry point: operation-sequence fuzzer.
//
// Interprets fuzz input as a sequence of operations on a carrier pair,
// exercising the full carrier API surface: send/recv, backpressure, error
// injection, lease retention, and deactivation. Runs with ASAN for memory
// safety and with TSan (concurrent poll thread mode) for thread safety.
//
// Link-time composition: the fuzz binary links this file + one backend's
// backends.cc to create a backend-specific fuzz target.
//
// Input format:
//   Byte 0: Configuration flags.
//     bits 0-1: auto-poll interval (every 1, 2, 4, or 8 operations).
//     bit 2:    concurrent poll thread (TSan mode).
//     bits 3-7: reserved.
//   Bytes 1+: Operation stream (each byte is one operation).
//     High nibble (bits 7-4): opcode.
//     Low nibble (bits 3-0):  parameter (size class for sends).
//
// Opcodes:
//   0x0: SEND_TO_SERVER       - Send from client. Low nibble = size class.
//   0x1: SEND_TO_CLIENT       - Send from server. Low nibble = size class.
//   0x2: POLL_SHORT           - Poll with 10ms timeout.
//   0x3: POLL_LONG            - Poll with 100ms timeout.
//   0x4: QUERY_BUDGET_CLIENT  - Query client send budget.
//   0x5: QUERY_BUDGET_SERVER  - Query server send budget.
//   0x6: INJECT_ERROR_CLIENT  - Next client recv returns INTERNAL.
//   0x7: INJECT_ERROR_SERVER  - Next server recv returns INTERNAL.
//   0x8: RETAIN_LEASE_CLIENT  - Client handler retains next buffer lease.
//   0x9: RETAIN_LEASE_SERVER  - Server handler retains next buffer lease.
//   0xA: RELEASE_CLIENT       - Release all retained client leases.
//   0xB: RELEASE_SERVER       - Release all retained server leases.
//   0xC: DEACTIVATE_CLIENT    - Deactivate client (one-shot).
//   0xD: DEACTIVATE_SERVER    - Deactivate server (one-shot).
//   0xE: POLL_SHORT           - (alias)
//   0xF: POLL_SHORT           - (alias)
//
// Size classes (low nibble for send opcodes):
//   0:    0 bytes (empty send, should be rejected).
//   1-15: min(2^(class-1), remaining_input_bytes).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/fuzz_base.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace {

using namespace iree::net::carrier::cts;

//===----------------------------------------------------------------------===//
// Global state (initialized once in LLVMFuzzerInitialize)
//===----------------------------------------------------------------------===//

static iree_async_proactor_t* g_proactor = nullptr;
static CarrierPairFactory g_factory;

//===----------------------------------------------------------------------===//
// Opcodes
//===----------------------------------------------------------------------===//

enum FuzzOpcode : uint8_t {
  kSendToServer = 0x0,
  kSendToClient = 0x1,
  kPollShort = 0x2,
  kPollLong = 0x3,
  kQueryBudgetClient = 0x4,
  kQueryBudgetServer = 0x5,
  kInjectErrorClient = 0x6,
  kInjectErrorServer = 0x7,
  kRetainLeaseClient = 0x8,
  kRetainLeaseServer = 0x9,
  kReleaseClient = 0xA,
  kReleaseServer = 0xB,
  kDeactivateClient = 0xC,
  kDeactivateServer = 0xD,
  kPollAlias1 = 0xE,
  kPollAlias2 = 0xF,
};

//===----------------------------------------------------------------------===//
// Send helper
//===----------------------------------------------------------------------===//

// Attempts to send data on a carrier. Ignores all errors (the fuzzer exercises
// error paths, not correctness assertions).
static void FuzzSend(iree_net_carrier_t* carrier, const uint8_t* data,
                     size_t size) {
  if (size == 0) {
    // Empty send: construct params with zero-length span list.
    iree_net_send_params_t params = {};
    iree_status_ignore(iree_net_carrier_send(carrier, &params));
    return;
  }

  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<uint8_t*>(data), size);
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  iree_status_ignore(iree_net_carrier_send(carrier, &params));
}

//===----------------------------------------------------------------------===//
// Poll helpers
//===----------------------------------------------------------------------===//

static void PollOnce(iree_async_proactor_t* proactor,
                     iree_duration_t timeout_ms) {
  iree_host_size_t completed = 0;
  iree_status_t status = iree_async_proactor_poll(
      proactor, iree_make_timeout_ms(timeout_ms), &completed);
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Fuzz scenario
//===----------------------------------------------------------------------===//

static void RunFuzzScenario(CarrierPair& pair, const uint8_t* data,
                            size_t size) {
  if (size < 2) return;

  // Decode config byte.
  uint8_t config = data[0];
  const uint8_t* operations = data + 1;
  size_t operations_size = size - 1;

  int auto_poll_interval = 1 << (config & 0x03);  // 1, 2, 4, or 8.
  bool concurrent = (config & 0x04) != 0;

  // Set up fuzz handlers.
  FuzzRecvHandler client_handler;
  FuzzRecvHandler server_handler;

  iree_net_carrier_set_recv_handler(pair.client, client_handler.AsHandler());
  iree_net_carrier_set_recv_handler(pair.server, server_handler.AsHandler());

  // Activate both carriers. If activation fails, bail early.
  iree_status_t status = iree_net_carrier_activate(pair.client);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }
  status = iree_net_carrier_activate(pair.server);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    // Deactivate the client before bailing.
    DeactivateAndDrain(pair.client, pair.proactor);
    return;
  }

  bool client_deactivated = false;
  bool server_deactivated = false;

  // Deactivation completion flags. Must outlive the operation loop since the
  // deactivation callback writes to these asynchronously.
  std::atomic<bool> client_deactivate_done{false};
  std::atomic<bool> server_deactivate_done{false};

  // Start concurrent poll thread for TSan mode.
  PollThread poll_thread;
  if (concurrent) {
    poll_thread.Start(pair.proactor);
  }

  // Process operation stream.
  // The operations buffer serves double duty: each byte encodes an operation,
  // and SEND operations consume subsequent bytes as payload data. This lets the
  // fuzzer's coverage guidance learn optimal mixes of operations and payloads.
  size_t cursor = 0;
  int operation_count = 0;

  while (cursor < operations_size) {
    uint8_t operation_byte = operations[cursor++];
    uint8_t opcode = (operation_byte >> 4) & 0x0F;
    uint8_t param = operation_byte & 0x0F;

    switch (opcode) {
      case kSendToServer: {
        size_t send_size = DecodeSendSize(param, operations_size - cursor);
        FuzzSend(pair.client, operations + cursor, send_size);
        cursor += send_size;
        break;
      }
      case kSendToClient: {
        size_t send_size = DecodeSendSize(param, operations_size - cursor);
        FuzzSend(pair.server, operations + cursor, send_size);
        cursor += send_size;
        break;
      }

      case kPollShort:
      case kPollAlias1:
      case kPollAlias2:
        if (!concurrent) PollOnce(pair.proactor, 10);
        break;
      case kPollLong:
        if (!concurrent) PollOnce(pair.proactor, 100);
        break;

      case kQueryBudgetClient:
        iree_net_carrier_query_send_budget(pair.client);
        break;
      case kQueryBudgetServer:
        iree_net_carrier_query_send_budget(pair.server);
        break;

      case kInjectErrorClient:
        client_handler.inject_error.store(true, std::memory_order_release);
        break;
      case kInjectErrorServer:
        server_handler.inject_error.store(true, std::memory_order_release);
        break;

      case kRetainLeaseClient:
        client_handler.retain_next_lease.store(true, std::memory_order_release);
        break;
      case kRetainLeaseServer:
        server_handler.retain_next_lease.store(true, std::memory_order_release);
        break;

      case kReleaseClient:
        client_handler.ReleaseAll();
        break;
      case kReleaseServer:
        server_handler.ReleaseAll();
        break;

      case kDeactivateClient:
        if (!client_deactivated) {
          // Non-blocking deactivation: callback writes to the outer-scope
          // atomic (not a block-local variable) to avoid use-after-scope.
          auto callback = [](void* user_data) {
            static_cast<std::atomic<bool>*>(user_data)->store(
                true, std::memory_order_release);
          };
          iree_status_t deactivate_status = iree_net_carrier_deactivate(
              pair.client, callback, &client_deactivate_done);
          iree_status_ignore(deactivate_status);
          client_deactivated = true;
          // Give the deactivation a chance to complete.
          if (!concurrent) PollOnce(pair.proactor, 10);
        }
        break;

      case kDeactivateServer:
        if (!server_deactivated) {
          auto callback = [](void* user_data) {
            static_cast<std::atomic<bool>*>(user_data)->store(
                true, std::memory_order_release);
          };
          iree_status_t deactivate_status = iree_net_carrier_deactivate(
              pair.server, callback, &server_deactivate_done);
          iree_status_ignore(deactivate_status);
          server_deactivated = true;
          if (!concurrent) PollOnce(pair.proactor, 10);
        }
        break;
    }

    // Auto-poll: drive completions periodically to prevent total starvation.
    ++operation_count;
    if (!concurrent && (operation_count % auto_poll_interval == 0)) {
      PollOnce(pair.proactor, 1);
    }
  }

  // ===== Cleanup: reverse order of setup =====

  // 1. Stop concurrent poll thread before touching handler state.
  if (concurrent) {
    poll_thread.Stop();
  }

  // 2. Release any retained leases (must happen before carrier teardown).
  client_handler.ReleaseAll();
  server_handler.ReleaseAll();

  // 3. Replace handlers with safe null handlers to prevent callbacks into
  //    the FuzzRecvHandler objects during drain (they're stack-allocated and
  //    will be destroyed when this function returns).
  iree_net_carrier_set_recv_handler(pair.client, MakeNullRecvHandler());
  iree_net_carrier_set_recv_handler(pair.server, MakeNullRecvHandler());

  // 4. Deactivate and drain both carriers.
  DeactivateAndDrain(pair.client, pair.proactor);
  DeactivateAndDrain(pair.server, pair.proactor);
}

}  // namespace

//===----------------------------------------------------------------------===//
// libFuzzer entry points
//===----------------------------------------------------------------------===//

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  auto* config = iree::net::carrier::cts::CtsRegistry::GetBackend();
  if (!config) {
    fprintf(stderr, "carrier_fuzz: no backend registered\n");
    return 1;
  }
  g_factory = config->info.factory;

  // Create a proactor once and reuse across all fuzz iterations.
  // The proactor is the most expensive resource to create (io_uring ring
  // setup, kernel memory mapping), so amortizing it across iterations gives
  // a significant throughput improvement.
  iree_status_t status =
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &g_proactor);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "carrier_fuzz: failed to create proactor\n");
    iree_status_ignore(status);
    return 1;
  }

  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2 || !g_proactor) return 0;

  // Create a fresh carrier pair on the reused proactor.
  auto result = g_factory(g_proactor);
  if (!result.ok()) return 0;
  iree::net::carrier::cts::CarrierPair pair = std::move(result).value();

  // Run the operation-sequence scenario.
  RunFuzzScenario(pair, data, size);

  // Release carriers (destroy).
  iree_net_carrier_release(pair.client);
  iree_net_carrier_release(pair.server);

  // Run pair cleanup (pools, slabs, listener, etc).
  // Does NOT release the proactor — it's reused across iterations.
  if (pair.cleanup) {
    pair.cleanup(pair.context);
  }

  return 0;
}
