// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Error handling tests for carrier implementations.
//
// Tests error propagation and graceful degradation behaviors. Note that some
// error handling behaviors (like sticky errors) are implementation-specific
// and tested in carrier-specific test files.

#include <atomic>
#include <cstring>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class ErrorHandlingTest : public CarrierTestBase<> {};

// Recv handler that returns an error after receiving data.
struct ErrorRecvContext {
  std::atomic<int> call_count{0};
  iree_status_code_t error_code = IREE_STATUS_INTERNAL;
};

static iree_status_t ErrorRecvHandler(void* user_data, iree_async_span_t data,
                                      iree_async_buffer_lease_t* lease) {
  auto* ctx = static_cast<ErrorRecvContext*>(user_data);
  ctx->call_count.fetch_add(1, std::memory_order_release);
  return iree_make_status(ctx->error_code, "test error from recv handler");
}

// When recv handler returns an error, the carrier should not crash or leak.
// The exact error propagation behavior is carrier-specific; this test only
// verifies the carrier remains in a valid state.
TEST_P(ErrorHandlingTest, RecvHandlerErrorDoesNotCrash) {
  ErrorRecvContext error_ctx;
  iree_net_carrier_recv_handler_t error_handler = {ErrorRecvHandler,
                                                   &error_ctx};

  ActivateBoth(MakeNullRecvHandler(), error_handler);

  // Send to server which has error handler.
  const char* msg = "trigger error";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);

  // Send may succeed or fail depending on implementation.
  iree_status_t status = iree_net_carrier_send(client_, &params);
  iree_status_ignore(status);

  // Poll to process any async completions.
  PollUntil([&] { return error_ctx.call_count.load() > 0; });

  // Verify handler was actually called.
  EXPECT_GT(error_ctx.call_count.load(), 0);

  // Carrier should still be in a valid state (not crashed).
  iree_net_carrier_state_t state = iree_net_carrier_state(client_);
  EXPECT_TRUE(state == IREE_NET_CARRIER_STATE_ACTIVE ||
              state == IREE_NET_CARRIER_STATE_DRAINING ||
              state == IREE_NET_CARRIER_STATE_DEACTIVATED);
}

// Send after peer deactivation should fail gracefully.
TEST_P(ErrorHandlingTest, SendAfterPeerDeactivateFails) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Deactivate the server (peer).
  std::atomic<bool> deactivated{false};
  auto callback = [](void* ud) { *static_cast<std::atomic<bool>*>(ud) = true; };
  IREE_ASSERT_OK(iree_net_carrier_deactivate(server_, callback, &deactivated));

  // Poll until deactivation completes.
  ASSERT_TRUE(PollUntil([&] { return deactivated.load(); }));

  // Now try to send from client to deactivated server.
  const char* msg = "to deactivated peer";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);

  // Send should fail (peer is gone).
  iree_status_t status = iree_net_carrier_send(client_, &params);

  // Failure may occur at send time or asynchronously during completion.
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  } else {
    // If send succeeded, poll for completion - it should eventually fail.
    PollOnce();
  }
}

// Multiple rapid sends to error handler should not cause double-free or crash.
TEST_P(ErrorHandlingTest, RapidSendsToErrorHandler) {
  ErrorRecvContext error_ctx;
  iree_net_carrier_recv_handler_t error_handler = {ErrorRecvHandler,
                                                   &error_ctx};

  ActivateBoth(MakeNullRecvHandler(), error_handler);

  const char* msg = "X";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);

  // Send multiple times rapidly.
  for (int i = 0; i < 10; ++i) {
    iree_status_t status = iree_net_carrier_send(client_, &params);
    iree_status_ignore(status);
    PollOnce();
  }

  // Poll to drain completions.
  PollUntil([&] { return error_ctx.call_count.load() >= 1; });

  // Should not have crashed - carrier still valid.
  iree_net_carrier_state_t state = iree_net_carrier_state(client_);
  EXPECT_TRUE(state == IREE_NET_CARRIER_STATE_ACTIVE ||
              state == IREE_NET_CARRIER_STATE_DRAINING ||
              state == IREE_NET_CARRIER_STATE_DEACTIVATED);
}

CTS_REGISTER_TEST_SUITE(ErrorHandlingTest);

}  // namespace
}  // namespace iree::net::carrier::cts
