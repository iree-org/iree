// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for carrier CTS tests.
//
// Provides the test fixture and helper utilities shared by all carrier tests.
// Test logic lives in separate .cc files (lifecycle_test.cc, send_recv_test.cc,
// etc.) which use CTS_REGISTER_TEST_SUITE() for self-registration.
//
// Key design:
//   - Fresh carrier pair per test via SetUp/TearDown (no state leakage)
//   - Capability-gated GTEST_SKIP instead of external EXCLUDED_TESTS lists
//   - Time-budgeted polling prevents slow-CI flakiness
//   - Link-time composition: test suites + backends linked together

#ifndef IREE_NET_CARRIER_CTS_UTIL_TEST_BASE_H_
#define IREE_NET_CARRIER_CTS_UTIL_TEST_BASE_H_

#include <atomic>
#include <functional>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Recv capture utilities
//===----------------------------------------------------------------------===//

// Captures received data for test verification.
// Thread-safe for use as recv handler user_data.
struct RecvCapture {
  std::vector<uint8_t>* buffer;
  std::atomic<iree_host_size_t> total_bytes{0};

  explicit RecvCapture(std::vector<uint8_t>* out_buffer) : buffer(out_buffer) {}

  static iree_status_t Handler(void* user_data, iree_async_span_t data,
                               iree_async_buffer_lease_t* lease) {
    auto* capture = static_cast<RecvCapture*>(user_data);
    uint8_t* ptr = iree_async_span_ptr(data);
    capture->buffer->insert(capture->buffer->end(), ptr, ptr + data.length);
    capture->total_bytes.fetch_add(data.length, std::memory_order_relaxed);
    // Release lease to return buffer to pool for reuse.
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsHandler() { return {Handler, this}; }
};

// NullRecvHandler and MakeNullRecvHandler are in registry.h (shared utilities).

//===----------------------------------------------------------------------===//
// Completion tracking
//===----------------------------------------------------------------------===//

// Tracks send completion callbacks for test assertions.
struct SendCompletionTracker {
  std::atomic<int> call_count{0};
  std::atomic<iree_host_size_t> total_bytes{0};
  iree_status_t last_status = iree_ok_status();

  SendCompletionTracker() = default;
  ~SendCompletionTracker() { iree_status_ignore(last_status); }

  // Non-copyable (status ownership).
  SendCompletionTracker(const SendCompletionTracker&) = delete;
  SendCompletionTracker& operator=(const SendCompletionTracker&) = delete;

  // Returns and transfers ownership of the last status for testing.
  iree_status_t ConsumeStatus() {
    iree_status_t result = last_status;
    last_status = iree_ok_status();
    return result;
  }

  void Reset() {
    last_status = iree_status_ignore(last_status);
    call_count = 0;
    total_bytes = 0;
  }

  static void Callback(void* callback_user_data, uint64_t operation_user_data,
                       iree_status_t status, iree_host_size_t bytes_transferred,
                       iree_async_buffer_lease_t* recv_lease) {
    auto* tracker = static_cast<SendCompletionTracker*>(callback_user_data);
    tracker->call_count.fetch_add(1, std::memory_order_relaxed);
    tracker->total_bytes.fetch_add(bytes_transferred,
                                   std::memory_order_relaxed);
    tracker->last_status = iree_status_ignore(tracker->last_status);
    tracker->last_status = status;  // Take ownership.
  }

  iree_net_carrier_callback_t AsCallback() { return {Callback, this}; }
};

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

// Base class for all carrier CTS tests. Parameterized on BackendInfo.
// Creates a fresh carrier pair in SetUp(), releases in TearDown().
template <typename BaseType = ::testing::TestWithParam<BackendInfo>>
class CarrierTestBase : public BaseType {
 protected:
  void SetUp() override {
    BackendInfo backend = this->GetParam();
    auto result = backend.factory(/*proactor=*/nullptr);
    if (!result.ok() &&
        result.status().code() == iree::StatusCode::kUnavailable) {
      GTEST_SKIP() << "Backend '" << backend.name
                   << "' unavailable on this system";
    }
    IREE_ASSERT_OK_AND_ASSIGN(pair_, std::move(result));

    client_ = pair_.client;
    server_ = pair_.server;
    proactor_ = pair_.proactor;
    capabilities_ = iree_net_carrier_capabilities(client_);
  }

  void TearDown() override {
    // Replace recv handlers with null handlers before draining. This prevents
    // any in-flight async receives from invoking the test's handlers, which
    // may point to stack-allocated objects that went out of scope when the
    // test body exited.
    if (client_) {
      iree_net_carrier_set_recv_handler(client_, MakeNullRecvHandler());
    }
    if (server_) {
      iree_net_carrier_set_recv_handler(server_, MakeNullRecvHandler());
    }

    // Deactivate and drain both carriers before release.
    if (client_) {
      DeactivateAndDrain(client_, proactor_);
      iree_net_carrier_release(client_);
      client_ = nullptr;
    }
    if (server_) {
      DeactivateAndDrain(server_, proactor_);
      iree_net_carrier_release(server_);
      server_ = nullptr;
    }

    // Run pair cleanup if provided.
    if (pair_.cleanup) {
      pair_.cleanup(pair_.context);
      pair_.cleanup = nullptr;
      pair_.context = nullptr;
    }

    // Release the proactor last.
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  //===--------------------------------------------------------------------===//
  // Capability checking
  //===--------------------------------------------------------------------===//

  // Check if carrier has a specific capability.
  bool HasCapability(iree_net_carrier_capabilities_t cap) {
    return (capabilities_ & cap) != 0;
  }

  //===--------------------------------------------------------------------===//
  // Polling helpers
  //===--------------------------------------------------------------------===//

  // Poll until condition is true or budget expires. Returns true if condition
  // became true, false on timeout.
  bool PollUntil(std::function<bool()> condition,
                 iree_duration_t total_budget = iree_make_duration_ms(5000)) {
    iree_time_t deadline_ns = iree_time_now() + total_budget;
    iree_timeout_t timeout = iree_make_deadline(deadline_ns);
    while (!condition()) {
      if (iree_time_now() >= deadline_ns) return false;
      iree_host_size_t completed = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor_, timeout, &completed);
      if (!iree_status_is_deadline_exceeded(status)) {
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          return false;
        }
      } else {
        iree_status_ignore(status);
      }
    }
    return true;
  }

  // Poll for at least N completions within budget.
  void PollCompletions(
      iree_host_size_t min_completions,
      iree_duration_t total_budget = iree_make_duration_ms(5000)) {
    iree_host_size_t total = 0;
    iree_time_t deadline_ns = iree_time_now() + total_budget;
    iree_timeout_t timeout = iree_make_deadline(deadline_ns);
    while (total < min_completions) {
      if (iree_time_now() >= deadline_ns) break;
      iree_host_size_t completed = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor_, timeout, &completed);
      if (!iree_status_is_deadline_exceeded(status)) {
        IREE_ASSERT_OK(status);
      } else {
        iree_status_ignore(status);
      }
      total += completed;
    }
    ASSERT_GE(total, min_completions)
        << "Expected at least " << min_completions << " completions but got "
        << total << " within budget";
  }

  // Convenience: poll once with a short timeout.
  void PollOnce() {
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_make_timeout_ms(100), &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }

  //===--------------------------------------------------------------------===//
  // Activation helpers
  //===--------------------------------------------------------------------===//

  // Activates both carriers with the given recv handlers.
  void ActivateBoth(iree_net_carrier_recv_handler_t client_handler,
                    iree_net_carrier_recv_handler_t server_handler) {
    iree_net_carrier_set_recv_handler(client_, client_handler);
    iree_net_carrier_set_recv_handler(server_, server_handler);
    IREE_ASSERT_OK(iree_net_carrier_activate(client_));
    IREE_ASSERT_OK(iree_net_carrier_activate(server_));
  }

  // Activates both carriers with null handlers (discarding received data).
  void ActivateBothWithNullHandlers() {
    ActivateBoth(MakeNullRecvHandler(), MakeNullRecvHandler());
  }

  //===--------------------------------------------------------------------===//
  // Send helpers
  //===--------------------------------------------------------------------===//

  // Creates send params for a single contiguous buffer.
  static iree_net_send_params_t MakeSendParams(const void* data,
                                               iree_host_size_t size,
                                               iree_async_span_t* span_storage,
                                               uint64_t user_data = 0) {
    *span_storage = iree_async_span_from_ptr(const_cast<void*>(data), size);
    iree_net_send_params_t params = {};
    params.data.values = span_storage;
    params.data.count = 1;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    params.user_data = user_data;
    return params;
  }

  // Creates send params from a pre-populated span array (scatter-gather).
  // The |spans| array must outlive the returned params.
  static iree_net_send_params_t MakeSendParamsFromSpans(
      iree_async_span_t* spans, iree_host_size_t span_count,
      uint64_t user_data = 0) {
    iree_net_send_params_t params = {};
    params.data.values = spans;
    params.data.count = span_count;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    params.user_data = user_data;
    return params;
  }

 protected:
  CarrierPair pair_ = {};
  iree_net_carrier_t* client_ = nullptr;
  iree_net_carrier_t* server_ = nullptr;
  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_carrier_capabilities_t capabilities_ = 0;
};

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_TEST_BASE_H_
