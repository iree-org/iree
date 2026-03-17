// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz infrastructure for carrier CTS.
//
// Provides the configurable recv handler, concurrent poll thread for TSan
// coverage, and cleanup utilities shared by all carrier fuzz targets.
//
// The fuzz handler supports:
//   - Error injection: next recv returns INTERNAL.
//   - Lease retention: holds buffer leases beyond the callback, releasing
//   later.
//   - Byte counting: tracks total received bytes for drain detection.
//
// The poll thread enables concurrent send/recv on separate threads, which is
// the production usage pattern and exercises thread safety in carrier
// internals.

#ifndef IREE_NET_CARRIER_CTS_UTIL_FUZZ_BASE_H_
#define IREE_NET_CARRIER_CTS_UTIL_FUZZ_BASE_H_

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Fuzz recv handler
//===----------------------------------------------------------------------===//

// Configurable recv handler for fuzz scenarios.
// Thread-safe: handler runs on the proactor thread, control flags are set from
// the main thread. The retained_leases vector is protected by a mutex.
struct FuzzRecvHandler {
  // Control flags (set from main thread, read from proactor thread).
  std::atomic<bool> inject_error{false};
  std::atomic<bool> retain_next_lease{false};

  // Retained buffer leases. Protected by mutex since the handler pushes from
  // the proactor thread and ReleaseAll is called from the main thread.
  std::mutex retained_mutex;
  std::vector<iree_async_buffer_lease_t> retained_leases;

  // Byte counter for drain detection.
  std::atomic<size_t> bytes_received{0};

  static iree_status_t Handler(void* user_data, iree_async_span_t data,
                               iree_async_buffer_lease_t* lease) {
    auto* self = static_cast<FuzzRecvHandler*>(user_data);
    self->bytes_received.fetch_add(data.length, std::memory_order_relaxed);

    if (self->retain_next_lease.exchange(false, std::memory_order_acq_rel) &&
        lease) {
      // Copy the lease value and take ownership of the release callback.
      iree_async_buffer_lease_t retained = *lease;
      // Clear the original so nobody else recycles this buffer.
      lease->release = iree_async_buffer_recycle_callback_null();
      {
        std::lock_guard<std::mutex> lock(self->retained_mutex);
        self->retained_leases.push_back(retained);
      }
    }

    // Always release our reference (retained leases hold their own copy).
    // iree_async_buffer_lease_release handles NULL leases (returns
    // immediately).
    iree_async_buffer_lease_release(lease);

    if (self->inject_error.exchange(false, std::memory_order_acq_rel)) {
      return iree_make_status(IREE_STATUS_INTERNAL, "fuzz-injected error");
    }
    return iree_ok_status();
  }

  iree_net_carrier_recv_handler_t AsHandler() { return {Handler, this}; }

  // Release all retained leases. Called from the main thread.
  void ReleaseAll() {
    std::lock_guard<std::mutex> lock(retained_mutex);
    for (auto& lease : retained_leases) {
      iree_async_buffer_lease_release(&lease);
    }
    retained_leases.clear();
  }
};

// NullRecvHandler and MakeNullRecvHandler are in registry.h (shared utilities).

//===----------------------------------------------------------------------===//
// Concurrent poll thread (for TSan coverage)
//===----------------------------------------------------------------------===//

// Polls the proactor on a background thread, exercising the concurrent
// send/recv path that production code uses. TSan instrumentation catches
// races between the main thread (sends) and this thread (completions).
struct PollThread {
  std::atomic<bool> stop{false};
  iree_async_proactor_t* proactor = nullptr;
  std::thread thread;

  void Start(iree_async_proactor_t* target_proactor) {
    proactor = target_proactor;
    thread = std::thread(&PollThread::Run, this);
  }

  void Stop() {
    stop.store(true, std::memory_order_release);
    if (thread.joinable()) thread.join();
  }

 private:
  void Run() {
    while (!stop.load(std::memory_order_acquire)) {
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor, iree_make_timeout_ms(10), &completed);
      iree_status_ignore(status);
    }
  }
};

// DeactivateAndDrain is in registry.h (shared utilities).

//===----------------------------------------------------------------------===//
// Send size decoding
//===----------------------------------------------------------------------===//

// Decodes a 4-bit size class to a byte count for send operations.
// Returns 0 for class 0 (empty send, should be rejected by the carrier).
// For classes 1-15, returns min(2^(class-1), remaining_bytes).
inline size_t DecodeSendSize(uint8_t size_class, size_t remaining_bytes) {
  if (size_class == 0) return 0;
  size_t requested = (size_t)1 << (size_class - 1);
  return (requested < remaining_bytes) ? requested : remaining_bytes;
}

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_FUZZ_BASE_H_
