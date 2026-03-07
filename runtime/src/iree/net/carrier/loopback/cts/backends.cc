// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loopback carrier CTS backend registration.
//
// Registers the loopback carrier as a CTS backend with appropriate capability
// tags. The loopback backend runs everywhere without network dependencies,
// making it ideal for CI environments.

#include <atomic>
#include <string>

#include "iree/async/platform/posix/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/loopback/carrier.h"
#include "iree/net/carrier/loopback/factory.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Factory function
//===----------------------------------------------------------------------===//

// Creates a loopback carrier pair. When |proactor| is NULL a new proactor is
// created and returned via pair.proactor; otherwise the caller's proactor is
// used (the production pattern for connection pools and proactor reuse).
static iree::StatusOr<CarrierPair> CreateLoopbackCarrierPair(
    iree_async_proactor_t* proactor) {
  bool created_proactor = (proactor == nullptr);
  if (created_proactor) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
  }

  iree_net_carrier_t* client = nullptr;
  iree_net_carrier_t* server = nullptr;
  iree_net_carrier_callback_t callback = {nullptr, nullptr};
  iree_status_t status = iree_net_loopback_carrier_create_pair(
      proactor, callback, iree_allocator_system(), &client, &server);
  if (!iree_status_is_ok(status)) {
    if (created_proactor) iree_async_proactor_release(proactor);
    return iree::Status(std::move(status));
  }

  CarrierPair pair;
  pair.client = client;
  pair.server = server;
  pair.proactor = proactor;
  pair.context = nullptr;
  pair.cleanup = nullptr;
  return pair;
}

//===----------------------------------------------------------------------===//
// Factory-level CTS support
//===----------------------------------------------------------------------===//

static iree_status_t CreateLoopbackFactory(
    iree_allocator_t allocator, iree_net_transport_factory_t** out_factory) {
  return iree_net_loopback_factory_create(
      iree_net_loopback_factory_options_default(), allocator, out_factory);
}

static std::string MakeLoopbackBindAddress() {
  static std::atomic<int> counter{0};
  return "cts_" + std::to_string(counter.fetch_add(1));
}

static std::string ResolveLoopbackConnectAddress(
    const std::string& bind_address, iree_net_listener_t* /*listener*/) {
  // Loopback uses name-based addressing — the bind name is the connect name.
  return bind_address;
}

static std::string MakeLoopbackUnreachableAddress(
    iree_async_proactor_t* /*proactor*/) {
  static std::atomic<int> counter{0};
  return "unreachable_" + std::to_string(counter.fetch_add(1));
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// Creates a loopback carrier pair using a POSIX proactor with the specified
// event backend. Returns UNAVAILABLE when the POSIX proactor or its event
// backend is not supported on this platform.
static iree::StatusOr<CarrierPair> CreateLoopbackCarrierPairPosix(
    iree_async_proactor_t* proactor,
    iree_async_posix_event_backend_t event_backend) {
  bool created_proactor = (proactor == nullptr);
  if (created_proactor) {
    iree_status_t status = iree_async_proactor_create_posix_with_backend(
        iree_async_proactor_options_default(), event_backend,
        iree_allocator_system(), &proactor);
    if (!iree_status_is_ok(status)) return iree::Status(std::move(status));
  }

  iree_net_carrier_t* client = nullptr;
  iree_net_carrier_t* server = nullptr;
  iree_net_carrier_callback_t callback = {nullptr, nullptr};
  iree_status_t status = iree_net_loopback_carrier_create_pair(
      proactor, callback, iree_allocator_system(), &client, &server);
  if (!iree_status_is_ok(status)) {
    if (created_proactor) iree_async_proactor_release(proactor);
    return iree::Status(std::move(status));
  }

  CarrierPair pair;
  pair.client = client;
  pair.server = server;
  pair.proactor = proactor;
  pair.context = nullptr;
  pair.cleanup = nullptr;
  return pair;
}

static iree::StatusOr<CarrierPair> CreateLoopbackCarrierPairEpoll(
    iree_async_proactor_t* proactor) {
  return CreateLoopbackCarrierPairPosix(proactor,
                                        IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL);
}

static iree::StatusOr<CarrierPair> CreateLoopbackCarrierPairPoll(
    iree_async_proactor_t* proactor) {
  return CreateLoopbackCarrierPairPosix(proactor,
                                        IREE_ASYNC_POSIX_EVENT_BACKEND_POLL);
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// Loopback on io_uring: reliable, ordered, zero-copy TX (for single spans).
static bool loopback_registered =
    (CtsRegistry::RegisterBackend(
         {"loopback",
          {"loopback", CreateLoopbackCarrierPair, CreateLoopbackFactory,
           MakeLoopbackBindAddress, ResolveLoopbackConnectAddress,
           MakeLoopbackUnreachableAddress},
          {"reliable", "ordered", "zerocopy_tx", "factory"}}),
     true);

// Loopback on POSIX epoll: same capabilities (loopback is proactor-agnostic).
static bool loopback_epoll_registered =
    (CtsRegistry::RegisterBackend({
         "loopback_epoll",
         {"loopback_epoll", CreateLoopbackCarrierPairEpoll},
         {"reliable", "ordered", "zerocopy_tx"},
     }),
     true);

// Loopback on POSIX poll: same capabilities (loopback is proactor-agnostic).
static bool loopback_poll_registered =
    (CtsRegistry::RegisterBackend({
         "loopback_poll",
         {"loopback_poll", CreateLoopbackCarrierPairPoll},
         {"reliable", "ordered", "zerocopy_tx"},
     }),
     true);

}  // namespace iree::net::carrier::cts
