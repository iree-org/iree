// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SHM carrier CTS backend registration.
//
// Registers the SHM carrier as a CTS backend with appropriate capability tags.
// The SHM backend is proactor-agnostic (notifications work with any backend).

#include <atomic>
#include <string>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/shm/carrier_pair.h"
#include "iree/net/carrier/shm/factory.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Factory function
//===----------------------------------------------------------------------===//

// Creates an SHM carrier pair. When |proactor| is NULL a new proactor is
// created and returned via pair.proactor; otherwise the caller's proactor is
// used.
static iree::StatusOr<CarrierPair> CreateShmCarrierPair(
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
  iree_status_t status = iree_net_shm_carrier_create_pair(
      proactor, proactor, iree_net_shm_carrier_options_default(), callback,
      iree_allocator_system(), &client, &server);
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

static iree_status_t CreateShmFactory(
    iree_allocator_t allocator, iree_net_transport_factory_t** out_factory) {
  return iree_net_shm_factory_allocate(iree_net_shm_carrier_options_default(),
                                       allocator, out_factory);
}

static std::string MakeShmBindAddress() {
  static std::atomic<int> counter{0};
  return "shm_cts_" + std::to_string(counter.fetch_add(1));
}

static std::string ResolveShmConnectAddress(
    const std::string& bind_address, iree_net_listener_t* /*listener*/) {
  // SHM uses name-based addressing — the bind name is the connect name.
  return bind_address;
}

static std::string MakeShmUnreachableAddress(
    iree_async_proactor_t* /*proactor*/) {
  static std::atomic<int> counter{0};
  return "shm_unreachable_" + std::to_string(counter.fetch_add(1));
}

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// SHM carrier: reliable, ordered, zero-copy TX, with factory support.
static bool shm_registered =
    (CtsRegistry::RegisterBackend(
         {"shm",
          {"shm", CreateShmCarrierPair, CreateShmFactory,
           MakeShmBindAddress, ResolveShmConnectAddress,
           MakeShmUnreachableAddress},
          {"reliable", "ordered", "zerocopy_tx", "factory"}}),
     true);

}  // namespace iree::net::carrier::cts
