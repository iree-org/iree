// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transport factory registry for runtime discovery of available transports.
//
// The registry provides HAL-driver style registration of transport factories.
// Applications look up factories by scheme rather than hardcoding transport
// types:
//
//   iree_net_transport_factory_t* tcp =
//       iree_net_transport_registry_lookup(registry, IREE_SV("tcp"));
//   if (tcp) {
//     iree_net_transport_factory_connect(tcp, "server:8080", ...);
//   }
//
// Registration follows the HAL pattern where each transport provides a module
// registration function, and an init.c file with #ifdef guards calls these:
//
//   #if defined(IREE_HAVE_NET_TCP_MODULE)
//   IREE_RETURN_AND_END_ZONE_IF_ERROR(
//       z0, iree_net_tcp_module_register(registry, host_allocator));
//   #endif
//
// Registration functions can probe for runtime dependencies. Return ok_status()
// without registering if a transport is unavailable at runtime; only return
// errors for actual failures. This allows graceful degradation when optional
// transports (like RDMA) are not available.
//
// The registry retains registered factories and releases them when the registry
// is freed. Callers may independently retain factories obtained via lookup.

#ifndef IREE_NET_TRANSPORT_REGISTRY_H_
#define IREE_NET_TRANSPORT_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_net_transport_registry_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_transport_registry_t iree_net_transport_registry_t;

// Allocates an empty transport registry.
//
// Applications typically have one registry per process. After allocation,
// populate it by calling registration functions for each desired transport.
// The registry is single-owner and not reference counted.
IREE_API_EXPORT iree_status_t iree_net_transport_registry_allocate(
    iree_allocator_t host_allocator,
    iree_net_transport_registry_t** out_registry);

// Frees the registry and releases all registered factories.
// Factory pointers obtained via lookup remain valid only if the caller
// independently retained them.
IREE_API_EXPORT void iree_net_transport_registry_free(
    iree_net_transport_registry_t* registry);

// Registers a factory for the given scheme (e.g., "tcp", "quic", "rdma").
//
// The registry retains the factory. The caller may release their reference
// after registration succeeds. If registration fails, the caller retains
// ownership and must release the factory.
//
// Returns IREE_STATUS_ALREADY_EXISTS if a factory is already registered for
// this scheme. Returns IREE_STATUS_INVALID_ARGUMENT if scheme is empty.
IREE_API_EXPORT iree_status_t iree_net_transport_registry_register(
    iree_net_transport_registry_t* registry, iree_string_view_t scheme,
    iree_net_transport_factory_t* factory);

// Looks up a factory by scheme.
// Returns NULL if no factory is registered for the scheme. The returned
// factory is valid as long as the registry is alive. Callers that need the
// factory to outlive the registry must retain it.
IREE_API_EXPORT iree_net_transport_factory_t*
iree_net_transport_registry_lookup(iree_net_transport_registry_t* registry,
                                   iree_string_view_t scheme);

// Callback invoked for each registered factory during enumeration.
// Return iree_ok_status() to continue enumeration, or any error status to
// stop early. The error status is propagated from enumerate().
typedef iree_status_t (*iree_net_transport_registry_enumerate_fn_t)(
    void* user_data, iree_string_view_t scheme,
    iree_net_transport_factory_t* factory);

// Enumerates all registered factories in registration order.
// The callback receives each scheme/factory pair. If the callback returns
// a non-OK status, iteration stops and that status is returned.
IREE_API_EXPORT iree_status_t iree_net_transport_registry_enumerate(
    iree_net_transport_registry_t* registry,
    iree_net_transport_registry_enumerate_fn_t callback, void* user_data);

// Returns the number of registered factories.
IREE_API_EXPORT iree_host_size_t
iree_net_transport_registry_count(iree_net_transport_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_TRANSPORT_REGISTRY_H_
