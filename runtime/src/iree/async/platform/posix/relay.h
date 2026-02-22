// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for POSIX relay implementation.
//
// The relay struct is defined in the shared iree/async/relay.h with a platform
// union. This header declares POSIX-specific implementation functions called by
// the POSIX proactor.
//
// POSIX relays dispatch via the unified fd_map:
//   - Primitive sources: registered in fd_map as RELAY handler, fd added to
//     event_set. Dispatched from the poll loop via single fd_map lookup.
//   - Notification sources: linked into the source notification's per-
//     notification relay_list. When the notification's fd fires, both pending
//     async waits and relay subscribers are dispatched from the NOTIFICATION
//     handler case.

#ifndef IREE_ASYNC_PLATFORM_POSIX_RELAY_H_
#define IREE_ASYNC_PLATFORM_POSIX_RELAY_H_

#include "iree/async/relay.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_posix_t iree_async_proactor_posix_t;

// Registers a relay with the POSIX proactor. Implements the vtable entry.
// For primitive sources: registers in fd_map as RELAY handler and adds the
//   source fd to the event_set.
// For notification sources: links into the source notification's relay_list
//   and activates the notification fd if this is the first consumer.
iree_status_t iree_async_proactor_posix_register_relay(
    iree_async_proactor_posix_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay);

// Unregisters a relay. Implements the vtable entry.
// Synchronous cleanup: removes from fd_map and event_set (primitive sources),
// or removes from notification relay_list (notification sources), closes owned
// fds, releases retained notifications, unlinks from relay list, and frees.
void iree_async_proactor_posix_unregister_relay(
    iree_async_proactor_posix_t* proactor, iree_async_relay_t* relay);

// Dispatches a relay whose primitive source fd became ready.
// Checks ERROR_SENSITIVE flags, fires sink, drains source for persistent
// relays, and handles one-shot cleanup.
void iree_async_proactor_posix_dispatch_relay(
    iree_async_proactor_posix_t* proactor, iree_async_relay_t* relay,
    short revents);

// Dispatches notification-source relays for a notification whose fd fired.
// Walks the notification's relay_list, checking each relay's wait_epoch against
// the current epoch. Fires sinks for relays whose epoch has advanced.
// Handles one-shot cleanup and persistent re-arming (epoch update).
void iree_async_proactor_posix_dispatch_notification_relays(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_t* notification);

// Destroys all relays during proactor cleanup. Releases all resources without
// firing sinks.
void iree_async_proactor_posix_destroy_all_relays(
    iree_async_proactor_posix_t* proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_RELAY_H_
