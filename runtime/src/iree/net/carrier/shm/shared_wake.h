// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-proactor shared wake: consolidates SHM carrier wake primitives.
//
// All SHM carriers on a given proactor share a single notification (one fd)
// instead of each carrier having its own event pair. On wake, a scan callback
// iterates all sleeping carriers and dispatches to per-carrier drain logic.
//
// This provides O(1) fd consumption per proactor regardless of carrier count,
// and enables wake coalescing: multiple carriers signaling simultaneously
// produce one kernel wake followed by one scan. The scan cost per carrier is
// a single acquire-load on the SPSC ring position (~50ns), which is negligible
// against the kernel wake latency (1-5us).
//
// The shared wake maintains an intrusive singly-linked list of carriers in
// sleep mode. All list mutations happen on the poll thread only (no
// synchronization needed). Carriers in poll mode (AIMD progress callbacks) are
// not in this list — they transition back to the sleeping list when idle.
//
// Lifetime:
//   - Created per-proactor (typically by a factory or test harness).
//   - Ref-counted: each carrier retains a reference.
//   - The notification is created as a local (non-shared) notification on the
//     proactor. Peer signaling uses iree_async_notification_signal on this
//     notification directly.

#ifndef IREE_NET_CARRIER_SHM_SHARED_WAKE_H_
#define IREE_NET_CARRIER_SHM_SHARED_WAKE_H_

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_shm_carrier_t iree_net_shm_carrier_t;

// Per-proactor shared wake for SHM carriers. One notification fd serves all
// carriers on this proactor. Carriers register/unregister as they transition
// between sleep and poll modes.
typedef struct iree_net_shm_shared_wake_t {
  iree_atomic_ref_count_t ref_count;

  // Proactor this shared wake is bound to. Retained.
  iree_async_proactor_t* proactor;

  // Local notification (one fd). Signaled by peer carriers to wake the scan.
  iree_async_notification_t* notification;

  // NOTIFICATION_WAIT operation for the scan callback.
  iree_async_notification_wait_operation_t wait_operation;

  // True when a NOTIFICATION_WAIT is in flight on the proactor.
  bool wait_posted;

  // Intrusive singly-linked list of carriers in sleep mode.
  // All mutations are poll-thread-only (no synchronization needed).
  // Linked via iree_net_shm_carrier_t::sleeping_next.
  iree_net_shm_carrier_t* sleeping_head;

  iree_allocator_t allocator;
} iree_net_shm_shared_wake_t;

// Creates a shared wake bound to |proactor|. Creates a local notification on
// the proactor for wake signaling. The caller receives one reference.
IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_create(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_net_shm_shared_wake_t** out_shared_wake);

// Retains a reference to the shared wake.
IREE_API_EXPORT void iree_net_shm_shared_wake_retain(
    iree_net_shm_shared_wake_t* shared_wake);

// Releases a reference. Destroys when the reference count reaches zero.
IREE_API_EXPORT void iree_net_shm_shared_wake_release(
    iree_net_shm_shared_wake_t* shared_wake);

// Adds a carrier to the sleeping list and ensures the NOTIFICATION_WAIT is
// posted. Poll-thread-only. The carrier must not already be in the list.
// On failure (proactor submission failed), the carrier is NOT added to the
// sleeping list and the caller must handle the error.
IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_register(
    iree_net_shm_shared_wake_t* shared_wake, iree_net_shm_carrier_t* carrier);

// Removes a carrier from the sleeping list. Poll-thread-only. Safe to call
// if the carrier is not in the list (no-op).
IREE_API_EXPORT void iree_net_shm_shared_wake_unregister(
    iree_net_shm_shared_wake_t* shared_wake, iree_net_shm_carrier_t* carrier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_SHARED_WAKE_H_
