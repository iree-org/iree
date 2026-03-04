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
// Two creation modes:
//
//   iree_net_shm_shared_wake_create():
//     Creates a local notification on the proactor. Suitable for in-process
//     carrier pairs where both sides run on proactors in the same process.
//     Peer signaling uses iree_async_notification_signal() directly.
//
//   iree_net_shm_shared_wake_create_shared():
//     Creates a cross-process-capable notification backed by a dedicated SHM
//     page for the epoch and platform-specific wake/signal primitives (eventfd
//     on Linux, pipe on macOS, Event on Windows). The signal primitive can be
//     exported to remote peers via iree_net_shm_shared_wake_export(). Remote
//     peers create a proxy notification using
//     iree_async_notification_create_shared() with the exported handles,
//     enabling cross-process signaling.
//
// Lifetime:
//   - Created per-proactor (typically by a factory or test harness).
//   - Ref-counted: each carrier retains a reference.

#ifndef IREE_NET_CARRIER_SHM_SHARED_WAKE_H_
#define IREE_NET_CARRIER_SHM_SHARED_WAKE_H_

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/primitive.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/shm.h"

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

  // Notification (one fd). Signaled by peer carriers to wake the scan.
  // For local mode: a local notification (proactor-managed lifetime).
  // For shared mode: a shared notification backed by epoch_mapping and
  // owned_wake_primitive/owned_signal_primitive below.
  iree_async_notification_t* notification;

  // NOTIFICATION_WAIT operation for the scan callback.
  iree_async_notification_wait_operation_t wait_operation;

  // True when a NOTIFICATION_WAIT is in flight on the proactor.
  bool wait_posted;

  // True when this shared wake uses cross-process-capable primitives.
  // Controls whether destroy closes the SHM mapping and owned primitives.
  bool is_shared;

  // Intrusive singly-linked list of carriers in sleep mode.
  // All mutations are poll-thread-only (no synchronization needed).
  // Linked via iree_net_shm_carrier_t::sleeping_next.
  iree_net_shm_carrier_t* sleeping_head;

  // For shared (cross-process) mode: SHM region containing the epoch counter.
  // The notification's epoch_ptr points into this mapping. Zero-initialized
  // for local mode (not used).
  iree_shm_mapping_t epoch_mapping;

  // For shared mode: owned wake/signal primitives. The shared notification
  // references but does not own these (SHARED flag means destroy does not
  // close them). We close them on shared_wake destroy.
  //
  // Linux:   both point to the same eventfd.
  // macOS:   wake = pipe read end, signal = pipe write end.
  // Windows: wake = Event HANDLE, signal = Event HANDLE.
  iree_async_primitive_t owned_wake_primitive;
  iree_async_primitive_t owned_signal_primitive;

  iree_allocator_t allocator;
} iree_net_shm_shared_wake_t;

// Creates a shared wake bound to |proactor| with a local notification.
// Suitable for in-process carrier pairs. The caller receives one reference.
IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_create(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_net_shm_shared_wake_t** out_shared_wake);

// Creates a shared wake bound to |proactor| with cross-process-capable
// notification primitives.
//
// Allocates a dedicated SHM page for the notification epoch and creates
// platform-specific wake/signal primitives (eventfd on Linux, pipe on macOS,
// Event on Windows). The resulting notification supports cross-process
// signaling when the signal primitive is exported to remote peers.
//
// Use iree_net_shm_shared_wake_export() to obtain duplicated handles for
// cross-process handshake.
IREE_API_EXPORT iree_status_t iree_net_shm_shared_wake_create_shared(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_net_shm_shared_wake_t** out_shared_wake);

// Exported handles for cross-process handshake.
// All handles are independently owned duplicates — the caller must close them
// after IPC transfer (or on error) using iree_shm_handle_close() and
// iree_async_primitive_close().
typedef struct iree_net_shm_shared_wake_export_t {
  // Duplicated handle to the SHM region containing the epoch counter.
  // The remote peer maps this to create a proxy notification whose epoch_ptr
  // points at the same physical page.
  iree_shm_handle_t epoch_shm_handle;
  // Size of the epoch SHM region (always one page).
  iree_host_size_t epoch_shm_size;
  // Duplicated signal primitive. The remote peer writes to this to wake our
  // proactor. On Linux: dup'd eventfd. On macOS: dup'd pipe write end.
  // On Windows: dup'd Event HANDLE.
  iree_async_primitive_t signal_primitive;
} iree_net_shm_shared_wake_export_t;

// Exports duplicated handles for cross-process handshake.
//
// Each call produces fresh duplicates safe for one IPC transfer. The caller
// must close the returned handles after sending them (or on error).
//
// Requires the shared wake to have been created with create_shared().
// Returns FAILED_PRECONDITION if called on a local shared wake.
IREE_API_EXPORT iree_status_t
iree_net_shm_shared_wake_export(iree_net_shm_shared_wake_t* shared_wake,
                                iree_net_shm_shared_wake_export_t* out_export);

// Retains a reference to the shared wake.
IREE_API_EXPORT void iree_net_shm_shared_wake_retain(
    iree_net_shm_shared_wake_t* shared_wake);

// Releases a reference. Destroys when the reference count reaches zero.
// For shared mode: closes the epoch SHM mapping and owned wake/signal
// primitives.
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

//===----------------------------------------------------------------------===//
// Carrier accessors for sleeping list traversal
//===----------------------------------------------------------------------===//
// Declared here (not in carrier.h) because these are implementation details
// of the shared_wake ↔ carrier interaction. The carrier struct is opaque to
// shared_wake.c; these accessors provide the minimal surface needed for
// sleeping list traversal without exposing the full carrier layout.

// Returns the next carrier in the sleeping list (NULL if tail).
iree_net_shm_carrier_t* iree_net_shm_carrier_sleeping_next(
    iree_net_shm_carrier_t* carrier);

// Sets the next carrier in the sleeping list.
void iree_net_shm_carrier_set_sleeping_next(iree_net_shm_carrier_t* carrier,
                                            iree_net_shm_carrier_t* next);

// Performs per-carrier drain logic when woken by the shared wake scan.
// Returns true if the carrier should remain in the sleeping list, false if
// it was removed (transitioned to poll mode or stopped).
bool iree_net_shm_carrier_drain_from_wake(iree_net_shm_carrier_t* carrier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_SHARED_WAKE_H_
