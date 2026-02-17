// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_UTIL_SIGNAL_H_
#define IREE_ASYNC_UTIL_SIGNAL_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Global signal ownership
//===----------------------------------------------------------------------===//

// Claims global signal ownership for |proactor|.
//
// Only one proactor per process may own signal subscriptions at a time. The
// first to call this function claims ownership. Subsequent calls from the same
// proactor return OK; calls from different proactors return
// FAILED_PRECONDITION.
//
// When a proactor that owns signals is destroyed, it should call
// iree_async_signal_release_ownership() to allow another proactor to claim
// ownership. This supports sequential ownership transfers while preventing
// concurrent ownership which would cause race conditions.
//
// Thread-safe: Uses atomic compare-exchange.
iree_status_t iree_async_signal_claim_ownership(
    iree_async_proactor_t* proactor);

// Releases signal ownership for |proactor| if it is the current owner.
//
// Call this when a proactor that was handling signals is being destroyed.
// After this call, another proactor may claim ownership.
//
// The goal is to prevent multiple CONCURRENT owners. Sequential ownership
// (proactor A claims, destroys, then proactor B claims) is allowed.
//
// Thread-safe: Uses atomic compare-exchange.
void iree_async_signal_release_ownership(iree_async_proactor_t* proactor);

//===----------------------------------------------------------------------===//
// Signal dispatch callback
//===----------------------------------------------------------------------===//

// Callback invoked for each signal during drain of the signal fd (signalfd or
// self-pipe). Used by both signal backends; defined here so both can share the
// same signature.
typedef struct iree_async_signal_dispatch_callback_t {
  void (*fn)(void* user_data, iree_async_signal_t signal);
  void* user_data;
} iree_async_signal_dispatch_callback_t;

//===----------------------------------------------------------------------===//
// Subscription list management
//===----------------------------------------------------------------------===//

// State for deferred unsubscribe during dispatch.
// Embedded in proactor's signal handling state.
typedef struct iree_async_signal_dispatch_state_t {
  // True while iterating subscription lists (prevents immediate unsubscribe).
  bool dispatching;
  // Linked list of subscriptions to unsubscribe after dispatch completes.
  iree_async_signal_subscription_t* pending_unsubscribes;
} iree_async_signal_dispatch_state_t;

// Initializes dispatch state. Call once during proactor signal setup.
static inline void iree_async_signal_dispatch_state_initialize(
    iree_async_signal_dispatch_state_t* state) {
  state->dispatching = false;
  state->pending_unsubscribes = NULL;
}

// Initializes a subscription with the given parameters.
// Does NOT link it into any list; call link() separately.
void iree_async_signal_subscription_initialize(
    iree_async_signal_subscription_t* subscription,
    iree_async_proactor_t* proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback);

// Links |subscription| at the tail of the list headed by |*head_ptr|.
// Updates |*head_ptr| if the list was empty.
// |subscription| must not already be linked.
void iree_async_signal_subscription_link(
    iree_async_signal_subscription_t** head_ptr,
    iree_async_signal_subscription_t* subscription);

// Unlinks |subscription| from the list headed by |*head_ptr|.
// Updates |*head_ptr| if the head was removed.
// Returns true if the subscription was actually unlinked, false if it was
// already unlinked or NULL.
bool iree_async_signal_subscription_unlink(
    iree_async_signal_subscription_t** head_ptr,
    iree_async_signal_subscription_t* subscription);

// Dispatches |signal| to all subscriptions in the list headed by |head|.
// Callbacks fire in registration order.
//
// Uses |dispatch_state| for deferred unsubscribe handling: if a callback
// unsubscribes itself or another subscription, the actual unlink is deferred
// until dispatch completes. This prevents list corruption during iteration.
//
// After all callbacks have fired, processes any pending unsubscribes. Returns
// the list of subscriptions that were unsubscribed (caller should free them).
iree_async_signal_subscription_t* iree_async_signal_subscription_dispatch(
    iree_async_signal_subscription_t* head,
    iree_async_signal_dispatch_state_t* dispatch_state,
    iree_async_signal_t signal);

// Marks |subscription| for deferred unsubscribe.
// Call this from unsubscribe_signal when dispatch_state->dispatching is true.
// Sets callback.fn to NULL to prevent future invocations.
void iree_async_signal_subscription_defer_unsubscribe(
    iree_async_signal_dispatch_state_t* dispatch_state,
    iree_async_signal_subscription_t* subscription);

//===----------------------------------------------------------------------===//
// POSIX signal number conversion
//===----------------------------------------------------------------------===//

#if !defined(IREE_PLATFORM_WINDOWS)

// Converts an IREE signal enum to the corresponding POSIX signal number.
// Returns 0 for IREE_ASYNC_SIGNAL_NONE or invalid signals.
int iree_async_signal_to_posix(iree_async_signal_t signal);

// Converts a POSIX signal number to the corresponding IREE signal enum.
// Returns IREE_ASYNC_SIGNAL_NONE for unknown signals.
iree_async_signal_t iree_async_signal_from_posix(int signo);

// Builds a sigset_t containing all signals that IREE handles.
// This is used by signalfd and pthread_sigmask.
void iree_async_signal_build_sigset(sigset_t* mask);

#endif  // !IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_SIGNAL_H_
