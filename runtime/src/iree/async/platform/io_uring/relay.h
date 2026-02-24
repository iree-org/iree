// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for io_uring relay implementation.
//
// io_uring-specific state enum and implementation functions. The relay struct
// is defined in the shared iree/async/relay.h with a platform union.
// Relays use multishot POLL_ADD for persistent monitoring and callback-based
// sink execution during poll().

#ifndef IREE_ASYNC_PLATFORM_IO_URING_RELAY_H_
#define IREE_ASYNC_PLATFORM_IO_URING_RELAY_H_

#include "iree/async/relay.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_io_uring_t iree_async_proactor_io_uring_t;

//===----------------------------------------------------------------------===//
// Relay state
//===----------------------------------------------------------------------===//

// Internal state for relay lifecycle management.
typedef enum iree_async_io_uring_relay_state_e {
  // Relay is active and monitoring source.
  IREE_ASYNC_IO_URING_RELAY_STATE_ACTIVE = 0,

  // Relay is marked for removal, waiting for final CQE.
  // The sink will not fire in this state.
  IREE_ASYNC_IO_URING_RELAY_STATE_ZOMBIE = 1,

  // Relay needs to re-arm its source monitoring but failed due to SQ pressure.
  // The proactor will retry submission on the next poll cycle.
  IREE_ASYNC_IO_URING_RELAY_STATE_PENDING_REARM = 2,

  // Relay failed to re-arm due to an unrecoverable error (syscall failure).
  // The error callback has been invoked. Relay should be unregistered.
  IREE_ASYNC_IO_URING_RELAY_STATE_FAULTED = 3,
} iree_async_io_uring_relay_state_t;

//===----------------------------------------------------------------------===//
// Implementation functions
//===----------------------------------------------------------------------===//

iree_status_t iree_async_io_uring_register_relay(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay);

void iree_async_io_uring_unregister_relay(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay);

// Called from CQE processing when a relay's source fires.
// Executes the sink action and handles re-arming or cleanup.
void iree_async_io_uring_handle_relay_cqe(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay,
    int32_t result, uint32_t cqe_flags);

// Retries re-arming relays that are in PENDING_REARM state.
// Called from poll() after processing CQEs when SQ space may be available.
void iree_async_io_uring_retry_pending_relays(
    iree_async_proactor_io_uring_t* proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_RELAY_H_
