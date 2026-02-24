// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared implementation for IREE_ASYNC_OPERATION_TYPE_SEQUENCE operations.
//
// Sequences run N operations in order. Two execution paths are available:
//
//   LINK path (step_fn == NULL):
//     Expands the sequence's steps as a linked batch through the backend's
//     normal submit path with IREE_ASYNC_OPERATION_FLAG_LINKED set on each
//     step. This reuses the existing linked operation infrastructure:
//       - io_uring: kernel IOSQE_IO_LINK chains with zero userspace
//         round-trips for kernel-linkable segments (split points at
//         TIMER/SEMAPHORE_* operations fall back to userspace linked_next).
//       - POSIX: userspace linked_next chains with one poll round-trip per
//         chain segment via dispatch_linked_continuation.
//     All backends supporting LINKED_OPERATIONS capability get this path.
//
//   Emulation path (step_fn != NULL):
//     Submits one step at a time, calling step_fn between steps to allow
//     dynamic construction of later steps based on earlier results. Each
//     step completion fires an internal trampoline that advances the
//     sequence. One poll round-trip per step.
//
// Both paths produce identical observable behavior: the sequence's base
// completion callback fires exactly once when all steps complete (with OK),
// when any step fails (with the error), or when the sequence is cancelled
// (with IREE_STATUS_CANCELLED).
//
// Backend integration:
//   Each backend adds a SEQUENCE case in its submit and cancel switches:
//     submit: route to submit_as_linked (step_fn==NULL) or
//             emulation_begin (step_fn!=NULL)
//     cancel: route to iree_async_sequence_cancel
//
// Thread safety:
//   Not thread-safe. All calls happen from the poll thread (submit is called
//   from the backend's submit path, trampolines fire from poll-dispatched
//   callbacks). The only cross-thread interaction is cancel(), which sets a
//   flag read by the emulation trampoline; ordering is guaranteed by the
//   cancel-step submission and completion processing barriers.

#ifndef IREE_ASYNC_UTIL_SEQUENCE_EMULATION_H_
#define IREE_ASYNC_UTIL_SEQUENCE_EMULATION_H_

#include "iree/async/operation.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Internal flags (stored in sequence->base.internal_flags)
//===----------------------------------------------------------------------===//

// A non-CANCELLED error was received from a step. The error status is buffered
// in sequence->internal.stashed_error. The link trampoline uses this to report
// the correct error when CQE ordering varies by backend: io_uring delivers the
// error CQE before cancelled CQEs, while POSIX delivers cancelled continuations
// before the triggering error callback.
#define IREE_ASYNC_SEQUENCE_INTERNAL_SAW_ERROR (1u << 14)

// Cancel was requested via iree_async_sequence_cancel(). The emulation
// trampoline checks this before submitting the next step and aborts with
// IREE_STATUS_CANCELLED if set.
#define IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED (1u << 15)

//===----------------------------------------------------------------------===//
// LINK path (step_fn == NULL)
//===----------------------------------------------------------------------===//

// Submits a sequence's steps as a linked batch through the proactor's normal
// submit path. Sets IREE_ASYNC_OPERATION_FLAG_LINKED on steps[0..N-2] and
// installs link trampolines on all steps that coordinate completion counting.
//
// The vtable submit call re-enters the backend's submit function, but the
// expanded steps are not SEQUENCE type and proceed through normal processing.
//
// Zero-step sequences complete immediately: the base callback fires with OK
// before this function returns.
//
// The caller (backend submit path) must not touch the sequence or its steps
// after this call. The link trampolines own them until the base callback fires.
iree_status_t iree_async_sequence_submit_as_linked(
    iree_async_proactor_t* proactor, iree_async_sequence_operation_t* sequence);

//===----------------------------------------------------------------------===//
// Emulation path (step_fn != NULL)
//===----------------------------------------------------------------------===//

// Callback used by the sequence emulator to submit the next step operation.
// The backend provides this during initialization so the emulator can submit
// individual step operations through the backend's normal submission path.
//
// |proactor| is the proactor to submit through.
// |operation| is the next step to submit (already prepared by the emulator).
//
// Returns OK if submission succeeded, or an error that will abort the sequence.
typedef iree_status_t (*iree_async_sequence_submit_fn_t)(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation);

// Drives sequence operations step-by-step for sequences with step_fn set.
//
// The emulator stores no per-sequence state beyond what's already in the
// iree_async_sequence_operation_t struct. It uses the sequence's current_step
// field to track progress and overwrites step callbacks in-place. The emulator
// pointer is stored in the sequence's internal.emulator union field during
// execution. No additional allocation.
typedef struct iree_async_sequence_emulator_t {
  // The proactor that owns this emulator (for submitting subsequent steps).
  iree_async_proactor_t* proactor;

  // Backend-provided function for submitting individual step operations.
  iree_async_sequence_submit_fn_t submit_fn;
} iree_async_sequence_emulator_t;

// Initializes a sequence emulator.
// |proactor| is the owning proactor (used as context for submit calls).
// |submit_fn| is the backend's submission function for individual operations.
//
// The emulator struct is typically embedded in the backend's proactor
// implementation struct (zero additional allocation).
static inline void iree_async_sequence_emulator_initialize(
    iree_async_sequence_emulator_t* emulator, iree_async_proactor_t* proactor,
    iree_async_sequence_submit_fn_t submit_fn) {
  emulator->proactor = proactor;
  emulator->submit_fn = submit_fn;
}

// Begins execution of a sequence operation via the emulation path.
// Submits step 0 and sets up internal trampolines for step advancement.
// The sequence's current_step is reset to 0.
//
// The caller (backend submit path) must not touch the sequence or its steps
// after this call — the emulator owns them until the base callback fires.
//
// Zero-step sequences complete immediately: the base callback fires with OK
// before this function returns.
//
// Returns OK if step 0 was submitted successfully, or the submission error
// if it failed (in which case the sequence's base callback fires with that
// error before this function returns).
iree_status_t iree_async_sequence_emulation_begin(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence);

// Called by the emulator's internal trampolines when a step completes.
// Advances the sequence to the next step, calling step_fn if present.
//
// Backend code does NOT call this directly — it is invoked through the
// completion callback trampolines that _begin() installs on each step.
// This function is exposed in the header for testing purposes only.
void iree_async_sequence_emulation_step_completed(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence, iree_status_t step_status);

//===----------------------------------------------------------------------===//
// Cancellation (shared by both paths)
//===----------------------------------------------------------------------===//

// Requests cancellation of an in-flight sequence operation.
// Sets CANCEL_REQUESTED on the sequence's internal_flags and attempts to
// cancel the current in-flight step via iree_async_proactor_cancel().
//
// Always returns iree_ok_status(). The actual cancellation is asynchronous:
// the CANCEL_REQUESTED flag guarantees both the LINK and emulation trampolines
// will produce IREE_STATUS_CANCELLED as the sequence's final status, even if
// the cancel-step call is a no-op (step completed between the flag-set and
// the cancel submission).
//
// For the LINK path: the link trampoline checks CANCEL_REQUESTED on each step
// completion. If set and no prior error, it stashes CANCELLED so the final
// base callback reports cancellation even when the cancel-step call raced
// with natural completion.
//
// For the emulation path: the trampoline checks CANCEL_REQUESTED both before
// calling step_fn and again before submitting the next step, minimizing the
// window for cancel-step races.
//
// Thread-safe: may be called from any thread. The CANCEL_REQUESTED flag is
// read by the poll-thread trampoline after the cancel-step barriers provide
// ordering.
iree_status_t iree_async_sequence_cancel(
    iree_async_proactor_t* proactor, iree_async_sequence_operation_t* sequence);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_SEQUENCE_EMULATION_H_
