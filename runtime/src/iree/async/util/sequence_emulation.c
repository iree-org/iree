// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/sequence_emulation.h"

//===----------------------------------------------------------------------===//
// LINK path (step_fn == NULL)
//===----------------------------------------------------------------------===//

// Completion callback installed on every step during submit_as_linked.
// Coordinates completion counting across all steps and fires the sequence's
// base callback exactly once when all step CQEs have been processed.
//
// CQE ordering varies by backend:
//   io_uring: error CQE fires before cancelled CQEs for downstream steps.
//   POSIX: dispatch_linked_continuation cancels downstream steps (firing their
//          callbacks with CANCELLED) before the triggering step's callback.
// The SAW_ERROR flag and internal.stashed_error handle both orderings.
static void iree_async_sequence_link_trampoline(
    void* user_data, iree_async_operation_t* step, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)step;
  (void)flags;
  iree_async_sequence_operation_t* sequence =
      (iree_async_sequence_operation_t*)user_data;

  // Capture the first non-CANCELLED error. Subsequent errors (which should not
  // occur in well-formed linked chains, but handled defensively) are discarded.
  bool is_real_error = !iree_status_is_ok(status) &&
                       iree_status_code(status) != IREE_STATUS_CANCELLED;
  if (is_real_error) {
    iree_async_operation_internal_flags_t seq_flags =
        iree_async_operation_load_internal_flags(&sequence->base);
    if (!iree_any_bit_set(seq_flags, IREE_ASYNC_SEQUENCE_INTERNAL_SAW_ERROR)) {
      iree_async_operation_set_internal_flags(
          &sequence->base, IREE_ASYNC_SEQUENCE_INTERNAL_SAW_ERROR);
      // Ownership of |status| transfers to the stash.
      sequence->internal.stashed_error = status;
      status = iree_ok_status();
    } else {
      iree_status_ignore(status);
      status = iree_ok_status();
    }
  }

  // Handle cancel-step race: if cancel was requested but this step completed
  // with OK (the cancel-step call was a no-op because the step completed
  // between the cancel thread's read of current_step and the cancel
  // submission), record CANCELLED as the final status. Subsequent steps in
  // the kernel chain may still execute, but the sequence result is correct.
  if (iree_status_is_ok(status)) {
    iree_async_operation_internal_flags_t cancel_check_flags =
        iree_async_operation_load_internal_flags(&sequence->base);
    if (iree_any_bit_set(cancel_check_flags,
                         IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED) &&
        !iree_any_bit_set(cancel_check_flags,
                          IREE_ASYNC_SEQUENCE_INTERNAL_SAW_ERROR)) {
      iree_async_operation_set_internal_flags(
          &sequence->base, IREE_ASYNC_SEQUENCE_INTERNAL_SAW_ERROR);
      sequence->internal.stashed_error =
          iree_status_from_code(IREE_STATUS_CANCELLED);
    }
  }

  ++sequence->current_step;

  if (sequence->current_step == sequence->step_count) {
    // All step CQEs processed. Determine final status and fire base callback.
    iree_status_t final_status;
    if (iree_any_bit_set(
            iree_async_operation_load_internal_flags(&sequence->base),
            IREE_ASYNC_SEQUENCE_INTERNAL_SAW_ERROR)) {
      // Use the captured error. Discard the current status (OK or CANCELLED).
      iree_status_ignore(status);
      final_status = sequence->internal.stashed_error;
      sequence->internal.stashed_error = NULL;
    } else {
      // All steps succeeded or the sequence was cancelled. The last step's
      // status carries the right answer: OK if all succeeded, CANCELLED if
      // the sequence (or a predecessor in the linked chain) was cancelled.
      final_status = status;
    }
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 final_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
  } else {
    // More step CQEs pending. Discard intermediate status.
    iree_status_ignore(status);
  }
}

iree_status_t iree_async_sequence_submit_as_linked(
    iree_async_proactor_t* proactor,
    iree_async_sequence_operation_t* sequence) {
  // LINKED on a SEQUENCE itself is not supported: the expansion logic does not
  // wire the sequence's last step to the next batch operation, so the
  // continuation would be silently dropped.
  if (iree_any_bit_set(sequence->base.flags,
                       IREE_ASYNC_OPERATION_FLAG_LINKED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LINKED flag on SEQUENCE operation is not supported; use the "
        "sequence's steps array to chain operations");
  }

  // Zero-step edge case: complete immediately.
  if (sequence->step_count == 0) {
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 iree_ok_status(),
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
    return iree_ok_status();
  }

  // Reset sequence state.
  sequence->current_step = 0;
  iree_async_operation_clear_internal_flags(&sequence->base);
  sequence->internal.stashed_error = NULL;

  // Save original step state before installing trampolines. On submit failure
  // the steps must be restored so the caller can retry or use them
  // independently.
  typedef struct {
    iree_async_completion_fn_t completion_fn;
    void* user_data;
    iree_async_operation_flags_t flags;
  } iree_async_step_saved_state_t;
  iree_async_step_saved_state_t* saved =
      (iree_async_step_saved_state_t*)iree_alloca(
          sequence->step_count * sizeof(iree_async_step_saved_state_t));

  // Install link trampolines on all steps and set LINKED flags.
  for (iree_host_size_t i = 0; i < sequence->step_count; ++i) {
    iree_async_operation_t* step = sequence->steps[i];
    saved[i].completion_fn = step->completion_fn;
    saved[i].user_data = step->user_data;
    saved[i].flags = step->flags;
    step->completion_fn = iree_async_sequence_link_trampoline;
    step->user_data = sequence;
    if (i + 1 < sequence->step_count) {
      step->flags |= IREE_ASYNC_OPERATION_FLAG_LINKED;
    } else {
      // Last step must NOT have LINKED (contract of linked batches).
      step->flags &= ~IREE_ASYNC_OPERATION_FLAG_LINKED;
    }
  }

  // Submit the steps as a linked batch through the proactor's vtable.
  // This re-enters the backend's submit function. The steps are not SEQUENCE
  // type, so the re-entered call processes them through normal linked handling.
  iree_async_operation_list_t step_list = {sequence->steps,
                                           sequence->step_count};
  iree_status_t status = iree_async_proactor_submit(proactor, step_list);
  if (!iree_status_is_ok(status)) {
    // Submit failed (e.g., SQ full). Restore original step state so the caller
    // can retry or use the steps independently.
    for (iree_host_size_t i = 0; i < sequence->step_count; ++i) {
      iree_async_operation_t* step = sequence->steps[i];
      step->completion_fn = saved[i].completion_fn;
      step->user_data = saved[i].user_data;
      step->flags = saved[i].flags;
      step->linked_next =
          NULL;  // Clear chain links set by the re-entered submit.
    }
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Emulation path (step_fn != NULL)
//===----------------------------------------------------------------------===//

// Completion callback installed on each step during emulation.
// Recovers the sequence and emulator, then advances to the next step.
static void iree_async_sequence_emulation_trampoline(
    void* user_data, iree_async_operation_t* step, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)step;
  (void)flags;
  iree_async_sequence_operation_t* sequence =
      (iree_async_sequence_operation_t*)user_data;
  iree_async_sequence_emulator_t* emulator =
      (iree_async_sequence_emulator_t*)sequence->internal.emulator;
  iree_async_sequence_emulation_step_completed(emulator, sequence, status);
}

// Submits the next step in an emulated sequence.
// Installs the emulation trampoline on the step and submits it.
static iree_status_t iree_async_sequence_emulation_submit_step(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence) {
  iree_async_operation_t* step = sequence->steps[sequence->current_step];
  step->completion_fn = iree_async_sequence_emulation_trampoline;
  step->user_data = sequence;
  // Clear LINKED flag on individual steps in emulation mode — each step is
  // submitted independently. The sequence logic handles ordering.
  step->flags &= ~IREE_ASYNC_OPERATION_FLAG_LINKED;
  return emulator->submit_fn(emulator->proactor, step);
}

iree_status_t iree_async_sequence_emulation_begin(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence) {
  // LINKED on a SEQUENCE itself is not supported: the expansion logic does not
  // wire the sequence's last step to the next batch operation, so the
  // continuation would be silently dropped.
  if (iree_any_bit_set(sequence->base.flags,
                       IREE_ASYNC_OPERATION_FLAG_LINKED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LINKED flag on SEQUENCE operation is not supported; use the "
        "sequence's steps array to chain operations");
  }

  // Zero-step edge case: complete immediately.
  if (sequence->step_count == 0) {
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 iree_ok_status(),
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
    return iree_ok_status();
  }

  // Reset sequence state and stash the emulator pointer.
  sequence->current_step = 0;
  iree_async_operation_clear_internal_flags(&sequence->base);
  sequence->internal.emulator = emulator;

  // Submit step 0.
  iree_status_t status =
      iree_async_sequence_emulation_submit_step(emulator, sequence);
  if (!iree_status_is_ok(status)) {
    // Step 0 submission failed. Fire base callback with the error and return
    // OK: the callback has consumed the operation (the caller must not double-
    // handle via both the callback and the return value).
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    return iree_ok_status();
  }
  return iree_ok_status();
}

void iree_async_sequence_emulation_step_completed(
    iree_async_sequence_emulator_t* emulator,
    iree_async_sequence_operation_t* sequence, iree_status_t step_status) {
  // Step failure: abort immediately.
  if (!iree_status_is_ok(step_status)) {
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 step_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    return;
  }

  // Advance to the next step.
  ++sequence->current_step;

  // Check if cancel was requested between steps.
  if (iree_any_bit_set(
          iree_async_operation_load_internal_flags(&sequence->base),
          IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED)) {
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 iree_status_from_code(IREE_STATUS_CANCELLED),
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
    return;
  }

  // Determine the completed step and the next step (NULL if this was the last).
  iree_async_operation_t* completed_step =
      sequence->steps[sequence->current_step - 1];
  iree_async_operation_t* next_step =
      (sequence->current_step < sequence->step_count)
          ? sequence->steps[sequence->current_step]
          : NULL;

  // Call the inter-step callback if provided.
  if (sequence->step_fn) {
    iree_status_t step_fn_status =
        sequence->step_fn(sequence->base.user_data, completed_step, next_step);
    if (!iree_status_is_ok(step_fn_status)) {
      // step_fn vetoed continuation. Abort with its error.
      sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                   step_fn_status,
                                   IREE_ASYNC_COMPLETION_FLAG_NONE);
      return;
    }
  }

  // All steps complete?
  if (sequence->current_step == sequence->step_count) {
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 iree_ok_status(),
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
    return;
  }

  // Re-check cancel after step_fn. Cancel may have arrived during step_fn
  // execution (which can take arbitrary time). Without this check, the next
  // step would be submitted despite the cancel request.
  if (iree_any_bit_set(
          iree_async_operation_load_internal_flags(&sequence->base),
          IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED)) {
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 iree_status_from_code(IREE_STATUS_CANCELLED),
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
    return;
  }

  // Submit the next step.
  iree_status_t submit_status =
      iree_async_sequence_emulation_submit_step(emulator, sequence);
  if (!iree_status_is_ok(submit_status)) {
    // Next step submission failed. Abort with the submission error.
    sequence->base.completion_fn(sequence->base.user_data, &sequence->base,
                                 submit_status,
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
  }
}

//===----------------------------------------------------------------------===//
// Cancellation
//===----------------------------------------------------------------------===//

iree_status_t iree_async_sequence_cancel(
    iree_async_proactor_t* proactor,
    iree_async_sequence_operation_t* sequence) {
  iree_async_operation_set_internal_flags(
      &sequence->base, IREE_ASYNC_SEQUENCE_INTERNAL_CANCEL_REQUESTED);

  // Best-effort cancel of the current in-flight step. If this fails (step
  // already completed due to a race between the cancel thread and the poll
  // thread), the CANCEL_REQUESTED flag ensures the trampolines (both LINK
  // and emulation) produce CANCELLED as the final sequence status.
  iree_host_size_t current = sequence->current_step;
  if (current < sequence->step_count) {
    iree_status_ignore(
        iree_async_proactor_cancel(proactor, sequence->steps[current]));
  }
  return iree_ok_status();
}
