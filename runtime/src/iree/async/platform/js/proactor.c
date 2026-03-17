// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/js/proactor.h"

#include <string.h>

#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/js/imports.h"
#include "iree/async/platform/js/token_table.h"
#include "iree/async/util/sequence_emulation.h"

// Forward-declare the vtable (defined at bottom of file).
static const iree_async_proactor_vtable_t iree_async_proactor_js_vtable;

//===----------------------------------------------------------------------===//
// Ready queue
//===----------------------------------------------------------------------===//

// Pushes an operation onto the ready queue tail.
static void iree_async_proactor_js_ready_enqueue(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation) {
  operation->next = NULL;
  if (proactor->ready_tail) {
    proactor->ready_tail->next = operation;
  } else {
    proactor->ready_head = operation;
  }
  proactor->ready_tail = operation;
}

// Pops an operation from the ready queue head. Returns NULL if empty.
static iree_async_operation_t* iree_async_proactor_js_ready_dequeue(
    iree_async_proactor_js_t* proactor) {
  iree_async_operation_t* operation = proactor->ready_head;
  if (operation) {
    proactor->ready_head = operation->next;
    if (!proactor->ready_head) {
      proactor->ready_tail = NULL;
    }
    operation->next = NULL;
  }
  return operation;
}

//===----------------------------------------------------------------------===//
// Linked chain dispatch
//===----------------------------------------------------------------------===//

// Forward-declare submit_one for use by linked continuation dispatch.
static iree_status_t iree_async_proactor_js_submit_one(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation);

// Cancels a linked_next continuation chain by directly invoking callbacks with
// CANCELLED. Cancelled continuations were never submitted, so no resources
// were retained — we call completion_fn directly (not through
// iree_async_proactor_js_complete which would try to release resources).
static void iree_async_proactor_js_cancel_continuation_chain(
    iree_async_operation_t* chain_head) {
  iree_async_operation_t* operation = chain_head;
  while (operation) {
    iree_async_operation_t* next = operation->linked_next;
    operation->linked_next = NULL;
    operation->completion_fn(operation->user_data, operation,
                             iree_status_from_code(IREE_STATUS_CANCELLED),
                             IREE_ASYNC_COMPLETION_FLAG_NONE);
    operation = next;
  }
}

// Dispatches a linked_next continuation chain based on the trigger's status.
// On success: submits the next operation for execution.
// On failure: cancels the entire chain with CANCELLED callbacks.
static void iree_async_proactor_js_dispatch_linked_continuation(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation,
    iree_status_t trigger_status) {
  iree_async_operation_t* continuation = operation->linked_next;
  if (!continuation) return;

  // Detach the chain before potentially recursive submit.
  operation->linked_next = NULL;

  if (iree_status_is_ok(trigger_status)) {
    // Success: submit the continuation (which may itself have linked_next).
    iree_status_t status =
        iree_async_proactor_js_submit_one(proactor, continuation);
    if (!iree_status_is_ok(status)) {
      // Submit failed. Fire continuation's callback with the error,
      // then cancel the rest of the chain.
      iree_async_operation_t* rest = continuation->linked_next;
      continuation->linked_next = NULL;
      continuation->completion_fn(continuation->user_data, continuation, status,
                                  IREE_ASYNC_COMPLETION_FLAG_NONE);
      iree_async_proactor_js_cancel_continuation_chain(rest);
    }
  } else {
    // Trigger failed: cancel the entire continuation chain.
    iree_async_proactor_js_cancel_continuation_chain(continuation);
  }
}

//===----------------------------------------------------------------------===//
// Completion dispatch
//===----------------------------------------------------------------------===//

// Dispatches a completion callback for an operation.
// Releases retained resources, dispatches any linked continuations, then
// invokes the user callback. The linked continuation dispatch must happen
// before the callback because the callback may free the operation.
static void iree_async_proactor_js_complete(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_operation_release_resources(operation);
  iree_async_proactor_js_dispatch_linked_continuation(proactor, operation,
                                                      status);
  operation->completion_fn(operation->user_data, operation, status, flags);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_js_submit_one(
    iree_async_proactor_js_t* proactor, iree_async_operation_t* operation) {
  iree_async_operation_clear_internal_flags(operation);
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP: {
      iree_async_operation_retain_resources(operation);
      IREE_TRACE(operation->submit_time_ns = iree_time_now();)
      // NOPs complete immediately: push to ready queue and schedule drain.
      iree_async_proactor_js_ready_enqueue(proactor, operation);
      iree_async_js_import_schedule_drain();
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_TIMER: {
      iree_async_timer_operation_t* timer =
          (iree_async_timer_operation_t*)operation;
      iree_async_operation_retain_resources(operation);
      IREE_TRACE(operation->submit_time_ns = iree_time_now();)

      // Check if the timer has already expired.
      iree_time_t now = iree_time_now();
      if (timer->deadline_ns <= now) {
        // Already expired: complete immediately via ready queue.
        iree_async_proactor_js_ready_enqueue(proactor, operation);
        iree_async_js_import_schedule_drain();
        return iree_ok_status();
      }

      // Acquire a token for the timer so JS can identify it on completion.
      uint32_t token = UINT32_MAX;
      iree_status_t status = iree_async_js_token_table_acquire(
          &proactor->token_table, operation, &token);
      if (!iree_status_is_ok(status)) {
        iree_async_operation_release_resources(operation);
        return status;
      }

      timer->platform.js.token = token;
      iree_async_js_import_timer_start(token, timer->deadline_ns);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE: {
      iree_async_sequence_operation_t* sequence =
          (iree_async_sequence_operation_t*)operation;
      if (!sequence->step_fn) {
        return iree_async_sequence_submit_as_linked(&proactor->base, sequence);
      } else {
        return iree_async_sequence_emulation_begin(&proactor->sequence_emulator,
                                                   sequence);
      }
    }

    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "JS proactor does not support operation type %u",
                              operation->type);
  }
}

static iree_status_t iree_async_proactor_js_submit(
    iree_async_proactor_t* proactor, iree_async_operation_list_t operations) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);

  // Build linked_next chains from LINKED flags and validate.
  // Operations with LINKED flag point to the next operation in the batch.
  // Only "chain heads" (operations not preceded by a LINKED operation) are
  // submitted; continuations stay in linked_next and are dispatched when the
  // predecessor completes.
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    operation->linked_next = NULL;
    if (!iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }
    // LINKED on last operation is a contract violation.
    if (i + 1 >= operations.count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LINKED flag set on last operation in batch (no successor)");
    }
    operation->linked_next = operations.values[i + 1];
  }

  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];

    // Skip continuation operations — they are held in the predecessor's
    // linked_next and will be submitted when it completes.
    if (i > 0 && iree_any_bit_set(operations.values[i - 1]->flags,
                                  IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }

    iree_status_t status =
        iree_async_proactor_js_submit_one(js_proactor, operation);
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Poll
//===----------------------------------------------------------------------===//

// Drains completions from the JS ring and dispatches callbacks.
static iree_host_size_t iree_async_proactor_js_drain_ring(
    iree_async_proactor_js_t* proactor) {
  uint32_t count = iree_async_js_import_ring_drain(
      proactor->completion_buffer, proactor->completion_buffer_capacity);

  for (uint32_t i = 0; i < count; ++i) {
    iree_async_js_completion_entry_t* entry = &proactor->completion_buffer[i];
    iree_async_operation_t* operation =
        iree_async_js_token_table_lookup(&proactor->token_table, entry->token);
    if (!operation) {
      // Stale completion for a token that was already released (e.g., timer
      // fired after cancel). This is expected and harmless.
      continue;
    }

    // Release the token table slot before dispatching the callback, since the
    // callback may submit new operations that reuse this slot.
    iree_async_js_token_table_release(&proactor->token_table, entry->token);

    // Check if the operation was cancelled while in flight.
    iree_async_operation_internal_flags_t internal_flags =
        iree_atomic_load(&operation->internal_flags, iree_memory_order_relaxed);
    if (internal_flags & IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED) {
      iree_async_proactor_js_complete(
          proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE);
    } else {
      iree_status_t status =
          entry->status_code == 0
              ? iree_ok_status()
              : iree_status_from_code((iree_status_code_t)entry->status_code);
      iree_async_proactor_js_complete(proactor, operation, status,
                                      IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
  }

  return (iree_host_size_t)count;
}

// Drains the ready queue and dispatches callbacks.
static iree_host_size_t iree_async_proactor_js_drain_ready(
    iree_async_proactor_js_t* proactor) {
  iree_host_size_t count = 0;
  iree_async_operation_t* operation;
  while ((operation = iree_async_proactor_js_ready_dequeue(proactor)) != NULL) {
    iree_async_operation_internal_flags_t internal_flags =
        iree_atomic_load(&operation->internal_flags, iree_memory_order_relaxed);
    if (internal_flags & IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED) {
      iree_async_proactor_js_complete(
          proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE);
    } else {
      iree_async_proactor_js_complete(proactor, operation, iree_ok_status(),
                                      IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    ++count;
  }
  return count;
}

static iree_status_t iree_async_proactor_js_poll(
    iree_async_proactor_t* proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);
  iree_host_size_t completed_count = 0;

  // Run progress callbacks (shared infrastructure with other backends).
  completed_count += iree_async_proactor_run_progress(proactor);

  // Drain ready queue (NOPs, expired timers queued during submit).
  completed_count += iree_async_proactor_js_drain_ready(js_proactor);

  // Drain completions from the JS ring.
  completed_count += iree_async_proactor_js_drain_ring(js_proactor);

  // If nothing completed and timeout is not immediate, block for completions.
  if (completed_count == 0 && !iree_timeout_is_immediate(timeout)) {
    iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
    uint32_t wait_result = iree_async_js_import_poll_wait(deadline_ns);
    if (wait_result == 0) {
      // Data available: drain again.
      completed_count += iree_async_proactor_js_drain_ring(js_proactor);
    }
  }

  if (out_completed_count) {
    *out_completed_count = completed_count;
  }
  IREE_TRACE_ZONE_END(z0);

  // If nothing completed, report deadline exceeded. The caller can distinguish
  // "poll found work" (OK with count > 0) from "poll found nothing within
  // the given timeout" (DEADLINE_EXCEEDED with count == 0).
  if (completed_count == 0) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Wake and cancel
//===----------------------------------------------------------------------===//

static void iree_async_proactor_js_wake(iree_async_proactor_t* proactor) {
  iree_async_js_import_wake();
}

static iree_status_t iree_async_proactor_js_cancel(
    iree_async_proactor_t* proactor, iree_async_operation_t* operation) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Set the CANCELLED flag. Since we're single-threaded, relaxed ordering is
  // sufficient.
  iree_atomic_fetch_or(&operation->internal_flags,
                       IREE_ASYNC_JS_OPERATION_INTERNAL_FLAG_CANCELLED,
                       iree_memory_order_relaxed);

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_TIMER: {
      iree_async_timer_operation_t* timer =
          (iree_async_timer_operation_t*)operation;
      uint32_t cancelled =
          iree_async_js_import_timer_cancel(timer->platform.js.token);
      if (cancelled) {
        // Timer was cancelled before firing. Release the token and complete
        // with CANCELLED status immediately.
        iree_async_js_token_table_release(
            &iree_async_proactor_js_cast(proactor)->token_table,
            timer->platform.js.token);
        iree_async_proactor_js_complete(
            iree_async_proactor_js_cast(proactor), operation,
            iree_status_from_code(IREE_STATUS_CANCELLED),
            IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
      // If cancelled == 0, the timer already fired. The completion will arrive
      // through the ring and be dispatched with CANCELLED status because we set
      // the internal flag above.
      break;
    }

    case IREE_ASYNC_OPERATION_TYPE_NOP:
      // NOP is in the ready queue. The CANCELLED flag is checked when it's
      // dequeued during poll.
      break;

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
      IREE_TRACE_ZONE_END(z0);
      return iree_async_sequence_cancel(
          proactor, (iree_async_sequence_operation_t*)operation);

    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "JS proactor does not support cancelling operation type %u",
          operation->type);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_async_proactor_capabilities_t
iree_async_proactor_js_query_capabilities(iree_async_proactor_t* proactor) {
  return iree_async_proactor_js_cast(proactor)->capabilities;
}

//===----------------------------------------------------------------------===//
// Unavailable vtable methods
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_js_create_socket(
    iree_async_proactor_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support sockets");
}

static iree_status_t iree_async_proactor_js_import_socket(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support sockets");
}

static void iree_async_proactor_js_destroy_socket(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket) {
  // Should never be called since create/import return UNAVAILABLE.
  IREE_ASSERT(false, "destroy_socket called on JS proactor");
}

static iree_status_t iree_async_proactor_js_import_file(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support files");
}

static void iree_async_proactor_js_destroy_file(iree_async_proactor_t* proactor,
                                                iree_async_file_t* file) {
  IREE_ASSERT(false, "destroy_file called on JS proactor");
}

static iree_status_t iree_async_proactor_js_create_event(
    iree_async_proactor_t* proactor, iree_async_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support events yet");
}

static void iree_async_proactor_js_destroy_event(
    iree_async_proactor_t* proactor, iree_async_event_t* event) {
  IREE_ASSERT(false, "destroy_event called on JS proactor");
}

static iree_status_t iree_async_proactor_js_register_event_source(
    iree_async_proactor_t* proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support event sources");
}

static void iree_async_proactor_js_unregister_event_source(
    iree_async_proactor_t* proactor, iree_async_event_source_t* event_source) {
  IREE_ASSERT(false, "unregister_event_source called on JS proactor");
}

static iree_status_t iree_async_proactor_js_create_notification(
    iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support notifications yet");
}

static iree_status_t iree_async_proactor_js_create_notification_shared(
    iree_async_proactor_t* proactor,
    const iree_async_notification_shared_options_t* options,
    iree_async_notification_t** out_notification) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support shared notifications");
}

static void iree_async_proactor_js_destroy_notification(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification) {
  IREE_ASSERT(false, "destroy_notification called on JS proactor");
}

static void iree_async_proactor_js_notification_signal(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification,
    int32_t wake_count) {
  IREE_ASSERT(false, "notification_signal called on JS proactor");
}

static bool iree_async_proactor_js_notification_wait(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification,
    iree_timeout_t timeout) {
  IREE_ASSERT(false, "notification_wait called on JS proactor");
  return false;
}

static iree_status_t iree_async_proactor_js_register_relay(
    iree_async_proactor_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support relays");
}

static void iree_async_proactor_js_unregister_relay(
    iree_async_proactor_t* proactor, iree_async_relay_t* relay) {
  IREE_ASSERT(false, "unregister_relay called on JS proactor");
}

static iree_status_t iree_async_proactor_js_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support buffer registration");
}

static iree_status_t iree_async_proactor_js_register_dmabuf(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support dmabuf");
}

static void iree_async_proactor_js_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  IREE_ASSERT(false, "unregister_buffer called on JS proactor");
}

static iree_status_t iree_async_proactor_js_register_slab(
    iree_async_proactor_t* proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support slabs");
}

static iree_status_t iree_async_proactor_js_import_fence(
    iree_async_proactor_t* proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support fences");
}

static iree_status_t iree_async_proactor_js_export_fence(
    iree_async_proactor_t* proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support fences");
}

static void iree_async_proactor_js_set_message_callback(
    iree_async_proactor_t* proactor,
    iree_async_proactor_message_callback_t callback) {
  // Accept but discard: messaging is not supported.
}

static iree_status_t iree_async_proactor_js_send_message(
    iree_async_proactor_t* target, uint64_t message_data) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support messaging");
}

static iree_status_t iree_async_proactor_js_subscribe_signal(
    iree_async_proactor_t* proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback,
    iree_async_signal_subscription_t** out_subscription) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "JS proactor does not support signals");
}

static void iree_async_proactor_js_unsubscribe_signal(
    iree_async_proactor_t* proactor,
    iree_async_signal_subscription_t* subscription) {
  IREE_ASSERT(false, "unsubscribe_signal called on JS proactor");
}

//===----------------------------------------------------------------------===//
// Lifecycle and vtable
//===----------------------------------------------------------------------===//

static void iree_async_proactor_js_destroy(iree_async_proactor_t* proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_js_t* js_proactor = iree_async_proactor_js_cast(proactor);
  iree_allocator_t allocator = proactor->allocator;

  // The ready queue should be empty by the time we destroy.
  IREE_ASSERT(js_proactor->ready_head == NULL,
              "JS proactor destroyed with operations in ready queue");

  iree_async_js_token_table_deinitialize(&js_proactor->token_table);
  iree_allocator_free(allocator, js_proactor);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_proactor_create_js(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  // Token table capacity from options, with default fallback.
  iree_host_size_t token_table_capacity = options.max_concurrent_operations;
  if (token_table_capacity == 0) {
    token_table_capacity = IREE_ASYNC_JS_DEFAULT_TOKEN_TABLE_CAPACITY;
  }

  iree_host_size_t completion_buffer_capacity =
      IREE_ASYNC_JS_DEFAULT_COMPLETION_BUFFER_CAPACITY;

  // Calculate allocation layout: [proactor | completion_buffer[]]
  iree_host_size_t total_size = 0;
  iree_host_size_t completion_buffer_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(iree_async_proactor_js_t), &total_size,
                             IREE_STRUCT_FIELD(completion_buffer_capacity,
                                               iree_async_js_completion_entry_t,
                                               &completion_buffer_offset)));

  // Single allocation for proactor + completion buffer.
  iree_async_proactor_js_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&proactor));
  memset(proactor, 0, total_size);

  // Initialize base proactor.
  iree_async_proactor_initialize(&iree_async_proactor_js_vtable,
                                 options.debug_name, allocator,
                                 &proactor->base);

  // Initialize token table (separately allocated by the table itself).
  iree_status_t status = iree_async_js_token_table_initialize(
      token_table_capacity, allocator, &proactor->token_table);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(allocator, proactor);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Set up completion buffer pointer into trailing data.
  proactor->completion_buffer =
      (iree_async_js_completion_entry_t*)((uint8_t*)proactor +
                                          completion_buffer_offset);
  proactor->completion_buffer_capacity = completion_buffer_capacity;

  // Ready queue starts empty.
  proactor->ready_head = NULL;
  proactor->ready_tail = NULL;

  // Initialize sequence emulator for SEQUENCE operation support. Uses the
  // public submit_one API which re-enters through the vtable submit path.
  iree_async_sequence_emulator_initialize(&proactor->sequence_emulator,
                                          &proactor->base,
                                          iree_async_proactor_submit_one);

  // Store capabilities. The JS proactor supports absolute timeouts natively
  // (JS setTimeout uses absolute deadlines internally). LINKED_OPERATIONS is
  // emulated in userspace via linked_next chains — the proactor builds a
  // linked list during submit and dispatches continuations on completion.
  iree_async_proactor_capabilities_t supported_capabilities =
      IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT |
      IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS;
  proactor->capabilities =
      options.allowed_capabilities & supported_capabilities;

  *out_proactor = &proactor->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_async_proactor_vtable_t iree_async_proactor_js_vtable = {
    .destroy = iree_async_proactor_js_destroy,
    .query_capabilities = iree_async_proactor_js_query_capabilities,
    .submit = iree_async_proactor_js_submit,
    .poll = iree_async_proactor_js_poll,
    .wake = iree_async_proactor_js_wake,
    .cancel = iree_async_proactor_js_cancel,
    .create_socket = iree_async_proactor_js_create_socket,
    .import_socket = iree_async_proactor_js_import_socket,
    .destroy_socket = iree_async_proactor_js_destroy_socket,
    .import_file = iree_async_proactor_js_import_file,
    .destroy_file = iree_async_proactor_js_destroy_file,
    .create_event = iree_async_proactor_js_create_event,
    .destroy_event = iree_async_proactor_js_destroy_event,
    .register_event_source = iree_async_proactor_js_register_event_source,
    .unregister_event_source = iree_async_proactor_js_unregister_event_source,
    .create_notification = iree_async_proactor_js_create_notification,
    .create_notification_shared =
        iree_async_proactor_js_create_notification_shared,
    .destroy_notification = iree_async_proactor_js_destroy_notification,
    .notification_signal = iree_async_proactor_js_notification_signal,
    .notification_wait = iree_async_proactor_js_notification_wait,
    .register_relay = iree_async_proactor_js_register_relay,
    .unregister_relay = iree_async_proactor_js_unregister_relay,
    .register_buffer = iree_async_proactor_js_register_buffer,
    .register_dmabuf = iree_async_proactor_js_register_dmabuf,
    .unregister_buffer = iree_async_proactor_js_unregister_buffer,
    .register_slab = iree_async_proactor_js_register_slab,
    .import_fence = iree_async_proactor_js_import_fence,
    .export_fence = iree_async_proactor_js_export_fence,
    .set_message_callback = iree_async_proactor_js_set_message_callback,
    .send_message = iree_async_proactor_js_send_message,
    .subscribe_signal = iree_async_proactor_js_subscribe_signal,
    .unsubscribe_signal = iree_async_proactor_js_unsubscribe_signal,
};
