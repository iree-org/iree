// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/frame_sender.h"

#include <string.h>

iree_status_t iree_net_frame_sender_initialize(
    iree_net_frame_sender_t* sender, iree_net_carrier_t* carrier,
    iree_async_buffer_pool_t* header_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_allocator_t context_allocator, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(sender);
  IREE_ASSERT_ARGUMENT(carrier);
  IREE_ASSERT_ARGUMENT(header_pool);

  memset(sender, 0, sizeof(*sender));
  sender->carrier = carrier;
  sender->header_pool = header_pool;
  sender->callback = callback;
  sender->has_batch_lease = false;
  sender->batch_used = 0;
  iree_atomic_store(&sender->sends_in_flight, 0, iree_memory_order_relaxed);
  sender->context_allocator = context_allocator;
  sender->host_allocator = host_allocator;

  return iree_ok_status();
}

void iree_net_frame_sender_deinitialize(iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);

  // Assert no sends in flight - caller must drain completions first.
  int32_t pending =
      iree_atomic_load(&sender->sends_in_flight, iree_memory_order_acquire);
  IREE_ASSERT(pending == 0, "frame_sender deinitialize with %d sends in flight",
              pending);

  // Release any pending batch lease.
  if (sender->has_batch_lease) {
    iree_async_buffer_lease_release(&sender->batch_lease);
    sender->has_batch_lease = false;
  }

  memset(sender, 0, sizeof(*sender));
}

// Allocates and initializes a send context.
static iree_status_t iree_net_frame_sender_allocate_context(
    iree_net_frame_sender_t* sender, uint64_t operation_user_data,
    iree_net_frame_send_context_t** out_context) {
  iree_net_frame_send_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      sender->context_allocator, sizeof(*context), (void**)&context));
  context->sender = sender;
  context->operation_user_data = operation_user_data;
  context->span_count = 0;
  *out_context = context;
  return iree_ok_status();
}

// Submits a send context to the carrier.
static iree_status_t iree_net_frame_sender_submit(
    iree_net_frame_sender_t* sender, iree_net_frame_send_context_t* context) {
  iree_net_send_params_t params = {
      .data = iree_async_span_list_make(context->spans, context->span_count),
      .flags = IREE_NET_SEND_FLAG_NONE,
      .user_data = (uint64_t)(uintptr_t)context,
  };

  iree_atomic_fetch_add(&sender->sends_in_flight, 1, iree_memory_order_relaxed);

  iree_status_t status = iree_net_carrier_send(sender->carrier, &params);
  if (!iree_status_is_ok(status)) {
    // Use release ordering to match completion path - ensures the decrement is
    // visible to other threads checking has_pending/pending_count.
    iree_atomic_fetch_sub(&sender->sends_in_flight, 1,
                          iree_memory_order_release);
  }
  return status;
}

iree_status_t iree_net_frame_sender_send(iree_net_frame_sender_t* sender,
                                         iree_const_byte_span_t header,
                                         iree_async_span_list_t payload,
                                         uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(sender);

  // NOTE: This function is thread-safe and may be called from any thread.
  // It does NOT auto-flush batched frames - if the caller is mixing send()
  // with queue()/flush() on the proactor thread and needs ordering, they
  // must call flush() explicitly before send().

  // Validate span count: 1 (header) + payload.count <= min(MAX, carrier limit).
  iree_host_size_t total_spans = 1 + payload.count;
  iree_host_size_t max_spans = IREE_NET_FRAME_SENDER_MAX_SPANS;
  if (sender->carrier->max_iov > 0 && sender->carrier->max_iov < max_spans) {
    max_spans = sender->carrier->max_iov;
  }
  if (total_spans > max_spans) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "send requires %" PRIhsz
                            " spans but max is %" PRIhsz,
                            total_spans, max_spans);
  }

  // Acquire header buffer from pool.
  iree_async_buffer_lease_t header_lease;
  iree_status_t status =
      iree_async_buffer_pool_acquire(sender->header_pool, &header_lease);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Verify header fits in pool buffer.
  if (header.data_length > header_lease.span.length) {
    iree_async_buffer_lease_release(&header_lease);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "header size %" PRIhsz
                            " exceeds pool buffer size %" PRIhsz,
                            header.data_length, header_lease.span.length);
  }

  // Copy header into pool buffer (guard against NULL for zero-length headers).
  if (header.data_length > 0) {
    memcpy(iree_async_span_ptr(header_lease.span), header.data,
           header.data_length);
  }

  // Allocate send context.
  iree_net_frame_send_context_t* context = NULL;
  status = iree_net_frame_sender_allocate_context(sender, operation_user_data,
                                                  &context);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&header_lease);
    return status;
  }

  // Transfer header lease to context.
  context->buffer_lease = header_lease;

  // Build span list: [header_span, payload_spans...].
  context->spans[0] = iree_async_span_make(
      header_lease.span.region, header_lease.span.offset, header.data_length);
  for (iree_host_size_t i = 0; i < payload.count; ++i) {
    context->spans[1 + i] = payload.values[i];
  }
  context->span_count = total_spans;

  // Submit to carrier.
  status = iree_net_frame_sender_submit(sender, context);
  if (!iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&context->buffer_lease);
    iree_allocator_free(sender->context_allocator, context);
    return status;
  }

  return iree_ok_status();
}

iree_status_t iree_net_frame_sender_queue(iree_net_frame_sender_t* sender,
                                          iree_const_byte_span_t frame) {
  IREE_ASSERT_ARGUMENT(sender);

  // Acquire batch buffer if we don't have one.
  if (!sender->has_batch_lease) {
    iree_status_t status = iree_async_buffer_pool_acquire(sender->header_pool,
                                                          &sender->batch_lease);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    sender->has_batch_lease = true;
    sender->batch_used = 0;
  }

  // Check if frame fits in remaining space.
  iree_host_size_t remaining =
      sender->batch_lease.span.length - sender->batch_used;
  if (frame.data_length > remaining) {
    // Batch buffer full - caller should flush first.
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  // Copy frame to batch buffer (guard against NULL for zero-length frames).
  if (frame.data_length > 0) {
    uint8_t* dest =
        iree_async_span_ptr(sender->batch_lease.span) + sender->batch_used;
    memcpy(dest, frame.data, frame.data_length);
  }
  sender->batch_used += frame.data_length;

  return iree_ok_status();
}

iree_status_t iree_net_frame_sender_flush(iree_net_frame_sender_t* sender,
                                          uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(sender);

  // No-op if nothing queued.
  if (!sender->has_batch_lease || sender->batch_used == 0) {
    return iree_ok_status();
  }

  // Allocate send context.
  iree_net_frame_send_context_t* context = NULL;
  iree_status_t status = iree_net_frame_sender_allocate_context(
      sender, operation_user_data, &context);
  if (!iree_status_is_ok(status)) {
    // Keep batch data for retry.
    return status;
  }

  // Transfer batch lease to context.
  context->buffer_lease = sender->batch_lease;

  // Build single-span list for the used portion of batch buffer.
  context->spans[0] =
      iree_async_span_make(sender->batch_lease.span.region,
                           sender->batch_lease.span.offset, sender->batch_used);
  context->span_count = 1;

  // Submit to carrier.
  status = iree_net_frame_sender_submit(sender, context);
  if (!iree_status_is_ok(status)) {
    // Submission failed - keep batch data for retry.
    // DO NOT release batch_lease, it's still in sender.
    iree_allocator_free(sender->context_allocator, context);
    return status;
  }

  // Success - ownership transferred to context.
  sender->has_batch_lease = false;
  sender->batch_used = 0;

  return iree_ok_status();
}

iree_host_size_t iree_net_frame_sender_queued_bytes(
    const iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);
  return sender->batch_used;
}

bool iree_net_frame_sender_has_pending(const iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);
  return iree_atomic_load(&((iree_net_frame_sender_t*)sender)->sends_in_flight,
                          iree_memory_order_acquire) > 0;
}

int32_t iree_net_frame_sender_pending_count(
    const iree_net_frame_sender_t* sender) {
  IREE_ASSERT_ARGUMENT(sender);
  return iree_atomic_load(&((iree_net_frame_sender_t*)sender)->sends_in_flight,
                          iree_memory_order_acquire);
}

void iree_net_frame_sender_handle_completion(
    iree_net_frame_send_context_t* context, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(context);
  iree_net_frame_sender_t* sender = context->sender;

  // Release buffer lease back to pool.
  iree_async_buffer_lease_release(&context->buffer_lease);

  // Decrement in-flight count.
  iree_atomic_fetch_sub(&sender->sends_in_flight, 1, iree_memory_order_release);

  // Fire user callback.
  if (sender->callback.fn) {
    sender->callback.fn(sender->callback.user_data,
                        context->operation_user_data, status);
  }

  // Free context.
  iree_allocator_free(sender->context_allocator, context);
}
