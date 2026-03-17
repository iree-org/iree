// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Frame sender: scatter-gather send utility for channels.
//
// iree_net_frame_sender_t is a formatting utility that assembles scatter-gather
// lists from headers and payloads, manages buffer pool allocations, and tracks
// in-flight sends. It sits below the channel layer and above the carrier layer.
//
// Channels embed a frame_sender and use it for all send operations. The channel
// handles protocol semantics (frame types, stream IDs, etc.) while the
// frame_sender handles the mechanics of constructing and submitting sends.
//
// ## Ownership model
//
// The frame_sender does NOT own the carrier or buffer pool - they must outlive
// the sender. The sender owns in-flight send contexts until completion.
//
// ## Callback contract
//
// Matches the carrier pattern:
//   - If send()/flush() returns non-OK: callback is NEVER called
//   - If send()/flush() returns OK: callback WILL be called (errors via
//   callback)
//
// ## Send modes
//
// Two modes are supported:
//
// 1. **send()**: Scatter-gather send with header from pool, payload zero-copy.
//    Header is copied to a pool buffer, payload spans are passed directly.
//    Best for large payloads where zero-copy matters.
//
// 2. **queue() + flush()**: Batched send for small frames.
//    Frames are copied to a batch buffer, then sent together on flush().
//    Reduces syscall overhead for many small control messages.
//
// When mixing modes, send() automatically flushes pending batched frames first
// to maintain ordering.
//
// ## Backpressure
//
// Frame sender propagates RESOURCE_EXHAUSTED from the carrier (or pool). It
// does NOT manage backpressure policy - the embedding channel interprets
// RESOURCE_EXHAUSTED and handles drain callbacks.
//
// ## Thread safety
//
// **send()** is thread-safe and may be called from any thread. This is the
// primary API for application threads sending data. The underlying carrier
// is assumed to be thread-safe for sends (io_uring, RDMA, etc.). If a carrier
// implementation is not naturally thread-safe, it handles serialization
// internally - this is not the channel's or frame_sender's concern.
//
// **queue() and flush()** are NOT thread-safe and must only be called from
// a single thread (the proactor thread). These are optimizations for batching
// small control messages where the proactor is assembling multiple frames.
//
// **Mixing send() with queue()/flush()**: If the proactor thread uses both
// APIs and needs ordering between them, it must call flush() explicitly
// before send(). send() does NOT auto-flush to maintain thread-safety.
// Off-thread callers only use send() and have no batched data to flush.
//
// **has_pending() and pending_count()** use atomic operations and are safe
// to call from any thread.
//
// **handle_completion()** may be called from the carrier's completion thread,
// which may differ from the submission thread. It uses atomic operations for
// the in-flight count and releases the buffer lease (pool is thread-safe).

#ifndef IREE_NET_CHANNEL_UTIL_FRAME_SENDER_H_
#define IREE_NET_CHANNEL_UTIL_FRAME_SENDER_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Frame sender
//===----------------------------------------------------------------------===//

typedef struct iree_net_frame_sender_t iree_net_frame_sender_t;
typedef struct iree_net_frame_send_context_t iree_net_frame_send_context_t;

// Maximum scatter-gather spans per send operation (header + payloads).
// Validated against carrier->max_iov at send time.
#define IREE_NET_FRAME_SENDER_MAX_SPANS 8

// Completion callback fired when a send completes (success or failure).
// Channel uses this for resource cleanup and drain signaling.
//
// |callback_user_data| is from the callback struct (set at initialize).
// |operation_user_data| is from the send()/flush() call (identifies the op).
// |status| is OK on success, or an error status on failure.
typedef void (*iree_net_frame_send_complete_fn_t)(void* callback_user_data,
                                                  uint64_t operation_user_data,
                                                  iree_status_t status);

// Callback struct for send completions.
typedef struct iree_net_frame_send_complete_callback_t {
  iree_net_frame_send_complete_fn_t fn;
  void* user_data;
} iree_net_frame_send_complete_callback_t;

// Per-send context allocated from context_allocator.
// Holds state needed until completion callback fires.
//
// Layout optimized for cache: hot fields (sender, lease, span_count, first
// span) fit in first cache line for common single-span case.
struct iree_net_frame_send_context_t {
  iree_net_frame_sender_t* sender;
  iree_async_buffer_lease_t buffer_lease;  // Header or batch buffer.
  uint64_t operation_user_data;
  iree_host_size_t span_count;
  iree_async_span_t spans[IREE_NET_FRAME_SENDER_MAX_SPANS];
};

// Embeddable sender structure.
// Use initialize/deinitialize pattern - do not allocate separately.
struct iree_net_frame_sender_t {
  iree_net_carrier_t* carrier;            // Not owned.
  iree_async_buffer_pool_t* header_pool;  // Not owned.

  // Completion callback (provided by channel).
  iree_net_frame_send_complete_callback_t callback;

  // Batch buffer for small frames (queue/flush mode).
  iree_async_buffer_lease_t batch_lease;
  bool has_batch_lease;
  iree_host_size_t batch_used;

  // Tracking.
  iree_atomic_int32_t sends_in_flight;

  // Allocators.
  iree_allocator_t context_allocator;  // For send contexts (may be pooled).
  iree_allocator_t host_allocator;     // For other allocations.
};

// Initializes a frame sender in pre-allocated memory.
//
// |carrier| and |header_pool| must outlive the sender.
// |callback| is fired for each completed send operation.
// |context_allocator| is used for per-send context structs (can be pool-backed
// for performance).
// |host_allocator| is used for other dynamic allocations.
iree_status_t iree_net_frame_sender_initialize(
    iree_net_frame_sender_t* sender, iree_net_carrier_t* carrier,
    iree_async_buffer_pool_t* header_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_allocator_t context_allocator, iree_allocator_t host_allocator);

// Deinitializes a frame sender.
// Asserts that no sends are in flight. Caller must drain completions first.
void iree_net_frame_sender_deinitialize(iree_net_frame_sender_t* sender);

// Sends a frame with scatter-gather. THREAD-SAFE.
//
// This is the primary send API and may be called from any thread. The
// underlying carrier and buffer pool are assumed to be thread-safe.
//
// |header| is copied into a pool buffer (small, unavoidable copy).
// |payload| is a span list sent directly (zero-copy if registered memory).
// |operation_user_data| is passed to the completion callback.
//
// Does NOT auto-flush batched frames. If the proactor thread is mixing
// send() with queue()/flush() and needs ordering, call flush() first.
//
// Returns OK if the operation was submitted (callback will fire).
// Returns RESOURCE_EXHAUSTED if carrier is backpressured or pool is empty.
// Returns OUT_OF_RANGE if 1 + payload.count > min(MAX_SPANS, carrier->max_iov).
//
// On non-OK return, the callback is NOT called.
iree_status_t iree_net_frame_sender_send(iree_net_frame_sender_t* sender,
                                         iree_const_byte_span_t header,
                                         iree_async_span_list_t payload,
                                         uint64_t operation_user_data);

// Queues a small frame for batched send. NOT THREAD-SAFE.
//
// Must only be called from the proactor thread. This is an optimization for
// batching small control messages to reduce syscall overhead.
//
// |frame| (header + payload combined) is copied into the batch buffer.
// Call flush() to actually send the batched data.
//
// Returns OK if the frame was queued successfully.
// Returns RESOURCE_EXHAUSTED if the batch buffer is full (call flush first).
iree_status_t iree_net_frame_sender_queue(iree_net_frame_sender_t* sender,
                                          iree_const_byte_span_t frame);

// Flushes all queued frames as a single send operation. NOT THREAD-SAFE.
//
// Must only be called from the proactor thread.
//
// |operation_user_data| is passed to the completion callback for this batch.
//
// Returns OK if the operation was submitted or if nothing was queued.
// Returns RESOURCE_EXHAUSTED if carrier is backpressured.
//
// On RESOURCE_EXHAUSTED, the batch data is preserved and flush() can be
// retried later.
//
// If nothing is queued, returns OK immediately without firing a callback.
iree_status_t iree_net_frame_sender_flush(iree_net_frame_sender_t* sender,
                                          uint64_t operation_user_data);

// Returns the number of bytes currently queued for batching.
iree_host_size_t iree_net_frame_sender_queued_bytes(
    const iree_net_frame_sender_t* sender);

// Returns true if any sends are currently in flight.
bool iree_net_frame_sender_has_pending(const iree_net_frame_sender_t* sender);

// Returns the number of sends currently in flight.
int32_t iree_net_frame_sender_pending_count(
    const iree_net_frame_sender_t* sender);

// Handles completion of a send operation.
// Called by the carrier's completion callback dispatcher.
//
// |context| is the send context passed as user_data to carrier->send().
// |status| is OK on success, or an error status on failure.
//
// This function:
//   1. Releases the buffer lease back to the pool
//   2. Decrements sends_in_flight
//   3. Fires the user callback
//   4. Frees the context
void iree_net_frame_sender_handle_completion(
    iree_net_frame_send_context_t* context, iree_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_UTIL_FRAME_SENDER_H_
