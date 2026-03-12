// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Queue channel: frontier-ordered command dispatch over a message endpoint.
//
// The queue channel parses 16-byte queue frame headers (from frame.h),
// extracts optional wait/signal frontiers from the payload, and dispatches
// command data to the application callback. On the send path, it encodes
// frontiers into outgoing frames and manages zero-copy payload delivery.
//
// The channel is agnostic to what the command payload contains — HAL
// operations, pipeline commands, collective ops, etc. It handles framing,
// frontier extraction, and delivery; consumers interpret the payload.
// Command payloads are typically 64KB-512KB (serialized HAL command buffers)
// and are always sent zero-copy through the span list.
//
// ## Frontier handling
//
// On receive, the channel strips frontier metadata from the payload based
// on the frame's HAS_WAIT_FRONTIER and HAS_SIGNAL_FRONTIER flags. Frontier
// pointers delivered to the callback point directly into the recv buffer
// (zero-copy). The callback can retain the lease to extend data lifetime.
//
// On send, the channel accepts frontier pointers and encodes them into the
// frame header region. The queue frame header + frontier data are copied
// into a pool buffer; the command payload is sent zero-copy.
//
// ## Stream multiplexing
//
// Each queue frame carries a stream_id for multiplexing command streams over
// a single endpoint. COMMAND frames are self-contained (one frame = one
// command). DATA/DATA_END frames fragment large payloads by stream_id,
// reassembled by the channel before delivery.
//
// ## Threading
//
// All operations happen on the proactor thread. No internal synchronization
// beyond the atomic reference count and state field.

#ifndef IREE_NET_CHANNEL_QUEUE_QUEUE_CHANNEL_H_
#define IREE_NET_CHANNEL_QUEUE_QUEUE_CHANNEL_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/channel/queue/frame.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Channel state
//===----------------------------------------------------------------------===//

// Queue channel lifecycle states.
//
// Linear state machine: CREATED -> OPERATIONAL -> ERROR.
// No DRAINING state — graceful shutdown is signaled out-of-band (e.g., via
// a control channel GOAWAY). The queue channel processes commands until
// the transport errors or the channel is released.
typedef enum iree_net_queue_channel_state_e {
  // Channel is created but not yet activated. No sends or receives.
  IREE_NET_QUEUE_CHANNEL_STATE_CREATED = 0,
  // Normal operation. All sends and receives are active.
  IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL = 1,
  // Terminal error. All operations fail. Only release is valid.
  IREE_NET_QUEUE_CHANNEL_STATE_ERROR = 2,
} iree_net_queue_channel_state_t;

//===----------------------------------------------------------------------===//
// Callbacks
//===----------------------------------------------------------------------===//

// Called when a COMMAND frame is received with frontiers extracted.
//
// |stream_id| identifies the command stream (for multiplexing).
// |wait_frontier| is the wait frontier if HAS_WAIT_FRONTIER was set, or NULL.
// |signal_frontier| is the signal frontier if HAS_SIGNAL_FRONTIER was set, or
// NULL.
// |command_data| is the command payload with frontiers already stripped.
// |lease| references the backing buffer. Retain it to keep frontier and
// command data valid beyond the callback.
//
// Frontier pointers point directly into the recv buffer — they share the
// lease's lifetime. The channel validates frontier headers before delivery
// (entry_count bounds check, size consistency) but does NOT validate sorted
// order — the caller should use iree_async_frontier_validate() in debug builds
// if needed.
//
// Return iree_ok_status() to continue receiving. Returning an error propagates
// to the endpoint and may cause deactivation.
typedef iree_status_t (*iree_net_queue_channel_on_command_fn_t)(
    void* user_data, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t command_data, iree_async_buffer_lease_t* lease);

// Called when the underlying transport reports an error.
//
// After this callback, the channel is in ERROR state. |status| ownership is
// transferred to the callback (must be consumed or ignored).
typedef void (*iree_net_queue_channel_on_transport_error_fn_t)(
    void* user_data, iree_status_t status);

// Called when an ADVANCE frame is received with signal frontier and optional
// advance payload (resolution entries, etc.).
//
// |signal_frontier| is the frontier the server has completed (always present
// on well-formed ADVANCE frames — NULL if the frame somehow had no frontier).
// |advance_data| is the remaining payload after the signal frontier (may be
// empty). For barrier operations this is empty. For alloca operations it
// contains resolution entries mapping provisional→resolved buffer IDs.
// |lease| references the backing buffer. Retain to extend data lifetime.
//
// Return iree_ok_status() to continue receiving.
typedef iree_status_t (*iree_net_queue_channel_on_advance_fn_t)(
    void* user_data, const iree_async_frontier_t* signal_frontier,
    iree_const_byte_span_t advance_data, iree_async_buffer_lease_t* lease);

// Called when a send_command or send_advance operation completes (payload
// buffers released).
//
// |operation_user_data| echoes the value from the send call.
// |status| indicates success or failure.
typedef void (*iree_net_queue_channel_on_send_complete_fn_t)(
    void* user_data, uint64_t operation_user_data, iree_status_t status);

// Bundled application callbacks for channel events.
//
// |on_command| is required. All other callbacks are optional (NULL is safe).
// All callbacks fire on the proactor thread. The shared |user_data| is passed
// as the first argument to each callback.
typedef struct iree_net_queue_channel_callbacks_t {
  iree_net_queue_channel_on_command_fn_t on_command;
  iree_net_queue_channel_on_advance_fn_t on_advance;
  iree_net_queue_channel_on_transport_error_fn_t on_transport_error;
  iree_net_queue_channel_on_send_complete_fn_t on_send_complete;
  void* user_data;
} iree_net_queue_channel_callbacks_t;

//===----------------------------------------------------------------------===//
// iree_net_queue_channel_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_queue_channel_t iree_net_queue_channel_t;

// Creates a queue channel that will operate over the given message endpoint.
//
// The |endpoint| is a borrowed view used for both receive and send paths.
// The caller must ensure the underlying transport object outlives the channel.
//
// The |header_pool| provides buffers for copying the queue frame header and
// frontier metadata into a contiguous block for scatter-gather sends. Pool
// buffers must be large enough for the queue frame header (16 bytes) plus
// the largest expected frontier pair. A conservative minimum is 1024 bytes
// (handles frontiers with up to ~30 entries each). The channel takes
// ownership of the pool and frees it on destroy. This ensures the pool
// remains valid as long as any reference to the channel exists (e.g.,
// barrier completion contexts that retain the channel for async sends).
// Note: the command payload (typically 64KB-512KB) is NOT copied into pool
// buffers — it is sent zero-copy through the span list.
//
// |max_send_spans| is the maximum scatter-gather spans per send, accounting
// for overhead from the endpoint's send path.
//
// |callbacks.on_command| must be non-NULL.
//
// The carrier backing this endpoint must have its send completion callback
// set to iree_net_frame_sender_dispatch_carrier_completion (or equivalent).
//
// The channel starts in CREATED state with ref_count = 1.
iree_status_t iree_net_queue_channel_create(
    iree_net_message_endpoint_t endpoint, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_queue_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_queue_channel_t** out_channel);

// Retains a reference. NULL-safe.
void iree_net_queue_channel_retain(iree_net_queue_channel_t* channel);

// Releases a reference. Destroys when last reference released. NULL-safe.
void iree_net_queue_channel_release(iree_net_queue_channel_t* channel);

// Activates the channel, enabling message receipt.
//
// Installs handlers on the endpoint and calls activate(). After this,
// incoming queue frames are parsed and dispatched.
//
// Must be called from the proactor thread. Transitions CREATED -> OPERATIONAL.
// Returns FAILED_PRECONDITION if not in CREATED state.
iree_status_t iree_net_queue_channel_activate(
    iree_net_queue_channel_t* channel);

// Detaches the channel from its underlying endpoint. Clears endpoint callbacks
// and zeroes the endpoint reference so that subsequent destroy does not
// attempt to operate on a freed endpoint.
//
// Must be called while the endpoint is still valid (before the session or
// connection that owns it is released). After detach, the channel cannot send
// or receive, but may still be retained (e.g., by barrier completions) and
// later released safely.
//
// Transitions to ERROR state. No-op if already detached or never activated.
void iree_net_queue_channel_detach(iree_net_queue_channel_t* channel);

// Returns the current channel state.
//
// Uses an atomic acquire load — safe to call from any thread for status
// queries. The state may change on the proactor thread after reading.
iree_net_queue_channel_state_t iree_net_queue_channel_state(
    const iree_net_queue_channel_t* channel);

// Returns true if any send operations are still in flight.
//
// Uses atomic operations — safe to call from any thread. Callers must drain
// pending sends (by polling the proactor) before releasing the channel;
// the frame_sender asserts sends_in_flight == 0 on deinitialize.
bool iree_net_queue_channel_has_pending_sends(
    const iree_net_queue_channel_t* channel);

// Sends a COMMAND frame with optional frontiers and command payload.
//
// |stream_id| identifies the command stream (for multiplexing).
// |wait_frontier| and |signal_frontier| may be NULL if no frontier is needed.
// |command_payload| is a scatter-gather list of command data (typically
// 64KB-512KB of serialized HAL commands) sent zero-copy.
// |operation_user_data| is echoed to on_send_complete for correlation.
//
// The queue frame header and frontier data (up to ~1KB) are copied into a
// pool buffer. The command payload buffers must remain valid until
// on_send_complete fires.
//
// Requires OPERATIONAL state. Returns FAILED_PRECONDITION otherwise.
// On non-OK return, on_send_complete is NOT called.
iree_status_t iree_net_queue_channel_send_command(
    iree_net_queue_channel_t* channel, uint32_t stream_id,
    const iree_async_frontier_t* wait_frontier,
    const iree_async_frontier_t* signal_frontier,
    iree_async_span_list_t command_payload, uint64_t operation_user_data);

// Sends an ADVANCE frame with a signal frontier and optional advance payload.
//
// ADVANCE frames flow server→client to indicate that operations up to the
// given signal frontier have completed. The frontier is encoded exactly as
// in COMMAND frames (HAS_SIGNAL_FRONTIER flag, serialized after the header).
//
// |signal_frontier| must be non-NULL with at least one entry.
// |advance_payload| is an optional scatter-gather list of advance data
// (resolution entries, etc.) sent zero-copy. May be empty.
// |operation_user_data| is echoed to on_send_complete.
//
// Requires OPERATIONAL state.
iree_status_t iree_net_queue_channel_send_advance(
    iree_net_queue_channel_t* channel,
    const iree_async_frontier_t* signal_frontier,
    iree_async_span_list_t advance_payload, uint64_t operation_user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_QUEUE_QUEUE_CHANNEL_H_
