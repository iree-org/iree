// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Control channel: typed message dispatch over a message endpoint.
//
// The control channel is a thin protocol layer that parses 8-byte frame headers
// (from frame.h) and dispatches by type: PING/PONG for liveness, GOAWAY for
// graceful shutdown, ERROR for error reporting, and DATA for opaque application
// payloads. It handles PING/PONG automatically and manages a simple lifecycle
// state machine (OPERATIONAL → DRAINING → ERROR).
//
// ## Composition model
//
// The control channel is a building block, not a session manager. It handles
// one endpoint and one message stream. Sessions that need multiple logical
// streams open multiple endpoints via connection.open_endpoint(), each with
// its own control channel. The control channel has no knowledge of HAL
// concepts (devices, buffers, RDMA) — those belong in the composition layer.
//
// ## Zero-copy
//
// On the receive path, the channel passes payload spans and buffer leases
// directly through to application callbacks without copying. The application
// can retain the lease to keep payload data valid beyond the callback.
//
// On the send path, the channel uses a frame_sender to manage buffer
// lifetimes. Small framing headers are copied into pool buffers that
// survive until send completion. Caller-provided payload data must remain
// valid until the on_send_complete callback fires.
//
// ## Threading
//
// All operations (send, receive, activate) happen on the proactor thread.
// No internal synchronization is needed beyond the atomic reference count
// and the atomic state field (which supports cross-thread state queries).

#ifndef IREE_NET_CHANNEL_CONTROL_CONTROL_CHANNEL_H_
#define IREE_NET_CHANNEL_CONTROL_CONTROL_CHANNEL_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Channel state
//===----------------------------------------------------------------------===//

// Control channel lifecycle states.
//
// The state machine is linear: CREATED → OPERATIONAL → DRAINING → ERROR,
// with a shortcut from OPERATIONAL directly to ERROR on transport failure
// or explicit error. No backward transitions.
typedef enum iree_net_control_channel_state_e {
  // Channel is created but not yet activated. No sends or receives.
  IREE_NET_CONTROL_CHANNEL_STATE_CREATED = 0,
  // Normal operation. All sends and receives are active.
  IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL = 1,
  // GOAWAY sent or received. Only ERROR sends are allowed (plus auto PONG).
  // In-flight DATA from the peer is still delivered.
  IREE_NET_CONTROL_CHANNEL_STATE_DRAINING = 2,
  // Terminal error. All operations fail. Only release is valid.
  IREE_NET_CONTROL_CHANNEL_STATE_ERROR = 3,
} iree_net_control_channel_state_t;

//===----------------------------------------------------------------------===//
// Callbacks and options
//===----------------------------------------------------------------------===//

// Called when a DATA frame is received.
//
// |flags| are the DATA frame's per-type flags (application-defined; the control
// channel passes them through without interpretation).
// |payload| is the DATA payload with the 8-byte header already stripped.
// |lease| references the backing buffer. The lease is valid for the duration of
// the callback. To keep payload data valid beyond the callback, retain the
// lease via iree_async_buffer_lease_retain().
//
// Return iree_ok_status() to continue receiving. Returning an error propagates
// to the endpoint and may cause deactivation.
typedef iree_status_t (*iree_net_control_channel_on_data_fn_t)(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

// Called when a GOAWAY frame is received from the remote peer.
//
// |reason_code| is 0 for normal shutdown, nonzero for an error category.
// |message| is the optional UTF-8 reason string (may be empty, not
// null-terminated in the span but safe to use as iree_string_view_t).
//
// The channel transitions to DRAINING before invoking this callback.
typedef void (*iree_net_control_channel_on_goaway_fn_t)(
    void* user_data, uint32_t reason_code, iree_string_view_t message);

// Called when an ERROR frame is received from the remote peer.
//
// |error_code| is the iree_status_code_t value from the error payload.
// |message| is the optional UTF-8 error description (may be empty).
//
// The channel transitions to ERROR state before invoking this callback.
typedef void (*iree_net_control_channel_on_error_fn_t)(
    void* user_data, uint32_t error_code, iree_string_view_t message);

// Called when a PONG response is received (in response to a PING we sent).
//
// |payload| echoes the payload from our original PING (excluding any appended
// responder timestamp). Applications that included a sender timestamp in the
// PING can extract it from |payload| to compute RTT.
//
// |responder_timestamp_ns| is the responder's monotonic timestamp in
// nanoseconds if the PONG had HAS_RESPONDER_TIMESTAMP set, or 0 if absent.
// Combined with the sender timestamp from |payload|, this enables one-way
// delay estimation and clock offset calculation.
typedef void (*iree_net_control_channel_on_pong_fn_t)(
    void* user_data, iree_const_byte_span_t payload,
    iree_time_t responder_timestamp_ns);

// Called when the underlying transport reports an error.
//
// After this callback, the channel is in ERROR state. |status| ownership is
// transferred to the callback (must be consumed or ignored).
typedef void (*iree_net_control_channel_on_transport_error_fn_t)(
    void* user_data, iree_status_t status);

// Called when a send_data operation completes (payload buffers are released).
//
// |operation_user_data| echoes the value from the send_data call, allowing
// callers to identify which send completed and release associated resources.
// |status| indicates success or failure of the send.
//
// This callback only fires for send_data operations (which use zero-copy
// payload delivery). Small control sends (ping, goaway, error) copy data
// synchronously and do not fire this callback.
typedef void (*iree_net_control_channel_on_send_complete_fn_t)(
    void* user_data, uint64_t operation_user_data, iree_status_t status);

// Bundled application callbacks for channel events.
//
// |on_data| is required — receiving a DATA frame with no handler is a protocol
// error. All other callbacks are optional; NULL callbacks are safe (the channel
// handles the frame internally and skips the notification).
//
// All callbacks fire on the proactor thread. The shared |user_data| is passed
// as the first argument to each callback.
typedef struct iree_net_control_channel_callbacks_t {
  iree_net_control_channel_on_data_fn_t on_data;
  iree_net_control_channel_on_goaway_fn_t on_goaway;
  iree_net_control_channel_on_error_fn_t on_error;
  iree_net_control_channel_on_pong_fn_t on_pong;
  iree_net_control_channel_on_transport_error_fn_t on_transport_error;
  iree_net_control_channel_on_send_complete_fn_t on_send_complete;
  void* user_data;
} iree_net_control_channel_callbacks_t;

// Configuration options for channel behavior.
typedef struct iree_net_control_channel_options_t {
  // Whether to append our monotonic timestamp to PONG responses.
  // When true, PONG frames include HAS_RESPONDER_TIMESTAMP and 8 bytes of
  // LE uint64 timestamp after the echoed PING payload. This enables the PING
  // sender to estimate clock offset and one-way delay.
  bool append_responder_timestamp;
} iree_net_control_channel_options_t;

// Returns default options: responder timestamp enabled.
static inline iree_net_control_channel_options_t
iree_net_control_channel_options_default(void) {
  iree_net_control_channel_options_t options;
  memset(&options, 0, sizeof(options));
  options.append_responder_timestamp = true;
  return options;
}

//===----------------------------------------------------------------------===//
// iree_net_control_channel_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_control_channel_t iree_net_control_channel_t;

// Creates a control channel that will operate over the given message endpoint.
//
// The |endpoint| is a borrowed view used for both receive and send paths. On
// the receive path, the channel installs message callbacks. On the send path,
// the channel routes data through the endpoint's send vtable, which ensures
// any transport-specific framing (e.g., TCP mux stream headers) is applied
// before the data reaches the carrier.
//
// The caller must ensure the underlying transport object (framing_adapter,
// stream_mux slot, etc.) outlives the channel. The channel does NOT activate
// the endpoint at creation; call iree_net_control_channel_activate() when
// ready to begin receiving.
//
// The |header_pool| provides buffers for copying frame headers and batching
// small control messages. Pool buffers must be at least 256 bytes. The pool
// is borrowed — caller must keep it alive for the channel's lifetime.
//
// |max_send_spans| is the maximum number of scatter-gather spans per send
// operation, accounting for overhead added by the endpoint's send path. For
// endpoints that are passthroughs to the carrier, use carrier->max_iov. For
// endpoints that add transport headers (e.g., TCP mux adds one span), subtract
// the header span count from carrier->max_iov.
//
// |callbacks.on_data| must be non-NULL. The callbacks struct is copied; the
// |callbacks.user_data| pointer must remain valid for the channel's lifetime.
//
// The carrier backing this endpoint must have its send completion callback
// set to iree_net_frame_sender_dispatch_carrier_completion (or equivalent)
// so that frame_sender completions are properly routed.
//
// The channel starts in CREATED state with ref_count = 1.
iree_status_t iree_net_control_channel_create(
    iree_net_message_endpoint_t endpoint, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_control_channel_options_t options,
    iree_net_control_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_control_channel_t** out_channel);

// Retains a reference to the channel. NULL-safe (no-op on NULL).
void iree_net_control_channel_retain(iree_net_control_channel_t* channel);

// Releases a reference. Destroys the channel when the last reference is
// released. NULL-safe (no-op on NULL).
void iree_net_control_channel_release(iree_net_control_channel_t* channel);

// Activates the channel, enabling message receipt.
//
// Installs the channel's handlers on the endpoint via set_callbacks() and
// calls endpoint activate(). After this, incoming messages are parsed and
// dispatched by frame type.
//
// Must be called from the proactor thread. Transitions CREATED → OPERATIONAL.
// Returns FAILED_PRECONDITION if the channel is not in CREATED state.
iree_status_t iree_net_control_channel_activate(
    iree_net_control_channel_t* channel);

// Returns the current channel state.
//
// Uses an atomic acquire load, so this is safe to call from any thread for
// status display. However, by the time the caller acts on the result, the
// state may have changed on the proactor thread.
iree_net_control_channel_state_t iree_net_control_channel_state(
    const iree_net_control_channel_t* channel);

// Sends a DATA frame with application-defined flags and payload.
//
// |flags| are per-frame flag bits passed through to the remote on_data
// callback without interpretation.
//
// |payload| is a scatter-gather list of application data. The payload buffers
// are sent zero-copy — they must remain valid until the on_send_complete
// callback fires with the matching |operation_user_data|.
//
// |operation_user_data| is echoed to the on_send_complete callback for
// correlation. Callers typically use this to identify the buffer to free.
//
// Requires OPERATIONAL state. Returns FAILED_PRECONDITION in CREATED,
// DRAINING, or ERROR. On non-OK return, on_send_complete is NOT called.
iree_status_t iree_net_control_channel_send_data(
    iree_net_control_channel_t* channel, iree_net_control_frame_flags_t flags,
    iree_async_span_list_t payload, uint64_t operation_user_data);

// Sends a PING frame for liveness detection and RTT measurement.
//
// |payload| is echoed back in the PONG response. The recommended payload
// format is an 8-byte LE uint64 monotonic timestamp in nanoseconds; the
// on_pong callback can then compute RTT = now - sender_timestamp.
//
// Requires OPERATIONAL state. Returns FAILED_PRECONDITION otherwise.
iree_status_t iree_net_control_channel_send_ping(
    iree_net_control_channel_t* channel, iree_const_byte_span_t payload);

// Sends a GOAWAY frame, initiating graceful shutdown.
//
// |reason_code| is 0 for normal shutdown, nonzero for an error category.
// |message| is an optional UTF-8 reason string.
//
// Transitions OPERATIONAL → DRAINING on success. After sending GOAWAY, only
// ERROR frames may be sent (plus auto PONG responses).
// Returns FAILED_PRECONDITION if not in OPERATIONAL state.
iree_status_t iree_net_control_channel_send_goaway(
    iree_net_control_channel_t* channel, uint32_t reason_code,
    iree_string_view_t message);

// Sends an ERROR frame reporting an error condition to the remote peer.
//
// |error_code| is an iree_status_code_t value. |message| is an optional
// UTF-8 error description.
//
// Allowed in OPERATIONAL and DRAINING states. Transitions to ERROR state on
// success. Returns FAILED_PRECONDITION if already in ERROR or CREATED state.
iree_status_t iree_net_control_channel_send_error(
    iree_net_control_channel_t* channel, iree_status_code_t error_code,
    iree_string_view_t message);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_CONTROL_CONTROL_CHANNEL_H_
