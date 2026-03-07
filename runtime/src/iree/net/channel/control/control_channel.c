// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/control/control_channel.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/net/channel/util/frame_sender.h"

//===----------------------------------------------------------------------===//
// iree_net_control_channel_t
//===----------------------------------------------------------------------===//

struct iree_net_control_channel_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Borrowed view into the transport. Must outlive the channel.
  // Used for the receive path (message callbacks, activation).
  iree_net_message_endpoint_t endpoint;

  // Embedded frame sender for the send path. Manages header pool buffer
  // allocations, in-flight tracking, and completion dispatch.
  iree_net_frame_sender_t sender;

  iree_net_control_channel_options_t options;
  iree_net_control_channel_callbacks_t callbacks;

  // Lifecycle state. Written on proactor thread, read with atomic acquire.
  iree_atomic_int32_t state;
};

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static iree_net_control_channel_state_t iree_net_control_channel_load_state(
    const iree_net_control_channel_t* channel) {
  return (iree_net_control_channel_state_t)iree_atomic_load(
      &((iree_net_control_channel_t*)channel)->state,
      iree_memory_order_acquire);
}

static void iree_net_control_channel_set_state(
    iree_net_control_channel_t* channel,
    iree_net_control_channel_state_t new_state) {
  iree_atomic_store(&channel->state, (int32_t)new_state,
                    iree_memory_order_release);
}

// Submit callback for frame_sender: routes sends through the message endpoint.
//
// This ensures that any transport-specific framing (e.g., TCP mux stream
// headers) is applied before data reaches the carrier. The endpoint's send
// vtable handles the framing transparently — for passthrough endpoints
// (loopback, shm), this is equivalent to carrier_send. For muxed endpoints
// (TCP), the endpoint prepends the stream header.
static iree_status_t iree_net_control_channel_submit_send(
    void* user_data, iree_async_span_list_t data, uint64_t send_user_data) {
  iree_net_control_channel_t* channel = (iree_net_control_channel_t*)user_data;
  iree_net_message_endpoint_send_params_t params = {
      .data = data,
      .user_data = send_user_data,
  };
  return iree_net_message_endpoint_send(channel->endpoint, &params);
}

// Frame sender completion callback. Routes to the channel's on_send_complete
// callback for send_data operations.
static void iree_net_control_channel_on_sender_complete(
    void* callback_user_data, uint64_t operation_user_data,
    iree_status_t status) {
  iree_net_control_channel_t* channel =
      (iree_net_control_channel_t*)callback_user_data;
  if (channel->callbacks.on_send_complete) {
    channel->callbacks.on_send_complete(channel->callbacks.user_data,
                                        operation_user_data, status);
  } else {
    iree_status_ignore(status);
  }
}

// Serializes a complete control frame (header + sub-header + optional trailing
// data) into a contiguous buffer for queue/flush sends. Returns the total
// number of bytes written to |buffer|.
//
// |sub_header| and |trailing| may be NULL/zero-length.
static iree_host_size_t iree_net_control_channel_serialize_frame(
    uint8_t* buffer, iree_net_control_frame_type_t type,
    iree_net_control_frame_flags_t flags, const void* sub_header,
    iree_host_size_t sub_header_size, const void* trailing,
    iree_host_size_t trailing_size) {
  iree_host_size_t offset = 0;

  // Frame header.
  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(type, flags, &header);
  memcpy(buffer + offset, &header, sizeof(header));
  offset += sizeof(header);

  // Sub-header (e.g., goaway_payload, error_payload).
  if (sub_header_size > 0) {
    memcpy(buffer + offset, sub_header, sub_header_size);
    offset += sub_header_size;
  }

  // Trailing data (e.g., message string, echoed payload).
  if (trailing_size > 0) {
    memcpy(buffer + offset, trailing, trailing_size);
    offset += trailing_size;
  }

  return offset;
}

//===----------------------------------------------------------------------===//
// Receive path
//===----------------------------------------------------------------------===//

// Maximum PING echo payload size for PONG responses. PONG frames are
// serialized contiguously into a batch buffer, so this limits the maximum
// PING payload that can be echoed. 64 bytes is generous — typical PING
// payloads are 8 bytes (a timestamp).
#define IREE_NET_CONTROL_MAX_PING_ECHO_SIZE 64

// Handles a received PING frame by auto-responding with PONG.
//
// The entire PONG response (header + echoed payload + optional timestamp) is
// serialized into a contiguous stack buffer and queued via the frame_sender's
// batch path. This avoids the recv buffer lifetime issue: the echoed payload
// is copied out of the recv buffer before the recv callback returns.
static iree_status_t iree_net_control_channel_handle_ping(
    iree_net_control_channel_t* channel, iree_const_byte_span_t payload) {
  iree_net_control_frame_flags_t pong_flags = IREE_NET_CONTROL_PONG_FLAG_NONE;

  // Clamp echo size to prevent oversized PONGs.
  iree_host_size_t echo_size = payload.data_length;
  if (echo_size > IREE_NET_CONTROL_MAX_PING_ECHO_SIZE) {
    echo_size = IREE_NET_CONTROL_MAX_PING_ECHO_SIZE;
  }

  // Build PONG frame contiguously on the stack.
  uint8_t pong_buffer[IREE_NET_CONTROL_FRAME_HEADER_SIZE +
                      IREE_NET_CONTROL_MAX_PING_ECHO_SIZE + sizeof(uint64_t)];
  iree_host_size_t offset = 0;

  // Header.
  if (channel->options.append_responder_timestamp) {
    pong_flags |= IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP;
  }
  iree_net_control_frame_header_t pong_header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_PONG,
                                           pong_flags, &pong_header);
  memcpy(pong_buffer + offset, &pong_header, sizeof(pong_header));
  offset += sizeof(pong_header);

  // Echoed PING payload (copied from recv buffer).
  if (echo_size > 0) {
    memcpy(pong_buffer + offset, payload.data, echo_size);
    offset += echo_size;
  }

  // Optional responder timestamp.
  if (channel->options.append_responder_timestamp) {
    uint64_t timestamp_le = (uint64_t)iree_time_now();
    memcpy(pong_buffer + offset, &timestamp_le, sizeof(timestamp_le));
    offset += sizeof(timestamp_le);
  }

  // Queue and flush. The batch buffer copy protects against the recv buffer
  // dying when this callback returns.
  iree_status_t status = iree_net_frame_sender_queue(
      &channel->sender, iree_make_const_byte_span(pong_buffer, offset));
  if (iree_status_is_ok(status)) {
    status = iree_net_frame_sender_flush(&channel->sender,
                                         /*operation_user_data=*/0);
  }

  // Best-effort: if the PONG send fails (backpressure, transport error), the
  // peer will detect liveness timeout and retry or close. Propagating the
  // error to the recv path would incorrectly kill the channel.
  iree_status_ignore(status);
  return iree_ok_status();
}

// Handles a received PONG frame by extracting the optional responder timestamp
// and delivering to the application callback.
static iree_status_t iree_net_control_channel_handle_pong(
    iree_net_control_channel_t* channel, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload) {
  if (!channel->callbacks.on_pong) return iree_ok_status();

  iree_time_t responder_timestamp_ns = 0;
  iree_const_byte_span_t echoed_payload = payload;

  if (flags & IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP) {
    // Last 8 bytes are the responder timestamp.
    if (payload.data_length < sizeof(uint64_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "PONG has HAS_RESPONDER_TIMESTAMP but payload is only %" PRIhsz
          " bytes (need at least 8)",
          payload.data_length);
    }
    uint64_t timestamp_le = 0;
    memcpy(&timestamp_le, payload.data + payload.data_length - sizeof(uint64_t),
           sizeof(uint64_t));
    responder_timestamp_ns = (iree_time_t)timestamp_le;
    echoed_payload = iree_make_const_byte_span(
        payload.data, payload.data_length - sizeof(uint64_t));
  }

  channel->callbacks.on_pong(channel->callbacks.user_data, echoed_payload,
                             responder_timestamp_ns);
  return iree_ok_status();
}

// Handles a received GOAWAY frame.
static iree_status_t iree_net_control_channel_handle_goaway(
    iree_net_control_channel_t* channel, iree_const_byte_span_t payload) {
  if (payload.data_length < sizeof(iree_net_control_goaway_payload_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "GOAWAY payload too short: %" PRIhsz " bytes (need at least %zu)",
        payload.data_length, sizeof(iree_net_control_goaway_payload_t));
  }

  iree_net_control_goaway_payload_t goaway;
  memcpy(&goaway, payload.data, sizeof(goaway));

  iree_string_view_t message =
      iree_make_string_view((const char*)payload.data + sizeof(goaway),
                            payload.data_length - sizeof(goaway));

  // Transition to DRAINING (idempotent if already draining).
  iree_net_control_channel_state_t current =
      iree_net_control_channel_load_state(channel);
  if (current == IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL) {
    iree_net_control_channel_set_state(channel,
                                       IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
  }

  if (channel->callbacks.on_goaway) {
    channel->callbacks.on_goaway(channel->callbacks.user_data,
                                 goaway.reason_code, message);
  }
  return iree_ok_status();
}

// Handles a received ERROR frame.
static iree_status_t iree_net_control_channel_handle_error_frame(
    iree_net_control_channel_t* channel, iree_const_byte_span_t payload) {
  if (payload.data_length < sizeof(iree_net_control_error_payload_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ERROR payload too short: %" PRIhsz " bytes (need at least %zu)",
        payload.data_length, sizeof(iree_net_control_error_payload_t));
  }

  iree_net_control_error_payload_t error;
  memcpy(&error, payload.data, sizeof(error));

  iree_string_view_t message =
      iree_make_string_view((const char*)payload.data + sizeof(error),
                            payload.data_length - sizeof(error));

  iree_net_control_channel_set_state(channel,
                                     IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  if (channel->callbacks.on_error) {
    channel->callbacks.on_error(channel->callbacks.user_data, error.error_code,
                                message);
  }
  return iree_ok_status();
}

// Endpoint message callback: parses header and dispatches by frame type.
static iree_status_t iree_net_control_channel_on_message(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease) {
  iree_net_control_channel_t* channel = (iree_net_control_channel_t*)user_data;

  // In ERROR state, reject all messages.
  iree_net_control_channel_state_t state =
      iree_net_control_channel_load_state(channel);
  if (state == IREE_NET_CONTROL_CHANNEL_STATE_ERROR) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "control channel is in error state");
  }

  // Validate minimum size. Using iree_status_from_code to avoid allocation
  // on the receive hot path — these indicate transport-layer bugs, not normal
  // operation, and the frame data itself is more useful for debugging than a
  // canned diagnostic string.
  if (message.data_length < IREE_NET_CONTROL_FRAME_HEADER_SIZE) {
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }

  // Parse header (memcpy to aligned local for safety).
  iree_net_control_frame_header_t header;
  memcpy(&header, message.data, sizeof(header));
  IREE_RETURN_IF_ERROR(iree_net_control_frame_header_validate(header));

  // Extract payload (everything after the 8-byte header).
  iree_const_byte_span_t payload = iree_make_const_byte_span(
      message.data + IREE_NET_CONTROL_FRAME_HEADER_SIZE,
      message.data_length - IREE_NET_CONTROL_FRAME_HEADER_SIZE);

  iree_net_control_frame_type_t type =
      iree_net_control_frame_header_type(header);
  iree_net_control_frame_flags_t flags =
      iree_net_control_frame_header_flags(header);

  switch (type) {
    case IREE_NET_CONTROL_FRAME_TYPE_PING:
      return iree_net_control_channel_handle_ping(channel, payload);

    case IREE_NET_CONTROL_FRAME_TYPE_PONG:
      return iree_net_control_channel_handle_pong(channel, flags, payload);

    case IREE_NET_CONTROL_FRAME_TYPE_GOAWAY:
      return iree_net_control_channel_handle_goaway(channel, payload);

    case IREE_NET_CONTROL_FRAME_TYPE_ERROR:
      return iree_net_control_channel_handle_error_frame(channel, payload);

    case IREE_NET_CONTROL_FRAME_TYPE_DATA:
      return channel->callbacks.on_data(channel->callbacks.user_data, flags,
                                        payload, lease);

    default:
      return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }
}

// Endpoint error callback: transitions to ERROR state.
static void iree_net_control_channel_on_endpoint_error(void* user_data,
                                                       iree_status_t status) {
  iree_net_control_channel_t* channel = (iree_net_control_channel_t*)user_data;

  iree_net_control_channel_set_state(channel,
                                     IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  if (channel->callbacks.on_transport_error) {
    channel->callbacks.on_transport_error(channel->callbacks.user_data, status);
  } else {
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_net_control_channel_destroy(
    iree_net_control_channel_t* channel);

iree_status_t iree_net_control_channel_create(
    iree_net_message_endpoint_t endpoint, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* header_pool,
    iree_net_control_channel_options_t options,
    iree_net_control_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_control_channel_t** out_channel) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(header_pool);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;

  // on_data is required.
  if (!callbacks.on_data) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "on_data callback is required");
  }

  iree_net_control_channel_t* channel = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*channel),
                                (void**)&channel));
  memset(channel, 0, sizeof(*channel));

  iree_atomic_ref_count_init(&channel->ref_count);
  channel->host_allocator = host_allocator;
  channel->endpoint = endpoint;
  channel->options = options;
  channel->callbacks = callbacks;
  iree_atomic_store(&channel->state,
                    (int32_t)IREE_NET_CONTROL_CHANNEL_STATE_CREATED,
                    iree_memory_order_release);

  // Initialize the embedded frame sender for the send path. Sends route
  // through the message endpoint to pick up any transport framing.
  iree_net_frame_send_complete_callback_t send_complete = {
      .fn = iree_net_control_channel_on_sender_complete,
      .user_data = channel,
  };
  iree_status_t status = iree_net_frame_sender_initialize(
      &channel->sender, iree_net_control_channel_submit_send, channel,
      max_send_spans, header_pool, send_complete, host_allocator,
      host_allocator);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, channel);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_channel = channel;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_net_control_channel_retain(iree_net_control_channel_t* channel) {
  if (channel) iree_atomic_ref_count_inc(&channel->ref_count);
}

void iree_net_control_channel_release(iree_net_control_channel_t* channel) {
  if (channel && iree_atomic_ref_count_dec(&channel->ref_count) == 1) {
    iree_net_control_channel_destroy(channel);
  }
}

static void iree_net_control_channel_destroy(
    iree_net_control_channel_t* channel) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clear endpoint callbacks to stop message delivery.
  iree_net_message_endpoint_callbacks_t empty_callbacks;
  memset(&empty_callbacks, 0, sizeof(empty_callbacks));
  iree_net_message_endpoint_set_callbacks(channel->endpoint, empty_callbacks);

  // Deinitialize the frame sender. Asserts no sends in flight.
  iree_net_frame_sender_deinitialize(&channel->sender);

  iree_allocator_t host_allocator = channel->host_allocator;
  iree_allocator_free(host_allocator, channel);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_net_control_channel_activate(
    iree_net_control_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_control_channel_state_t state =
      iree_net_control_channel_load_state(channel);
  if (state != IREE_NET_CONTROL_CHANNEL_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "channel not in CREATED state (state=%d)",
                            (int)state);
  }

  // Install our handlers on the endpoint.
  iree_net_message_endpoint_callbacks_t endpoint_callbacks = {
      .on_message = iree_net_control_channel_on_message,
      .on_error = iree_net_control_channel_on_endpoint_error,
      .user_data = channel,
  };
  iree_net_message_endpoint_set_callbacks(channel->endpoint,
                                          endpoint_callbacks);

  iree_status_t status = iree_net_message_endpoint_activate(channel->endpoint);
  if (iree_status_is_ok(status)) {
    iree_net_control_channel_set_state(
        channel, IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Query
//===----------------------------------------------------------------------===//

iree_net_control_channel_state_t iree_net_control_channel_state(
    const iree_net_control_channel_t* channel) {
  IREE_ASSERT_ARGUMENT(channel);
  return iree_net_control_channel_load_state(channel);
}

//===----------------------------------------------------------------------===//
// Send path
//===----------------------------------------------------------------------===//

iree_status_t iree_net_control_channel_send_data(
    iree_net_control_channel_t* channel, iree_net_control_frame_flags_t flags,
    iree_async_span_list_t payload, uint64_t operation_user_data) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_control_channel_state_t state =
      iree_net_control_channel_load_state(channel);
  if (state != IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send DATA: channel state is %d",
                            (int)state);
  }

  // Build the 8-byte frame header on the stack. frame_sender.send() copies
  // it into a pool buffer, so the stack-local data is safe.
  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_DATA,
                                           flags, &header);

  iree_status_t status = iree_net_frame_sender_send(
      &channel->sender,
      iree_make_const_byte_span((const uint8_t*)&header, sizeof(header)),
      payload, operation_user_data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_net_control_channel_send_ping(
    iree_net_control_channel_t* channel, iree_const_byte_span_t payload) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_control_channel_state_t state =
      iree_net_control_channel_load_state(channel);
  if (state != IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send PING: channel state is %d",
                            (int)state);
  }

  // Serialize complete PING frame contiguously. queue() copies data into a
  // batch buffer, so stack-local and caller-provided data are safe to free
  // immediately after this function returns.
  uint8_t frame_buffer[IREE_NET_CONTROL_FRAME_HEADER_SIZE + 256];
  iree_host_size_t frame_size = iree_net_control_channel_serialize_frame(
      frame_buffer, IREE_NET_CONTROL_FRAME_TYPE_PING, 0, NULL, 0, payload.data,
      payload.data_length);

  iree_status_t status = iree_net_frame_sender_queue(
      &channel->sender, iree_make_const_byte_span(frame_buffer, frame_size));
  if (iree_status_is_ok(status)) {
    status = iree_net_frame_sender_flush(&channel->sender,
                                         /*operation_user_data=*/0);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_net_control_channel_send_goaway(
    iree_net_control_channel_t* channel, uint32_t reason_code,
    iree_string_view_t message) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_control_channel_state_t state =
      iree_net_control_channel_load_state(channel);
  if (state != IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send GOAWAY: channel state is %d",
                            (int)state);
  }

  // Serialize complete GOAWAY frame contiguously.
  iree_net_control_goaway_payload_t goaway_payload;
  memset(&goaway_payload, 0, sizeof(goaway_payload));
  goaway_payload.reason_code = reason_code;

  uint8_t frame_buffer[IREE_NET_CONTROL_FRAME_HEADER_SIZE +
                       sizeof(iree_net_control_goaway_payload_t) + 256];
  iree_host_size_t frame_size = iree_net_control_channel_serialize_frame(
      frame_buffer, IREE_NET_CONTROL_FRAME_TYPE_GOAWAY, 0, &goaway_payload,
      sizeof(goaway_payload), message.data, message.size);

  iree_status_t status = iree_net_frame_sender_queue(
      &channel->sender, iree_make_const_byte_span(frame_buffer, frame_size));
  if (iree_status_is_ok(status)) {
    status = iree_net_frame_sender_flush(&channel->sender,
                                         /*operation_user_data=*/0);
  }
  if (iree_status_is_ok(status)) {
    iree_net_control_channel_set_state(channel,
                                       IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_net_control_channel_send_error(
    iree_net_control_channel_t* channel, iree_status_code_t error_code,
    iree_string_view_t message) {
  IREE_ASSERT_ARGUMENT(channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_control_channel_state_t state =
      iree_net_control_channel_load_state(channel);
  if (state == IREE_NET_CONTROL_CHANNEL_STATE_ERROR ||
      state == IREE_NET_CONTROL_CHANNEL_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot send ERROR: channel state is %d",
                            (int)state);
  }

  // Serialize complete ERROR frame contiguously.
  iree_net_control_error_payload_t error_payload;
  memset(&error_payload, 0, sizeof(error_payload));
  error_payload.error_code = (uint32_t)error_code;

  uint8_t frame_buffer[IREE_NET_CONTROL_FRAME_HEADER_SIZE +
                       sizeof(iree_net_control_error_payload_t) + 256];
  iree_host_size_t frame_size = iree_net_control_channel_serialize_frame(
      frame_buffer, IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0, &error_payload,
      sizeof(error_payload), message.data, message.size);

  iree_status_t status = iree_net_frame_sender_queue(
      &channel->sender, iree_make_const_byte_span(frame_buffer, frame_size));
  if (iree_status_is_ok(status)) {
    status = iree_net_frame_sender_flush(&channel->sender,
                                         /*operation_user_data=*/0);
  }
  if (iree_status_is_ok(status)) {
    iree_net_control_channel_set_state(channel,
                                       IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
