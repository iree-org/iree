// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/control/control_channel.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

//===----------------------------------------------------------------------===//
// iree_net_control_channel_t
//===----------------------------------------------------------------------===//

struct iree_net_control_channel_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Borrowed view into the transport. Must outlive the channel.
  iree_net_message_endpoint_t endpoint;

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

// Sends a frame through the endpoint using scatter-gather.
// |header| is an 8-byte stack-local control frame header.
// |spans| and |span_count| include the header as spans[0] plus any payload
// spans. Span data must remain valid until the proactor flushes the SQE to
// the kernel — for poll-thread callers (CQE callbacks), this happens during
// submit's Phase 4 before this function returns.
static iree_status_t iree_net_control_channel_send_frame(
    iree_net_control_channel_t* channel, iree_async_span_t* spans,
    iree_host_size_t span_count) {
  iree_net_message_endpoint_send_params_t params = {
      .data = iree_async_span_list_make(spans, span_count),
      .user_data = 0,
  };
  return iree_net_message_endpoint_send(channel->endpoint, &params);
}

//===----------------------------------------------------------------------===//
// Receive path
//===----------------------------------------------------------------------===//

// Handles a received PING frame by auto-responding with PONG.
// The echoed payload references the recv lease buffer directly (safe because
// the lease is valid for the duration of this callback, and the proactor
// flushes the SQE to the kernel during submit).
static iree_status_t iree_net_control_channel_handle_ping(
    iree_net_control_channel_t* channel, iree_const_byte_span_t payload) {
  iree_net_control_frame_flags_t pong_flags = IREE_NET_CONTROL_PONG_FLAG_NONE;
  iree_async_span_t spans[3];
  iree_host_size_t span_count = 1;

  // PONG header.
  iree_net_control_frame_header_t pong_header;
  if (channel->options.append_responder_timestamp) {
    pong_flags |= IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP;
  }
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_PONG,
                                           pong_flags, &pong_header);
  spans[0] = iree_async_span_from_ptr(&pong_header, sizeof(pong_header));

  // Echoed PING payload (zero-copy from recv buffer).
  if (payload.data_length > 0) {
    spans[span_count] =
        iree_async_span_from_ptr((void*)payload.data, payload.data_length);
    ++span_count;
  }

  // Optional responder timestamp.
  uint64_t timestamp_le = 0;
  if (channel->options.append_responder_timestamp) {
    timestamp_le = (uint64_t)iree_time_now();
    spans[span_count] =
        iree_async_span_from_ptr(&timestamp_le, sizeof(timestamp_le));
    ++span_count;
  }

  // Best-effort: if the PONG send fails (backpressure, transport error), the
  // peer will detect liveness timeout and retry or close. Propagating the
  // error to the recv path would incorrectly kill the channel.
  iree_status_t status =
      iree_net_control_channel_send_frame(channel, spans, span_count);
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
    iree_net_message_endpoint_t endpoint,
    iree_net_control_channel_options_t options,
    iree_net_control_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_control_channel_t** out_channel) {
  IREE_TRACE_ZONE_BEGIN(z0);
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
    iree_const_byte_span_t payload) {
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

  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_DATA,
                                           flags, &header);

  iree_async_span_t spans[2];
  iree_host_size_t span_count = 1;
  spans[0] = iree_async_span_from_ptr(&header, sizeof(header));
  if (payload.data_length > 0) {
    spans[span_count] =
        iree_async_span_from_ptr((void*)payload.data, payload.data_length);
    ++span_count;
  }

  iree_status_t status =
      iree_net_control_channel_send_frame(channel, spans, span_count);
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

  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_PING, 0,
                                           &header);

  iree_async_span_t spans[2];
  iree_host_size_t span_count = 1;
  spans[0] = iree_async_span_from_ptr(&header, sizeof(header));
  if (payload.data_length > 0) {
    spans[span_count] =
        iree_async_span_from_ptr((void*)payload.data, payload.data_length);
    ++span_count;
  }

  iree_status_t status =
      iree_net_control_channel_send_frame(channel, spans, span_count);
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

  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_GOAWAY,
                                           0, &header);

  iree_net_control_goaway_payload_t goaway_payload;
  memset(&goaway_payload, 0, sizeof(goaway_payload));
  goaway_payload.reason_code = reason_code;

  iree_async_span_t spans[3];
  iree_host_size_t span_count = 2;
  spans[0] = iree_async_span_from_ptr(&header, sizeof(header));
  spans[1] = iree_async_span_from_ptr(&goaway_payload, sizeof(goaway_payload));
  if (message.size > 0) {
    spans[span_count] =
        iree_async_span_from_ptr((void*)message.data, message.size);
    ++span_count;
  }

  iree_status_t status =
      iree_net_control_channel_send_frame(channel, spans, span_count);
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

  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0,
                                           &header);

  iree_net_control_error_payload_t error_payload;
  memset(&error_payload, 0, sizeof(error_payload));
  error_payload.error_code = (uint32_t)error_code;

  iree_async_span_t spans[3];
  iree_host_size_t span_count = 2;
  spans[0] = iree_async_span_from_ptr(&header, sizeof(header));
  spans[1] = iree_async_span_from_ptr(&error_payload, sizeof(error_payload));
  if (message.size > 0) {
    spans[span_count] =
        iree_async_span_from_ptr((void*)message.data, message.size);
    ++span_count;
  }

  iree_status_t status =
      iree_net_control_channel_send_frame(channel, spans, span_count);
  if (iree_status_is_ok(status)) {
    iree_net_control_channel_set_state(channel,
                                       IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
