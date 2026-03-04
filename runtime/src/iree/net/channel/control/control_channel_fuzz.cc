// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for control channel: tests resilience against arbitrary messages,
// operation sequences, and boundary conditions.
//
// The fuzzer exercises both the receive path (injecting arbitrary bytes as
// messages) and the send path (interleaving send operations with receives to
// explore state machine transitions). All statuses are silently ignored; the
// goal is no crash, no leak, no UB on arbitrary input.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/net/channel/control/control_channel.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/message_endpoint.h"

// Minimum input: 1 byte config + at least 1 byte data.
#define MIN_INPUT_SIZE 2

//===----------------------------------------------------------------------===//
// Mock endpoint
//===----------------------------------------------------------------------===//

// Minimal mock endpoint for fuzzing. Stores callbacks for message injection,
// captures sends (discards the data), and records activation state.
typedef struct fuzz_endpoint_t {
  iree_net_message_endpoint_callbacks_t callbacks;
  bool activated;
  bool inject_send_failures;  // When true, every other send fails.
  uint32_t send_count;        // Tracks sends for alternating failures.
} fuzz_endpoint_t;

static void fuzz_endpoint_set_callbacks(
    void* self, iree_net_message_endpoint_callbacks_t callbacks) {
  fuzz_endpoint_t* endpoint = (fuzz_endpoint_t*)self;
  endpoint->callbacks = callbacks;
}

static iree_status_t fuzz_endpoint_activate(void* self) {
  fuzz_endpoint_t* endpoint = (fuzz_endpoint_t*)self;
  endpoint->activated = true;
  return iree_ok_status();
}

static iree_status_t fuzz_endpoint_deactivate(
    void* self, iree_net_message_endpoint_deactivate_fn_t callback,
    void* user_data) {
  return iree_ok_status();
}

static iree_status_t fuzz_endpoint_send(
    void* self, const iree_net_message_endpoint_send_params_t* params) {
  fuzz_endpoint_t* endpoint = (fuzz_endpoint_t*)self;
  if (endpoint->inject_send_failures && (endpoint->send_count++ % 3 == 1)) {
    return iree_status_from_code(IREE_STATUS_UNAVAILABLE);
  }
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t fuzz_endpoint_query_send_budget(
    void* self) {
  iree_net_carrier_send_budget_t budget;
  memset(&budget, 0, sizeof(budget));
  budget.bytes = 64 * 1024;
  budget.slots = 16;
  return budget;
}

static const iree_net_message_endpoint_vtable_t fuzz_endpoint_vtable = {
    .set_callbacks = fuzz_endpoint_set_callbacks,
    .activate = fuzz_endpoint_activate,
    .deactivate = fuzz_endpoint_deactivate,
    .send = fuzz_endpoint_send,
    .query_send_budget = fuzz_endpoint_query_send_budget,
};

//===----------------------------------------------------------------------===//
// Mock lease
//===----------------------------------------------------------------------===//

typedef struct fuzz_mock_lease_t {
  iree_async_buffer_lease_t lease;
  int release_count;
} fuzz_mock_lease_t;

static void fuzz_mock_release(void* user_data,
                              iree_async_buffer_index_t buffer_index) {
  fuzz_mock_lease_t* mock = (fuzz_mock_lease_t*)user_data;
  ++mock->release_count;
}

static void fuzz_mock_lease_init(fuzz_mock_lease_t* mock, void* data,
                                 size_t size) {
  memset(mock, 0, sizeof(*mock));
  mock->lease.span = iree_async_span_from_ptr(data, size);
  mock->lease.release.fn = fuzz_mock_release;
  mock->lease.release.user_data = mock;
  mock->lease.buffer_index = 0;
}

//===----------------------------------------------------------------------===//
// Fuzz callbacks
//===----------------------------------------------------------------------===//

static iree_status_t fuzz_on_data(void* user_data,
                                  iree_net_control_frame_flags_t flags,
                                  iree_const_byte_span_t payload,
                                  iree_async_buffer_lease_t* lease) {
  // Accept all data.
  return iree_ok_status();
}

static void fuzz_on_goaway(void* user_data, uint32_t reason_code,
                           iree_string_view_t message) {}

static void fuzz_on_error(void* user_data, uint32_t error_code,
                          iree_string_view_t message) {}

static void fuzz_on_pong(void* user_data, iree_const_byte_span_t payload,
                         iree_time_t responder_timestamp_ns) {}

static void fuzz_on_transport_error(void* user_data, iree_status_t status) {
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Injects a raw message into the channel through the mock endpoint callbacks.
static void fuzz_inject_message(fuzz_endpoint_t* endpoint, const uint8_t* data,
                                size_t size) {
  if (!endpoint->callbacks.on_message) return;

  // The channel memcpy's the header to a stack local and passes payload spans
  // by reference (valid for callback duration). The mock lease just needs a
  // valid span; for empty messages we point at a dummy byte.
  uint8_t dummy = 0;
  fuzz_mock_lease_t mock;
  fuzz_mock_lease_init(&mock, size > 0 ? (void*)data : &dummy,
                       size > 0 ? size : 1);

  iree_const_byte_span_t message = iree_make_const_byte_span(data, size);
  iree_status_t status = endpoint->callbacks.on_message(
      endpoint->callbacks.user_data, message, &mock.lease);
  iree_status_ignore(status);
}

// Injects a transport error into the channel.
static void fuzz_inject_transport_error(fuzz_endpoint_t* endpoint) {
  if (!endpoint->callbacks.on_error) return;
  endpoint->callbacks.on_error(
      endpoint->callbacks.user_data,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "fuzz transport error"));
}

//===----------------------------------------------------------------------===//
// Operation stream
//===----------------------------------------------------------------------===//

// Operation opcodes for the operation-stream mode.
// Each operation consumes the opcode byte + a variable-length payload.
enum {
  FUZZ_OP_INJECT_MESSAGE = 0,  // Next byte = length, then length bytes of data.
  FUZZ_OP_SEND_DATA = 1,       // Next byte = flags, then 1 byte length + data.
  FUZZ_OP_SEND_PING = 2,       // Next byte = length, then length bytes payload.
  FUZZ_OP_SEND_GOAWAY =
      3,  // Next 4 bytes = reason, rest = message (1 byte len).
  FUZZ_OP_SEND_ERROR = 4,  // Next byte = error_code, then 1 byte len + message.
  FUZZ_OP_INJECT_ERROR = 5,  // No payload. Injects transport error.
  FUZZ_OP_INJECT_RAW = 6,    // Inject remaining bytes as one raw message.
};
#define FUZZ_OP_COUNT 7

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < MIN_INPUT_SIZE) return 0;

  // Config byte.
  uint8_t config = data[0];
  const uint8_t* stream = data + 1;
  size_t remaining = size - 1;

  // Config bits:
  //   bit 0: append_responder_timestamp
  //   bit 1: operation-stream mode (vs single-message mode)
  //   bit 2: inject send failures (every 3rd send returns UNAVAILABLE)
  bool append_timestamp = (config & 0x01) != 0;
  bool operation_mode = (config & 0x02) != 0;
  bool inject_send_failures = (config & 0x04) != 0;

  // Set up mock endpoint.
  fuzz_endpoint_t mock_endpoint;
  memset(&mock_endpoint, 0, sizeof(mock_endpoint));
  mock_endpoint.inject_send_failures = inject_send_failures;

  iree_net_message_endpoint_t endpoint = {
      .self = &mock_endpoint,
      .vtable = &fuzz_endpoint_vtable,
  };

  // Configure channel.
  iree_net_control_channel_options_t options =
      iree_net_control_channel_options_default();
  options.append_responder_timestamp = append_timestamp;

  iree_net_control_channel_callbacks_t callbacks;
  memset(&callbacks, 0, sizeof(callbacks));
  callbacks.on_data = fuzz_on_data;
  callbacks.on_goaway = fuzz_on_goaway;
  callbacks.on_error = fuzz_on_error;
  callbacks.on_pong = fuzz_on_pong;
  callbacks.on_transport_error = fuzz_on_transport_error;
  callbacks.user_data = NULL;

  // Create and activate channel.
  iree_net_control_channel_t* channel = NULL;
  iree_status_t status = iree_net_control_channel_create(
      endpoint, options, callbacks, iree_allocator_system(), &channel);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  status = iree_net_control_channel_activate(channel);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_net_control_channel_release(channel);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Mode 1: Single-message injection
  //===--------------------------------------------------------------------===//
  // Feed the entire remaining input as one message to the receive path.
  // This is the simplest mode: good for finding parsing bugs in headers and
  // payloads with fully arbitrary bytes.

  if (!operation_mode) {
    fuzz_inject_message(&mock_endpoint, stream, remaining);
    iree_net_control_channel_release(channel);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Mode 2: Operation stream
  //===--------------------------------------------------------------------===//
  // Interprets the remaining bytes as a sequence of operations, interleaving
  // message injection with send calls to explore state machine transitions.

  while (remaining > 0) {
    uint8_t opcode = stream[0] % FUZZ_OP_COUNT;
    stream++;
    remaining--;

    switch (opcode) {
      case FUZZ_OP_INJECT_MESSAGE: {
        // Next byte = message length (clamped to remaining).
        if (remaining == 0) break;
        size_t message_length = stream[0];
        stream++;
        remaining--;
        if (message_length > remaining) message_length = remaining;
        fuzz_inject_message(&mock_endpoint, stream, message_length);
        stream += message_length;
        remaining -= message_length;
        break;
      }

      case FUZZ_OP_SEND_DATA: {
        // Next byte = flags, then 1 byte length + payload.
        if (remaining < 2) {
          remaining = 0;
          break;
        }
        uint8_t flags = stream[0];
        size_t payload_length = stream[1];
        stream += 2;
        remaining -= 2;
        if (payload_length > remaining) payload_length = remaining;
        iree_const_byte_span_t payload =
            iree_make_const_byte_span(stream, payload_length);
        status = iree_net_control_channel_send_data(channel, flags, payload);
        iree_status_ignore(status);
        stream += payload_length;
        remaining -= payload_length;
        break;
      }

      case FUZZ_OP_SEND_PING: {
        // Next byte = length, then payload.
        if (remaining == 0) break;
        size_t payload_length = stream[0];
        stream++;
        remaining--;
        if (payload_length > remaining) payload_length = remaining;
        iree_const_byte_span_t payload =
            iree_make_const_byte_span(stream, payload_length);
        status = iree_net_control_channel_send_ping(channel, payload);
        iree_status_ignore(status);
        stream += payload_length;
        remaining -= payload_length;
        break;
      }

      case FUZZ_OP_SEND_GOAWAY: {
        // Next 4 bytes = reason_code (or fewer if near end), then 1 byte
        // message length + message.
        uint32_t reason_code = 0;
        size_t reason_bytes = remaining < 4 ? remaining : 4;
        memcpy(&reason_code, stream, reason_bytes);
        stream += reason_bytes;
        remaining -= reason_bytes;

        size_t message_length = 0;
        if (remaining > 0) {
          message_length = stream[0];
          stream++;
          remaining--;
          if (message_length > remaining) message_length = remaining;
        }
        iree_string_view_t message =
            iree_make_string_view((const char*)stream, message_length);
        status =
            iree_net_control_channel_send_goaway(channel, reason_code, message);
        iree_status_ignore(status);
        stream += message_length;
        remaining -= message_length;
        break;
      }

      case FUZZ_OP_SEND_ERROR: {
        // Next byte = error_code, then 1 byte message length + message.
        if (remaining == 0) break;
        iree_status_code_t error_code =
            (iree_status_code_t)(stream[0] % (IREE_STATUS_CODE_MASK + 1));
        stream++;
        remaining--;

        size_t message_length = 0;
        if (remaining > 0) {
          message_length = stream[0];
          stream++;
          remaining--;
          if (message_length > remaining) message_length = remaining;
        }
        iree_string_view_t message =
            iree_make_string_view((const char*)stream, message_length);
        status =
            iree_net_control_channel_send_error(channel, error_code, message);
        iree_status_ignore(status);
        stream += message_length;
        remaining -= message_length;
        break;
      }

      case FUZZ_OP_INJECT_ERROR: {
        // No payload. Inject transport error.
        fuzz_inject_transport_error(&mock_endpoint);
        break;
      }

      case FUZZ_OP_INJECT_RAW: {
        // Inject all remaining bytes as one raw message.
        fuzz_inject_message(&mock_endpoint, stream, remaining);
        remaining = 0;
        break;
      }

      default:
        break;
    }
  }

  iree_net_control_channel_release(channel);
  return 0;
}
