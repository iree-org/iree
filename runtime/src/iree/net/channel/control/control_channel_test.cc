// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/control/control_channel.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Mock message endpoint
//===----------------------------------------------------------------------===//

// A captured send operation (concatenated span data).
struct CapturedSend {
  std::vector<uint8_t> data;
  uint64_t user_data;
};

// Mock message endpoint that stores callbacks and captures sends.
// Allows injecting received messages to trigger the installed on_message.
struct MockEndpoint {
  iree_net_message_endpoint_callbacks_t callbacks = {};
  bool activated = false;
  std::vector<CapturedSend> sends;
  iree_status_code_t next_send_error = IREE_STATUS_OK;
  iree_status_code_t next_activate_error = IREE_STATUS_OK;

  // Vtable functions.
  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<MockEndpoint*>(self)->callbacks = callbacks;
  }

  static iree_status_t Activate(void* self) {
    MockEndpoint* mock = static_cast<MockEndpoint*>(self);
    if (mock->next_activate_error != IREE_STATUS_OK) {
      iree_status_code_t error = mock->next_activate_error;
      mock->next_activate_error = IREE_STATUS_OK;
      return iree_status_from_code(error);
    }
    mock->activated = true;
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data) {
    static_cast<MockEndpoint*>(self)->activated = false;
    if (callback) callback(user_data);
    return iree_ok_status();
  }

  static iree_status_t Send(
      void* self, const iree_net_message_endpoint_send_params_t* params) {
    MockEndpoint* mock = static_cast<MockEndpoint*>(self);
    if (mock->next_send_error != IREE_STATUS_OK) {
      iree_status_code_t error = mock->next_send_error;
      mock->next_send_error = IREE_STATUS_OK;
      return iree_status_from_code(error);
    }
    CapturedSend captured;
    captured.user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      uint8_t* ptr = iree_async_span_ptr(params->data.values[i]);
      iree_host_size_t length = params->data.values[i].length;
      captured.data.insert(captured.data.end(), ptr, ptr + length);
    }
    mock->sends.push_back(std::move(captured));
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    iree_net_carrier_send_budget_t budget;
    budget.bytes = 1024 * 1024;
    budget.slots = 64;
    return budget;
  }

  static const iree_net_message_endpoint_vtable_t vtable;

  iree_net_message_endpoint_t as_endpoint() {
    iree_net_message_endpoint_t endpoint;
    endpoint.self = this;
    endpoint.vtable = &vtable;
    return endpoint;
  }

  // Inject a received message (calls installed on_message callback).
  iree_status_t InjectMessage(const std::vector<uint8_t>& data) {
    iree_const_byte_span_t message =
        iree_make_const_byte_span(data.data(), data.size());
    // Use a simple stack lease (no pool backing).
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(const_cast<uint8_t*>(data.data()),
                                          data.size());
    return callbacks.on_message(callbacks.user_data, message, &lease);
  }

  // Inject a transport error.
  void InjectError(iree_status_t status) {
    if (callbacks.on_error) {
      callbacks.on_error(callbacks.user_data, status);
    }
  }
};

const iree_net_message_endpoint_vtable_t MockEndpoint::vtable = {
    MockEndpoint::SetCallbacks,    MockEndpoint::Activate,
    MockEndpoint::Deactivate,      MockEndpoint::Send,
    MockEndpoint::QuerySendBudget,
};

//===----------------------------------------------------------------------===//
// Test context for capturing callbacks
//===----------------------------------------------------------------------===//

struct DataRecord {
  uint8_t flags;
  std::vector<uint8_t> payload;
};

struct GoawayRecord {
  uint32_t reason_code;
  std::string message;
};

struct ErrorRecord {
  uint32_t error_code;
  std::string message;
};

struct PongRecord {
  std::vector<uint8_t> payload;
  iree_time_t responder_timestamp_ns;
};

struct TestContext {
  std::vector<DataRecord> data_records;
  std::vector<GoawayRecord> goaway_records;
  std::vector<ErrorRecord> error_records;
  std::vector<PongRecord> pong_records;
  std::vector<iree_status_code_t> transport_errors;
  iree_status_code_t next_data_error = IREE_STATUS_OK;

  static iree_status_t OnData(void* user_data,
                              iree_net_control_frame_flags_t flags,
                              iree_const_byte_span_t payload,
                              iree_async_buffer_lease_t* lease) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    if (ctx->next_data_error != IREE_STATUS_OK) {
      iree_status_code_t error = ctx->next_data_error;
      ctx->next_data_error = IREE_STATUS_OK;
      return iree_make_status(error, "injected data error");
    }
    DataRecord record;
    record.flags = flags;
    record.payload.assign(payload.data, payload.data + payload.data_length);
    ctx->data_records.push_back(std::move(record));
    return iree_ok_status();
  }

  static void OnGoaway(void* user_data, uint32_t reason_code,
                       iree_string_view_t message) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    GoawayRecord record;
    record.reason_code = reason_code;
    record.message.assign(message.data, message.size);
    ctx->goaway_records.push_back(std::move(record));
  }

  static void OnError(void* user_data, uint32_t error_code,
                      iree_string_view_t message) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    ErrorRecord record;
    record.error_code = error_code;
    record.message.assign(message.data, message.size);
    ctx->error_records.push_back(std::move(record));
  }

  static void OnPong(void* user_data, iree_const_byte_span_t payload,
                     iree_time_t responder_timestamp_ns) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    PongRecord record;
    record.payload.assign(payload.data, payload.data + payload.data_length);
    record.responder_timestamp_ns = responder_timestamp_ns;
    ctx->pong_records.push_back(std::move(record));
  }

  static void OnTransportError(void* user_data, iree_status_t status) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    ctx->transport_errors.push_back(iree_status_code(status));
    iree_status_ignore(status);
  }

  iree_net_control_channel_callbacks_t MakeCallbacks() {
    iree_net_control_channel_callbacks_t callbacks = {};
    callbacks.on_data = OnData;
    callbacks.on_goaway = OnGoaway;
    callbacks.on_error = OnError;
    callbacks.on_pong = OnPong;
    callbacks.on_transport_error = OnTransportError;
    callbacks.user_data = this;
    return callbacks;
  }
};

//===----------------------------------------------------------------------===//
// Frame builders (construct wire-format messages for injection)
//===----------------------------------------------------------------------===//

static std::vector<uint8_t> MakeControlFrame(
    iree_net_control_frame_type_t type, uint8_t flags,
    const std::vector<uint8_t>& payload = {}) {
  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(type, flags, &header);
  std::vector<uint8_t> frame(sizeof(header) + payload.size());
  memcpy(frame.data(), &header, sizeof(header));
  if (!payload.empty()) {
    memcpy(frame.data() + sizeof(header), payload.data(), payload.size());
  }
  return frame;
}

static std::vector<uint8_t> MakePingFrame(
    const std::vector<uint8_t>& payload = {}) {
  return MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_PING, 0, payload);
}

static std::vector<uint8_t> MakePongFrame(const std::vector<uint8_t>& payload,
                                          bool with_timestamp = false,
                                          uint64_t timestamp = 0) {
  uint8_t flags = with_timestamp
                      ? IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP
                      : IREE_NET_CONTROL_PONG_FLAG_NONE;
  std::vector<uint8_t> full_payload = payload;
  if (with_timestamp) {
    uint8_t ts_bytes[8];
    memcpy(ts_bytes, &timestamp, sizeof(timestamp));
    full_payload.insert(full_payload.end(), ts_bytes, ts_bytes + 8);
  }
  return MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_PONG, flags,
                          full_payload);
}

static std::vector<uint8_t> MakeGoawayFrame(uint32_t reason_code,
                                            const std::string& message = "") {
  std::vector<uint8_t> payload(4 + message.size());
  memcpy(payload.data(), &reason_code, 4);
  if (!message.empty()) {
    memcpy(payload.data() + 4, message.data(), message.size());
  }
  return MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_GOAWAY, 0, payload);
}

static std::vector<uint8_t> MakeErrorFrame(uint32_t error_code,
                                           const std::string& message = "") {
  std::vector<uint8_t> payload(4 + message.size());
  memcpy(payload.data(), &error_code, 4);
  if (!message.empty()) {
    memcpy(payload.data() + 4, message.data(), message.size());
  }
  return MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0, payload);
}

static std::vector<uint8_t> MakeDataFrame(
    uint8_t flags, const std::vector<uint8_t>& payload = {}) {
  return MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_DATA, flags, payload);
}

//===----------------------------------------------------------------------===//
// Helper to parse a captured send back into header + payload
//===----------------------------------------------------------------------===//

static iree_net_control_frame_header_t ParseCapturedHeader(
    const CapturedSend& send) {
  EXPECT_GE(send.data.size(), IREE_NET_CONTROL_FRAME_HEADER_SIZE);
  iree_net_control_frame_header_t header;
  memcpy(&header, send.data.data(), sizeof(header));
  return header;
}

static std::vector<uint8_t> CapturedPayload(const CapturedSend& send) {
  if (send.data.size() <= IREE_NET_CONTROL_FRAME_HEADER_SIZE) return {};
  return std::vector<uint8_t>(
      send.data.begin() + IREE_NET_CONTROL_FRAME_HEADER_SIZE, send.data.end());
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class ControlChannelTest : public ::testing::Test {
 protected:
  void SetUp() override { mock_ = std::make_unique<MockEndpoint>(); }

  void TearDown() override {
    iree_net_control_channel_release(channel_);
    channel_ = nullptr;
  }

  void CreateAndActivate(iree_net_control_channel_options_t options =
                             iree_net_control_channel_options_default()) {
    IREE_ASSERT_OK(iree_net_control_channel_create(
        mock_->as_endpoint(), options, ctx_.MakeCallbacks(),
        iree_allocator_system(), &channel_));
    IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  }

  std::unique_ptr<MockEndpoint> mock_;
  TestContext ctx_;
  iree_net_control_channel_t* channel_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, CreateAndRelease) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_CREATED);
}

TEST_F(ControlChannelTest, NullOnDataFails) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_data = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_control_channel_create(
          mock_->as_endpoint(), iree_net_control_channel_options_default(),
          callbacks, iree_allocator_system(), &channel_));
}

TEST_F(ControlChannelTest, ActivateTransitionsToOperational) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_CREATED);
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
  EXPECT_TRUE(mock_->activated);
}

TEST_F(ControlChannelTest, DoubleActivateFails) {
  CreateAndActivate();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_activate(channel_));
}

TEST_F(ControlChannelTest, RetainRelease) {
  CreateAndActivate();
  iree_net_control_channel_retain(channel_);
  iree_net_control_channel_release(channel_);
  // Channel should still be alive.
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

TEST_F(ControlChannelTest, RetainReleaseNull) {
  // Should not crash.
  iree_net_control_channel_retain(nullptr);
  iree_net_control_channel_release(nullptr);
}

//===----------------------------------------------------------------------===//
// Receive PING → auto PONG
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, PingAutoRespondsWithPong) {
  CreateAndActivate();
  std::vector<uint8_t> ping_payload = {0xDE, 0xAD, 0xBE, 0xEF};
  IREE_ASSERT_OK(mock_->InjectMessage(MakePingFrame(ping_payload)));

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_PONG);
  EXPECT_TRUE(iree_net_control_frame_header_has_flag(
      header, IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP));

  // Payload should be echoed ping payload + 8-byte timestamp.
  auto payload = CapturedPayload(mock_->sends[0]);
  ASSERT_EQ(payload.size(), ping_payload.size() + 8);
  EXPECT_EQ(std::vector<uint8_t>(payload.begin(),
                                 payload.begin() + ping_payload.size()),
            ping_payload);
}

TEST_F(ControlChannelTest, PingWithEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(MakePingFrame()));

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_PONG);

  // With timestamp enabled, payload is just 8-byte timestamp.
  auto payload = CapturedPayload(mock_->sends[0]);
  EXPECT_EQ(payload.size(), 8u);
}

TEST_F(ControlChannelTest, PingWithoutTimestamp) {
  iree_net_control_channel_options_t options =
      iree_net_control_channel_options_default();
  options.append_responder_timestamp = false;
  CreateAndActivate(options);

  std::vector<uint8_t> ping_payload = {0x01, 0x02};
  IREE_ASSERT_OK(mock_->InjectMessage(MakePingFrame(ping_payload)));

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_FALSE(iree_net_control_frame_header_has_flag(
      header, IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP));

  // Payload should be just the echoed ping payload (no timestamp).
  auto payload = CapturedPayload(mock_->sends[0]);
  EXPECT_EQ(payload, ping_payload);
}

TEST_F(ControlChannelTest, PingSendFailureDoesNotKillChannel) {
  CreateAndActivate();
  mock_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_ASSERT_OK(mock_->InjectMessage(MakePingFrame({0x01})));

  // Channel should still be OPERATIONAL despite PONG send failure.
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

//===----------------------------------------------------------------------===//
// Receive PONG
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, PongDeliveredToCallback) {
  CreateAndActivate();
  std::vector<uint8_t> echoed = {0x11, 0x22, 0x33};
  IREE_ASSERT_OK(mock_->InjectMessage(MakePongFrame(echoed)));

  ASSERT_EQ(ctx_.pong_records.size(), 1u);
  EXPECT_EQ(ctx_.pong_records[0].payload, echoed);
  EXPECT_EQ(ctx_.pong_records[0].responder_timestamp_ns, 0);
}

TEST_F(ControlChannelTest, PongWithResponderTimestamp) {
  CreateAndActivate();
  std::vector<uint8_t> echoed = {0xAA, 0xBB};
  uint64_t timestamp = 1234567890123456ULL;
  IREE_ASSERT_OK(mock_->InjectMessage(MakePongFrame(echoed, true, timestamp)));

  ASSERT_EQ(ctx_.pong_records.size(), 1u);
  EXPECT_EQ(ctx_.pong_records[0].payload, echoed);
  EXPECT_EQ(ctx_.pong_records[0].responder_timestamp_ns,
            static_cast<iree_time_t>(timestamp));
}

TEST_F(ControlChannelTest, PongWithTimestampFlagButTooShort) {
  CreateAndActivate();
  // HAS_RESPONDER_TIMESTAMP set but only 4 bytes of payload (need 8).
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
  auto frame = MakeControlFrame(
      IREE_NET_CONTROL_FRAME_TYPE_PONG,
      IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP, payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, PongWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_pong = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));

  // Should not crash with NULL on_pong.
  IREE_ASSERT_OK(mock_->InjectMessage(MakePongFrame({0x01}, false)));
}

//===----------------------------------------------------------------------===//
// Receive DATA
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, DataDeliveredToCallback) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0x10, 0x20, 0x30, 0x40};
  uint8_t flags = 0x42;
  IREE_ASSERT_OK(mock_->InjectMessage(MakeDataFrame(flags, payload)));

  ASSERT_EQ(ctx_.data_records.size(), 1u);
  EXPECT_EQ(ctx_.data_records[0].flags, flags);
  EXPECT_EQ(ctx_.data_records[0].payload, payload);
}

TEST_F(ControlChannelTest, DataWithEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(MakeDataFrame(0)));

  ASSERT_EQ(ctx_.data_records.size(), 1u);
  EXPECT_EQ(ctx_.data_records[0].flags, 0);
  EXPECT_TRUE(ctx_.data_records[0].payload.empty());
}

TEST_F(ControlChannelTest, DataCallbackErrorPropagates) {
  CreateAndActivate();
  ctx_.next_data_error = IREE_STATUS_INTERNAL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        mock_->InjectMessage(MakeDataFrame(0, {0x01})));
}

TEST_F(ControlChannelTest, DataDeliveredInDrainingState) {
  CreateAndActivate();
  // Transition to DRAINING via GOAWAY.
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(0, "bye")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  // DATA should still be delivered (in-flight from peer).
  IREE_ASSERT_OK(mock_->InjectMessage(MakeDataFrame(0, {0xAA})));
  ASSERT_EQ(ctx_.data_records.size(), 1u);
  EXPECT_EQ(ctx_.data_records[0].payload, std::vector<uint8_t>({0xAA}));
}

//===----------------------------------------------------------------------===//
// Receive GOAWAY
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, GoawayTransitionsToDraining) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(42, "shutting down")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
  ASSERT_EQ(ctx_.goaway_records.size(), 1u);
  EXPECT_EQ(ctx_.goaway_records[0].reason_code, 42u);
  EXPECT_EQ(ctx_.goaway_records[0].message, "shutting down");
}

TEST_F(ControlChannelTest, GoawayWithEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(0)));

  ASSERT_EQ(ctx_.goaway_records.size(), 1u);
  EXPECT_EQ(ctx_.goaway_records[0].reason_code, 0u);
  EXPECT_TRUE(ctx_.goaway_records[0].message.empty());
}

TEST_F(ControlChannelTest, DoubleGoawayNoDoubleTrans) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(1)));
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(2)));

  // Both delivered, state stays DRAINING.
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
  EXPECT_EQ(ctx_.goaway_records.size(), 2u);
}

TEST_F(ControlChannelTest, GoawayWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_goaway = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));

  // Should not crash, should still transition to DRAINING.
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(0)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
}

TEST_F(ControlChannelTest, GoawayPayloadTooShort) {
  CreateAndActivate();
  // GOAWAY requires at least 4 bytes for reason_code.
  auto frame =
      MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_GOAWAY, 0, {0x01, 0x02});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

//===----------------------------------------------------------------------===//
// Receive ERROR
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, ErrorTransitionsToErrorState) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(
      MakeErrorFrame((uint32_t)IREE_STATUS_INTERNAL, "something broke")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
  ASSERT_EQ(ctx_.error_records.size(), 1u);
  EXPECT_EQ(ctx_.error_records[0].error_code, (uint32_t)IREE_STATUS_INTERNAL);
  EXPECT_EQ(ctx_.error_records[0].message, "something broke");
}

TEST_F(ControlChannelTest, ErrorWithEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(
      mock_->InjectMessage(MakeErrorFrame((uint32_t)IREE_STATUS_UNAVAILABLE)));

  ASSERT_EQ(ctx_.error_records.size(), 1u);
  EXPECT_TRUE(ctx_.error_records[0].message.empty());
}

TEST_F(ControlChannelTest, ErrorPayloadTooShort) {
  CreateAndActivate();
  // ERROR requires at least 4 bytes for error_code.
  auto frame = MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0, {0x01});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, ErrorWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_error = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));

  // Should not crash, should still transition to ERROR.
  IREE_ASSERT_OK(
      mock_->InjectMessage(MakeErrorFrame((uint32_t)IREE_STATUS_INTERNAL)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

//===----------------------------------------------------------------------===//
// Receive malformed frames
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, RuntFrameRejected) {
  CreateAndActivate();
  std::vector<uint8_t> runt = {0x01, 0x02, 0x03};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(runt));
}

TEST_F(ControlChannelTest, EmptyFrameRejected) {
  CreateAndActivate();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, mock_->InjectMessage({}));
}

TEST_F(ControlChannelTest, BadVersionRejected) {
  CreateAndActivate();
  std::vector<uint8_t> frame(8, 0);
  frame[0] = 0xFF;  // Bad version.
  frame[1] = (uint8_t)IREE_NET_CONTROL_FRAME_TYPE_DATA;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, NonzeroReservedBytesRejected) {
  CreateAndActivate();
  // Build a valid-looking frame but with nonzero reserved byte 3.
  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_DATA, 0,
                                           &header);
  header.reserved0 = 0xFF;
  std::vector<uint8_t> frame(sizeof(header));
  memcpy(frame.data(), &header, sizeof(header));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, NonzeroReserved1Rejected) {
  CreateAndActivate();
  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_DATA, 0,
                                           &header);
  header.reserved1 = 0x12345678;
  std::vector<uint8_t> frame(sizeof(header));
  memcpy(frame.data(), &header, sizeof(header));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, UnknownTypeRejected) {
  CreateAndActivate();
  std::vector<uint8_t> frame(8, 0);
  frame[0] = IREE_NET_CONTROL_FRAME_VERSION;
  frame[1] = 0x7F;  // Unknown type.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, ErrorStateRejectsAllMessages) {
  CreateAndActivate();
  // Transition to ERROR.
  IREE_ASSERT_OK(
      mock_->InjectMessage(MakeErrorFrame((uint32_t)IREE_STATUS_INTERNAL)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  // All subsequent messages should be rejected.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        mock_->InjectMessage(MakeDataFrame(0, {0x01})));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        mock_->InjectMessage(MakePingFrame()));
}

//===----------------------------------------------------------------------===//
// Send path — state enforcement
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, SendDataInCreatedFails) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_data(
          channel_, 0, iree_make_const_byte_span(nullptr, 0)));
}

TEST_F(ControlChannelTest, SendDataInOperational) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03};
  IREE_ASSERT_OK(iree_net_control_channel_send_data(
      channel_, 0x55,
      iree_make_const_byte_span(payload.data(), payload.size())));

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_DATA);
  EXPECT_EQ(iree_net_control_frame_header_flags(header), 0x55);
  EXPECT_EQ(CapturedPayload(mock_->sends[0]), payload);
}

TEST_F(ControlChannelTest, SendDataInDrainingFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_data(
          channel_, 0, iree_make_const_byte_span(nullptr, 0)));
}

TEST_F(ControlChannelTest, SendDataInErrorFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, IREE_STATUS_INTERNAL, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_data(
          channel_, 0, iree_make_const_byte_span(nullptr, 0)));
}

TEST_F(ControlChannelTest, SendPingInOperational) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0xCA, 0xFE};
  IREE_ASSERT_OK(iree_net_control_channel_send_ping(
      channel_, iree_make_const_byte_span(payload.data(), payload.size())));

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_PING);
  EXPECT_EQ(CapturedPayload(mock_->sends[0]), payload);
}

TEST_F(ControlChannelTest, SendPingInDrainingFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_ping(
                            channel_, iree_make_const_byte_span(nullptr, 0)));
}

TEST_F(ControlChannelTest, SendGoawayTransitionsToDraining) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 99, iree_make_cstring_view("done")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_GOAWAY);

  auto payload = CapturedPayload(mock_->sends[0]);
  ASSERT_GE(payload.size(), 4u);
  uint32_t reason_code = 0;
  memcpy(&reason_code, payload.data(), 4);
  EXPECT_EQ(reason_code, 99u);
  std::string message(payload.begin() + 4, payload.end());
  EXPECT_EQ(message, "done");
}

TEST_F(ControlChannelTest, SendGoawayInDrainingFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_goaway(
                            channel_, 0, iree_string_view_empty()));
}

TEST_F(ControlChannelTest, SendErrorInOperational) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, IREE_STATUS_ABORTED, iree_make_cstring_view("oops")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  ASSERT_EQ(mock_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_ERROR);

  auto payload = CapturedPayload(mock_->sends[0]);
  ASSERT_GE(payload.size(), 4u);
  uint32_t error_code = 0;
  memcpy(&error_code, payload.data(), 4);
  EXPECT_EQ(error_code, (uint32_t)IREE_STATUS_ABORTED);
  std::string message(payload.begin() + 4, payload.end());
  EXPECT_EQ(message, "oops");
}

TEST_F(ControlChannelTest, SendErrorInDrainingAllowed) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, IREE_STATUS_INTERNAL, iree_string_view_empty()));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

TEST_F(ControlChannelTest, SendErrorInErrorFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, IREE_STATUS_INTERNAL, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_error(channel_, IREE_STATUS_ABORTED,
                                          iree_string_view_empty()));
}

TEST_F(ControlChannelTest, SendErrorInCreatedFails) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_error(channel_, IREE_STATUS_INTERNAL,
                                          iree_string_view_empty()));
}

//===----------------------------------------------------------------------===//
// Send/activate failure paths
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, ActivateFailureKeepsCreated) {
  mock_->next_activate_error = IREE_STATUS_UNAVAILABLE;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_control_channel_activate(channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_CREATED);
  EXPECT_FALSE(mock_->activated);
}

TEST_F(ControlChannelTest, SendGoawayFailureKeepsOperational) {
  CreateAndActivate();
  mock_->next_send_error = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_control_channel_send_goaway(
                            channel_, 0, iree_string_view_empty()));
  // State must remain OPERATIONAL — the GOAWAY was not sent.
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

TEST_F(ControlChannelTest, SendErrorFailureKeepsState) {
  CreateAndActivate();
  mock_->next_send_error = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_net_control_channel_send_error(channel_, IREE_STATUS_INTERNAL,
                                          iree_string_view_empty()));
  // State must remain OPERATIONAL — the ERROR frame was not sent.
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

//===----------------------------------------------------------------------===//
// Send path — wire format verification
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, SendDataEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_data(
      channel_, 0, iree_make_const_byte_span(nullptr, 0)));

  ASSERT_EQ(mock_->sends.size(), 1u);
  // Should be exactly 8 bytes (header only).
  EXPECT_EQ(mock_->sends[0].data.size(), IREE_NET_CONTROL_FRAME_HEADER_SIZE);
  auto header = ParseCapturedHeader(mock_->sends[0]);
  IREE_ASSERT_OK(iree_net_control_frame_header_validate(header));
}

TEST_F(ControlChannelTest, SendPingEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_ping(
      channel_, iree_make_const_byte_span(nullptr, 0)));

  ASSERT_EQ(mock_->sends.size(), 1u);
  EXPECT_EQ(mock_->sends[0].data.size(), IREE_NET_CONTROL_FRAME_HEADER_SIZE);
}

TEST_F(ControlChannelTest, SendGoawayEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));

  ASSERT_EQ(mock_->sends.size(), 1u);
  // Header (8) + goaway_payload (4) = 12 bytes.
  EXPECT_EQ(mock_->sends[0].data.size(), 12u);
}

TEST_F(ControlChannelTest, SendErrorEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, IREE_STATUS_INTERNAL, iree_string_view_empty()));

  ASSERT_EQ(mock_->sends.size(), 1u);
  // Header (8) + error_payload (4) = 12 bytes.
  EXPECT_EQ(mock_->sends[0].data.size(), 12u);
}

//===----------------------------------------------------------------------===//
// State machine integration
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, FullLifecycleOperationalDrainingError) {
  CreateAndActivate();
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);

  // Send some data.
  IREE_ASSERT_OK(iree_net_control_channel_send_data(
      channel_, 0, iree_make_const_byte_span(nullptr, 0)));

  // Initiate shutdown.
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_make_cstring_view("shutting down")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  // Can't send DATA anymore.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_data(
          channel_, 0, iree_make_const_byte_span(nullptr, 0)));

  // Can still send ERROR.
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, IREE_STATUS_INTERNAL, iree_make_cstring_view("fatal")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  // Everything fails in ERROR.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_data(
          channel_, 0, iree_make_const_byte_span(nullptr, 0)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_ping(
                            channel_, iree_make_const_byte_span(nullptr, 0)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_goaway(
                            channel_, 0, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_error(channel_, IREE_STATUS_INTERNAL,
                                          iree_string_view_empty()));
}

TEST_F(ControlChannelTest, RecvDrivenLifecycle) {
  CreateAndActivate();

  // Receive GOAWAY from peer.
  IREE_ASSERT_OK(
      mock_->InjectMessage(MakeGoawayFrame(0, "peer shutting down")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  // Receive ERROR from peer.
  IREE_ASSERT_OK(mock_->InjectMessage(
      MakeErrorFrame((uint32_t)IREE_STATUS_UNAVAILABLE, "gone")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

TEST_F(ControlChannelTest, PingStillWorksInDraining) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_->InjectMessage(MakeGoawayFrame(0)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  // PING auto-PONG should still work in DRAINING.
  mock_->sends.clear();
  IREE_ASSERT_OK(mock_->InjectMessage(MakePingFrame({0x01})));
  EXPECT_EQ(mock_->sends.size(), 1u);
}

//===----------------------------------------------------------------------===//
// Transport error handling
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, TransportErrorTransitionsToError) {
  CreateAndActivate();
  mock_->InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "connection lost"));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
  ASSERT_EQ(ctx_.transport_errors.size(), 1u);
  EXPECT_EQ(ctx_.transport_errors[0], IREE_STATUS_UNAVAILABLE);
}

TEST_F(ControlChannelTest, TransportErrorWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_transport_error = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_->as_endpoint(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));

  // Should not crash and should still transition.
  mock_->InjectError(iree_make_status(IREE_STATUS_INTERNAL, "error"));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

}  // namespace
}  // namespace net
}  // namespace iree
