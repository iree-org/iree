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
#include "iree/net/carrier.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"
#include "iree/net/status_wire.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Test buffer pool
//===----------------------------------------------------------------------===//

// Creates a real buffer pool backed by a test region. Avoids mocking the pool.
class TestBufferPool {
 public:
  TestBufferPool(iree_host_size_t buffer_count, iree_host_size_t buffer_size)
      : buffer_count_(buffer_count), buffer_size_(buffer_size) {
    buffer_memory_.resize(buffer_count * buffer_size, 0);
    region_ =
        static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
    memset(region_, 0, sizeof(*region_));
    iree_atomic_ref_count_init(&region_->ref_count);
    region_->destroy_fn = DestroyRegion;
    region_->base_ptr = buffer_memory_.data();
    region_->length = buffer_memory_.size();
    region_->buffer_size = buffer_size;
    region_->buffer_count = static_cast<uint32_t>(buffer_count);

    iree_status_t status = iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &pool_);
    IREE_CHECK_OK(status);
    iree_async_region_release(region_);
  }

  ~TestBufferPool() {
    if (pool_) iree_async_buffer_pool_free(pool_);
  }

  iree_async_buffer_pool_t* get() { return pool_; }

 private:
  static void DestroyRegion(iree_async_region_t* region) { free(region); }

  iree_host_size_t buffer_count_;
  iree_host_size_t buffer_size_;
  std::vector<uint8_t> buffer_memory_;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* pool_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Mock carrier for send path
//===----------------------------------------------------------------------===//

// A captured carrier send operation (concatenated span data).
struct CapturedSend {
  std::vector<uint8_t> data;
  uint64_t user_data;
};

// Mock carrier that captures sends and auto-completes them via the callback.
// The control channel's frame_sender routes all sends through this carrier.
struct MockCarrier {
  iree_net_carrier_t base;
  std::vector<CapturedSend> sends;
  iree_status_code_t next_send_error = IREE_STATUS_OK;

  static void Destroy(iree_net_carrier_t* carrier) {}

  static void SetRecvHandler(iree_net_carrier_t* carrier,
                             iree_net_carrier_recv_handler_t handler) {
    carrier->recv_handler = handler;
  }

  static iree_status_t Activate(iree_net_carrier_t* carrier) {
    iree_net_carrier_set_state(carrier, IREE_NET_CARRIER_STATE_ACTIVE);
    return iree_ok_status();
  }

  static iree_status_t Deactivate(
      iree_net_carrier_t* carrier,
      iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
    iree_net_carrier_set_state(carrier, IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) callback(user_data);
    return iree_ok_status();
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(
      iree_net_carrier_t* carrier) {
    return {1024 * 1024, 64};
  }

  static iree_status_t Send(iree_net_carrier_t* carrier,
                            const iree_net_send_params_t* params) {
    MockCarrier* mock = reinterpret_cast<MockCarrier*>(carrier);

    if (mock->next_send_error != IREE_STATUS_OK) {
      iree_status_code_t error = mock->next_send_error;
      mock->next_send_error = IREE_STATUS_OK;
      return iree_status_from_code(error);
    }

    CapturedSend captured;
    captured.user_data = params->user_data;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      iree_async_span_t span = params->data.values[i];
      uint8_t* ptr = iree_async_span_ptr(span);
      captured.data.insert(captured.data.end(), ptr, ptr + span.length);
    }
    mock->sends.push_back(std::move(captured));

    // Auto-complete: fire the carrier callback (frame_sender dispatch).
    carrier->callback.fn(carrier->callback.user_data, params->user_data,
                         iree_ok_status(), 0, nullptr);
    return iree_ok_status();
  }

  static iree_status_t BeginSend(iree_net_carrier_t* carrier,
                                 iree_host_size_t size, void** out_ptr,
                                 iree_net_carrier_send_handle_t* out_handle) {
    IREE_ASSERT(false && "begin_send not implemented on mock carrier");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t CommitSend(iree_net_carrier_t* carrier,
                                  iree_net_carrier_send_handle_t handle) {
    IREE_ASSERT(false && "commit_send not implemented on mock carrier");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void AbortSend(iree_net_carrier_t* carrier,
                        iree_net_carrier_send_handle_t handle) {
    IREE_ASSERT(false && "abort_send not implemented on mock carrier");
  }

  static iree_status_t Shutdown(iree_net_carrier_t* carrier) {
    return iree_ok_status();
  }

  static iree_status_t DirectWrite(
      iree_net_carrier_t* carrier,
      const iree_net_direct_write_params_t* params) {
    IREE_ASSERT(false && "direct_write not implemented on mock carrier");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t DirectRead(iree_net_carrier_t* carrier,
                                  const iree_net_direct_read_params_t* params) {
    IREE_ASSERT(false && "direct_read not implemented on mock carrier");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static iree_status_t RegisterBuffer(iree_net_carrier_t* carrier,
                                      iree_async_region_t* region,
                                      iree_net_remote_handle_t* out_handle) {
    IREE_ASSERT(false && "register_buffer not implemented on mock carrier");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "mock");
  }

  static void UnregisterBuffer(iree_net_carrier_t* carrier,
                               iree_net_remote_handle_t handle) {
    IREE_ASSERT(false && "unregister_buffer not implemented on mock carrier");
  }

  static const iree_net_carrier_vtable_t kVtable;

  static std::unique_ptr<MockCarrier> Create() {
    auto mock = std::make_unique<MockCarrier>();
    iree_net_carrier_callback_t send_callback;
    send_callback.fn = iree_net_frame_sender_dispatch_carrier_completion;
    send_callback.user_data = nullptr;
    iree_net_carrier_initialize(&kVtable, IREE_NET_CARRIER_CAPABILITY_RELIABLE,
                                0, 8, send_callback, iree_allocator_system(),
                                &mock->base);
    return mock;
  }
};

const iree_net_carrier_vtable_t MockCarrier::kVtable = {
    MockCarrier::Destroy,         MockCarrier::SetRecvHandler,
    MockCarrier::Activate,        MockCarrier::Deactivate,
    MockCarrier::QuerySendBudget, MockCarrier::Send,
    MockCarrier::BeginSend,       MockCarrier::CommitSend,
    MockCarrier::AbortSend,       MockCarrier::Shutdown,
    MockCarrier::DirectWrite,     MockCarrier::DirectRead,
    MockCarrier::RegisterBuffer,  MockCarrier::UnregisterBuffer,
};

//===----------------------------------------------------------------------===//
// Mock message endpoint (receive path only)
//===----------------------------------------------------------------------===//

// Mock message endpoint that stores callbacks for receive injection and
// forwards send operations to the carrier via the frame_sender dispatch path.
struct MockEndpoint {
  iree_net_message_endpoint_callbacks_t callbacks = {};
  MockCarrier* carrier = nullptr;  // For send forwarding.
  bool activated = false;
  iree_status_code_t next_activate_error = IREE_STATUS_OK;

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
    iree_net_send_params_t carrier_params = {};
    carrier_params.data = params->data;
    carrier_params.flags = IREE_NET_SEND_FLAG_NONE;
    carrier_params.user_data = params->user_data;
    return iree_net_carrier_send(&mock->carrier->base, &carrier_params);
  }

  static iree_net_carrier_send_budget_t QuerySendBudget(void* self) {
    return {1024 * 1024, 64};
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
  iree_status_code_t status_code;
  std::string message;    // Primary message from iree_status_message().
  std::string formatted;  // Full formatted output from iree_status_format().
};

struct PongRecord {
  std::vector<uint8_t> payload;
  iree_time_t responder_timestamp_ns;
};

struct SendCompleteRecord {
  uint64_t operation_user_data;
  iree_status_code_t status_code;
};

struct TestContext {
  std::vector<DataRecord> data_records;
  std::vector<GoawayRecord> goaway_records;
  std::vector<ErrorRecord> error_records;
  std::vector<PongRecord> pong_records;
  std::vector<iree_status_code_t> transport_errors;
  std::vector<SendCompleteRecord> send_completes;
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

  static void OnError(void* user_data, iree_status_t status) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    ErrorRecord record;
    record.status_code = iree_status_code(status);
    iree_string_view_t message = iree_status_message(status);
    record.message.assign(message.data, message.size);
    char format_buffer[1024];
    iree_host_size_t format_length = 0;
    iree_status_format(status, sizeof(format_buffer), format_buffer,
                       &format_length);
    record.formatted.assign(format_buffer, format_length);
    ctx->error_records.push_back(std::move(record));
    iree_status_ignore(status);
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

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    SendCompleteRecord record;
    record.operation_user_data = operation_user_data;
    record.status_code = iree_status_code(status);
    ctx->send_completes.push_back(record);
    iree_status_ignore(status);
  }

  iree_net_control_channel_callbacks_t MakeCallbacks() {
    iree_net_control_channel_callbacks_t callbacks = {};
    callbacks.on_data = OnData;
    callbacks.on_goaway = OnGoaway;
    callbacks.on_error = OnError;
    callbacks.on_pong = OnPong;
    callbacks.on_transport_error = OnTransportError;
    callbacks.on_send_complete = OnSendComplete;
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

// Constructs an ERROR frame with a status_wire payload from a status code
// and optional message. The status is serialized via status_wire, matching
// what the control channel produces on the send path.
static std::vector<uint8_t> MakeErrorFrame(iree_status_code_t code,
                                           const std::string& message = "") {
  iree_status_t status =
      message.empty() ? iree_status_from_code(code)
                      : iree_status_allocate_f(code, /*file=*/NULL, /*line=*/0,
                                               "%s", message.c_str());
  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(status, &wire_size);
  std::vector<uint8_t> wire_data(wire_size);
  IREE_CHECK_OK(iree_net_status_wire_serialize(
      status, iree_make_byte_span(wire_data.data(), wire_data.size())));
  iree_status_ignore(status);
  return MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0, wire_data);
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
  void SetUp() override {
    mock_carrier_ = MockCarrier::Create();
    mock_endpoint_ = std::make_unique<MockEndpoint>();
    mock_endpoint_->carrier = mock_carrier_.get();
    // 32 buffers of 256 bytes — enough for all control frame headers.
    header_pool_ = std::make_unique<TestBufferPool>(32, 256);
  }

  void TearDown() override {
    iree_net_control_channel_release(channel_);
    channel_ = nullptr;
  }

  void CreateAndActivate(iree_net_control_channel_options_t options =
                             iree_net_control_channel_options_default()) {
    IREE_ASSERT_OK(iree_net_control_channel_create(
        mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
        header_pool_->get(), options, ctx_.MakeCallbacks(),
        iree_allocator_system(), &channel_));
    IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  }

  std::unique_ptr<MockEndpoint> mock_endpoint_;
  std::unique_ptr<MockCarrier> mock_carrier_;
  std::unique_ptr<TestBufferPool> header_pool_;
  TestContext ctx_;
  iree_net_control_channel_t* channel_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, CreateAndRelease) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
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
          mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
          header_pool_->get(), iree_net_control_channel_options_default(),
          callbacks, iree_allocator_system(), &channel_));
}

TEST_F(ControlChannelTest, ActivateTransitionsToOperational) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_CREATED);
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
  EXPECT_TRUE(mock_endpoint_->activated);
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
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

TEST_F(ControlChannelTest, RetainReleaseNull) {
  iree_net_control_channel_retain(nullptr);
  iree_net_control_channel_release(nullptr);
}

//===----------------------------------------------------------------------===//
// Receive PING → auto PONG
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, PingAutoRespondsWithPong) {
  CreateAndActivate();
  std::vector<uint8_t> ping_payload = {0xDE, 0xAD, 0xBE, 0xEF};
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePingFrame(ping_payload)));

  // PONG goes through the carrier (via frame_sender).
  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_PONG);
  EXPECT_TRUE(iree_net_control_frame_header_has_flag(
      header, IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP));

  // Payload should be echoed ping payload + 8-byte timestamp.
  auto payload = CapturedPayload(mock_carrier_->sends[0]);
  ASSERT_EQ(payload.size(), ping_payload.size() + 8);
  EXPECT_EQ(std::vector<uint8_t>(payload.begin(),
                                 payload.begin() + ping_payload.size()),
            ping_payload);
}

TEST_F(ControlChannelTest, PingWithEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePingFrame()));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_PONG);
  auto payload = CapturedPayload(mock_carrier_->sends[0]);
  EXPECT_EQ(payload.size(), 8u);
}

TEST_F(ControlChannelTest, PingWithoutTimestamp) {
  iree_net_control_channel_options_t options =
      iree_net_control_channel_options_default();
  options.append_responder_timestamp = false;
  CreateAndActivate(options);

  std::vector<uint8_t> ping_payload = {0x01, 0x02};
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePingFrame(ping_payload)));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_FALSE(iree_net_control_frame_header_has_flag(
      header, IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP));
  auto payload = CapturedPayload(mock_carrier_->sends[0]);
  EXPECT_EQ(payload, ping_payload);
}

TEST_F(ControlChannelTest, PingSendFailureDoesNotKillChannel) {
  CreateAndActivate();
  mock_carrier_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePingFrame({0x01})));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

//===----------------------------------------------------------------------===//
// Receive PONG
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, PongDeliveredToCallback) {
  CreateAndActivate();
  std::vector<uint8_t> echoed = {0x11, 0x22, 0x33};
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePongFrame(echoed)));

  ASSERT_EQ(ctx_.pong_records.size(), 1u);
  EXPECT_EQ(ctx_.pong_records[0].payload, echoed);
  EXPECT_EQ(ctx_.pong_records[0].responder_timestamp_ns, 0);
}

TEST_F(ControlChannelTest, PongWithResponderTimestamp) {
  CreateAndActivate();
  std::vector<uint8_t> echoed = {0xAA, 0xBB};
  uint64_t timestamp = 1234567890123456ULL;
  IREE_ASSERT_OK(
      mock_endpoint_->InjectMessage(MakePongFrame(echoed, true, timestamp)));

  ASSERT_EQ(ctx_.pong_records.size(), 1u);
  EXPECT_EQ(ctx_.pong_records[0].payload, echoed);
  EXPECT_EQ(ctx_.pong_records[0].responder_timestamp_ns,
            static_cast<iree_time_t>(timestamp));
}

TEST_F(ControlChannelTest, PongWithTimestampFlagButTooShort) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
  auto frame = MakeControlFrame(
      IREE_NET_CONTROL_FRAME_TYPE_PONG,
      IREE_NET_CONTROL_PONG_FLAG_HAS_RESPONDER_TIMESTAMP, payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, PongWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_pong = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePongFrame({0x01}, false)));
}

//===----------------------------------------------------------------------===//
// Receive DATA
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, DataDeliveredToCallback) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0x10, 0x20, 0x30, 0x40};
  uint8_t flags = 0x42;
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeDataFrame(flags, payload)));

  ASSERT_EQ(ctx_.data_records.size(), 1u);
  EXPECT_EQ(ctx_.data_records[0].flags, flags);
  EXPECT_EQ(ctx_.data_records[0].payload, payload);
}

TEST_F(ControlChannelTest, DataWithEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeDataFrame(0)));

  ASSERT_EQ(ctx_.data_records.size(), 1u);
  EXPECT_EQ(ctx_.data_records[0].flags, 0);
  EXPECT_TRUE(ctx_.data_records[0].payload.empty());
}

TEST_F(ControlChannelTest, DataCallbackErrorPropagates) {
  CreateAndActivate();
  ctx_.next_data_error = IREE_STATUS_INTERNAL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, mock_endpoint_->InjectMessage(
                                                  MakeDataFrame(0, {0x01})));
}

TEST_F(ControlChannelTest, DataDeliveredInDrainingState) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeGoawayFrame(0, "bye")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeDataFrame(0, {0xAA})));
  ASSERT_EQ(ctx_.data_records.size(), 1u);
  EXPECT_EQ(ctx_.data_records[0].payload, std::vector<uint8_t>({0xAA}));
}

//===----------------------------------------------------------------------===//
// Receive GOAWAY
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, GoawayTransitionsToDraining) {
  CreateAndActivate();
  IREE_ASSERT_OK(
      mock_endpoint_->InjectMessage(MakeGoawayFrame(42, "shutting down")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
  ASSERT_EQ(ctx_.goaway_records.size(), 1u);
  EXPECT_EQ(ctx_.goaway_records[0].reason_code, 42u);
  EXPECT_EQ(ctx_.goaway_records[0].message, "shutting down");
}

TEST_F(ControlChannelTest, GoawayWithEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeGoawayFrame(0)));

  ASSERT_EQ(ctx_.goaway_records.size(), 1u);
  EXPECT_EQ(ctx_.goaway_records[0].reason_code, 0u);
  EXPECT_TRUE(ctx_.goaway_records[0].message.empty());
}

TEST_F(ControlChannelTest, DoubleGoawayNoDoubleTrans) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeGoawayFrame(1)));
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeGoawayFrame(2)));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
  EXPECT_EQ(ctx_.goaway_records.size(), 2u);
}

TEST_F(ControlChannelTest, GoawayWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_goaway = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeGoawayFrame(0)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);
}

TEST_F(ControlChannelTest, GoawayPayloadTooShort) {
  CreateAndActivate();
  auto frame =
      MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_GOAWAY, 0, {0x01, 0x02});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(frame));
}

//===----------------------------------------------------------------------===//
// Receive ERROR
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, ErrorTransitionsToErrorState) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(
      MakeErrorFrame(IREE_STATUS_INTERNAL, "something broke")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
  ASSERT_EQ(ctx_.error_records.size(), 1u);
  EXPECT_EQ(ctx_.error_records[0].status_code, IREE_STATUS_INTERNAL);
  EXPECT_EQ(ctx_.error_records[0].message, "something broke");
}

TEST_F(ControlChannelTest, ErrorWithEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(
      mock_endpoint_->InjectMessage(MakeErrorFrame(IREE_STATUS_UNAVAILABLE)));

  ASSERT_EQ(ctx_.error_records.size(), 1u);
  EXPECT_TRUE(ctx_.error_records[0].message.empty());
}

TEST_F(ControlChannelTest, ErrorPayloadTooShort) {
  CreateAndActivate();
  // 4 bytes is too short for the 8-byte status_wire header.
  auto frame = MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0,
                                {0x01, 0x02, 0x03, 0x04});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, ErrorWithNullCallback) {
  iree_net_control_channel_callbacks_t callbacks = ctx_.MakeCallbacks();
  callbacks.on_error = nullptr;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  IREE_ASSERT_OK(
      mock_endpoint_->InjectMessage(MakeErrorFrame(IREE_STATUS_INTERNAL)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

//===----------------------------------------------------------------------===//
// Structured error round-trips
//===----------------------------------------------------------------------===//

// Verifies that a rich status with source location and annotations survives
// the send_error → wire → on_error round-trip.
TEST_F(ControlChannelTest, StructuredErrorRoundTrip) {
  CreateAndActivate();

  // Build a status with annotations — the kind of error a real HAL driver
  // would produce when something goes wrong deep in the stack.
  iree_status_t error =
      iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "device OOM");
  error = iree_status_annotate(
      error, iree_make_cstring_view("allocating command buffer"));
  error = iree_status_annotate(error,
                               iree_make_cstring_view("during queue_execute"));

  // Send it.
  IREE_ASSERT_OK(iree_net_control_channel_send_error(channel_, error));

  // Verify the channel transitioned to ERROR.
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  // Deserialize the captured wire data and verify structured content.
  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto payload = CapturedPayload(mock_carrier_->sends[0]);
  iree_status_t deserialized = iree_ok_status();
  IREE_ASSERT_OK(iree_net_status_wire_deserialize(
      iree_make_const_byte_span(payload.data(), payload.size()),
      &deserialized));
  EXPECT_EQ(iree_status_code(deserialized), IREE_STATUS_RESOURCE_EXHAUSTED);

  // The primary message should survive.
  iree_string_view_t message = iree_status_message(deserialized);
  EXPECT_EQ(std::string(message.data, message.size), "device OOM");

  // The full formatted output should include annotations.
  char format_buffer[1024];
  iree_host_size_t format_length = 0;
  iree_status_format(deserialized, sizeof(format_buffer), format_buffer,
                     &format_length);
  std::string formatted(format_buffer, format_length);
  EXPECT_NE(formatted.find("device OOM"), std::string::npos);
  EXPECT_NE(formatted.find("allocating command buffer"), std::string::npos);
  EXPECT_NE(formatted.find("during queue_execute"), std::string::npos);

  iree_status_ignore(deserialized);
}

// Verifies that injecting a structured ERROR frame with annotations delivers
// the full status to the on_error callback.
TEST_F(ControlChannelTest, ReceiveStructuredError) {
  CreateAndActivate();

  // Build a wire frame from a rich status (simulates what the remote peer
  // would produce via send_error).
  iree_status_t remote_error =
      iree_make_status(IREE_STATUS_DATA_LOSS, "checksum mismatch");
  remote_error = iree_status_annotate(
      remote_error, iree_make_cstring_view("reading block 42"));

  iree_host_size_t wire_size = 0;
  iree_net_status_wire_size(remote_error, &wire_size);
  std::vector<uint8_t> wire_data(wire_size);
  IREE_CHECK_OK(iree_net_status_wire_serialize(
      remote_error, iree_make_byte_span(wire_data.data(), wire_data.size())));
  iree_status_ignore(remote_error);

  auto frame =
      MakeControlFrame(IREE_NET_CONTROL_FRAME_TYPE_ERROR, 0, wire_data);
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(frame));

  // The on_error callback should have received the full structured status.
  ASSERT_EQ(ctx_.error_records.size(), 1u);
  EXPECT_EQ(ctx_.error_records[0].status_code, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(ctx_.error_records[0].message, "checksum mismatch");
  EXPECT_NE(ctx_.error_records[0].formatted.find("reading block 42"),
            std::string::npos);
}

// Verifies that source location survives the round-trip when present.
TEST_F(ControlChannelTest, ErrorSourceLocationRoundTrip) {
  CreateAndActivate();

  // iree_make_status captures __FILE__ and __LINE__.
  iree_status_t error =
      iree_make_status(IREE_STATUS_INTERNAL, "test source location");
  iree_status_source_location_t original_location =
      iree_status_source_location(error);

  IREE_ASSERT_OK(iree_net_control_channel_send_error(channel_, error));

  // Deserialize the wire data.
  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto payload = CapturedPayload(mock_carrier_->sends[0]);
  iree_status_t deserialized = iree_ok_status();
  IREE_ASSERT_OK(iree_net_status_wire_deserialize(
      iree_make_const_byte_span(payload.data(), payload.size()),
      &deserialized));

  if (original_location.file) {
    iree_status_source_location_t deserialized_location =
        iree_status_source_location(deserialized);
    ASSERT_NE(deserialized_location.file, nullptr);
    EXPECT_STREQ(deserialized_location.file, original_location.file);
    EXPECT_EQ(deserialized_location.line, original_location.line);
  }

  iree_status_ignore(deserialized);
}

//===----------------------------------------------------------------------===//
// Receive malformed frames
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, RuntFrameRejected) {
  CreateAndActivate();
  std::vector<uint8_t> runt = {0x01, 0x02, 0x03};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(runt));
}

TEST_F(ControlChannelTest, EmptyFrameRejected) {
  CreateAndActivate();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage({}));
}

TEST_F(ControlChannelTest, BadVersionRejected) {
  CreateAndActivate();
  std::vector<uint8_t> frame(8, 0);
  frame[0] = 0xFF;
  frame[1] = (uint8_t)IREE_NET_CONTROL_FRAME_TYPE_DATA;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, NonzeroReservedBytesRejected) {
  CreateAndActivate();
  iree_net_control_frame_header_t header;
  iree_net_control_frame_header_initialize(IREE_NET_CONTROL_FRAME_TYPE_DATA, 0,
                                           &header);
  header.reserved0 = 0xFF;
  std::vector<uint8_t> frame(sizeof(header));
  memcpy(frame.data(), &header, sizeof(header));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(frame));
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
                        mock_endpoint_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, UnknownTypeRejected) {
  CreateAndActivate();
  std::vector<uint8_t> frame(8, 0);
  frame[0] = IREE_NET_CONTROL_FRAME_VERSION;
  frame[1] = 0x7F;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        mock_endpoint_->InjectMessage(frame));
}

TEST_F(ControlChannelTest, ErrorStateRejectsAllMessages) {
  CreateAndActivate();
  IREE_ASSERT_OK(
      mock_endpoint_->InjectMessage(MakeErrorFrame(IREE_STATUS_INTERNAL)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      mock_endpoint_->InjectMessage(MakeDataFrame(0, {0x01})));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        mock_endpoint_->InjectMessage(MakePingFrame()));
}

//===----------------------------------------------------------------------===//
// Send path — state enforcement
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, SendDataInCreatedFails) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_data(
                            channel_, 0, iree_async_span_list_empty(), 0));
}

TEST_F(ControlChannelTest, SendDataInOperational) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03};
  iree_async_span_t span =
      iree_async_span_from_ptr(payload.data(), payload.size());
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(
      iree_net_control_channel_send_data(channel_, 0x55, span_list,
                                         /*operation_user_data=*/42));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_DATA);
  EXPECT_EQ(iree_net_control_frame_header_flags(header), 0x55);
  EXPECT_EQ(CapturedPayload(mock_carrier_->sends[0]), payload);

  // Verify completion callback fired.
  ASSERT_EQ(ctx_.send_completes.size(), 1u);
  EXPECT_EQ(ctx_.send_completes[0].operation_user_data, 42u);
  EXPECT_EQ(ctx_.send_completes[0].status_code, IREE_STATUS_OK);
}

TEST_F(ControlChannelTest, SendDataInDrainingFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_data(
                            channel_, 0, iree_async_span_list_empty(), 0));
}

TEST_F(ControlChannelTest, SendDataInErrorFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_data(
                            channel_, 0, iree_async_span_list_empty(), 0));
}

TEST_F(ControlChannelTest, SendPingInOperational) {
  CreateAndActivate();
  std::vector<uint8_t> payload = {0xCA, 0xFE};
  IREE_ASSERT_OK(iree_net_control_channel_send_ping(
      channel_, iree_make_const_byte_span(payload.data(), payload.size())));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_PING);
  EXPECT_EQ(CapturedPayload(mock_carrier_->sends[0]), payload);
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

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_GOAWAY);

  auto payload = CapturedPayload(mock_carrier_->sends[0]);
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
      channel_, iree_make_status(IREE_STATUS_ABORTED, "oops")));

  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  EXPECT_EQ(iree_net_control_frame_header_type(header),
            IREE_NET_CONTROL_FRAME_TYPE_ERROR);

  // Verify the payload is a valid status_wire blob.
  auto payload = CapturedPayload(mock_carrier_->sends[0]);
  iree_status_t deserialized = iree_ok_status();
  IREE_ASSERT_OK(iree_net_status_wire_deserialize(
      iree_make_const_byte_span(payload.data(), payload.size()),
      &deserialized));
  EXPECT_EQ(iree_status_code(deserialized), IREE_STATUS_ABORTED);
  iree_string_view_t message = iree_status_message(deserialized);
  EXPECT_EQ(std::string(message.data, message.size), "oops");
  iree_status_ignore(deserialized);
}

TEST_F(ControlChannelTest, SendErrorInDrainingAllowed) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

TEST_F(ControlChannelTest, SendErrorInErrorFails) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_error(
          channel_, iree_status_from_code(IREE_STATUS_ABORTED)));
}

TEST_F(ControlChannelTest, SendErrorInCreatedFails) {
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_error(
          channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));
}

//===----------------------------------------------------------------------===//
// Send/activate failure paths
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, ActivateFailureKeepsCreated) {
  mock_endpoint_->next_activate_error = IREE_STATUS_UNAVAILABLE;
  IREE_ASSERT_OK(iree_net_control_channel_create(
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      ctx_.MakeCallbacks(), iree_allocator_system(), &channel_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_control_channel_activate(channel_));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_CREATED);
  EXPECT_FALSE(mock_endpoint_->activated);
}

TEST_F(ControlChannelTest, SendGoawayFailureKeepsOperational) {
  CreateAndActivate();
  mock_carrier_->next_send_error = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_control_channel_send_goaway(
                            channel_, 0, iree_string_view_empty()));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

TEST_F(ControlChannelTest, SendErrorFailureKeepsState) {
  CreateAndActivate();
  mock_carrier_->next_send_error = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_net_control_channel_send_error(
          channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_OPERATIONAL);
}

//===----------------------------------------------------------------------===//
// Send path — wire format verification
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, SendDataEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_data(
      channel_, 0, iree_async_span_list_empty(), 0));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  EXPECT_EQ(mock_carrier_->sends[0].data.size(),
            IREE_NET_CONTROL_FRAME_HEADER_SIZE);
  auto header = ParseCapturedHeader(mock_carrier_->sends[0]);
  IREE_ASSERT_OK(iree_net_control_frame_header_validate(header));
}

TEST_F(ControlChannelTest, SendPingEmptyPayload) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_ping(
      channel_, iree_make_const_byte_span(nullptr, 0)));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  EXPECT_EQ(mock_carrier_->sends[0].data.size(),
            IREE_NET_CONTROL_FRAME_HEADER_SIZE);
}

TEST_F(ControlChannelTest, SendGoawayEmptyMessage) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_string_view_empty()));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  // Header (8) + goaway_payload (4) = 12 bytes.
  EXPECT_EQ(mock_carrier_->sends[0].data.size(), 12u);
}

TEST_F(ControlChannelTest, SendErrorCodeOnly) {
  CreateAndActivate();
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  // Control frame header (8) + status_wire header (8) = 16 bytes.
  EXPECT_EQ(mock_carrier_->sends[0].data.size(), 16u);
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
      channel_, 0, iree_async_span_list_empty(), 0));

  // Initiate shutdown.
  IREE_ASSERT_OK(iree_net_control_channel_send_goaway(
      channel_, 0, iree_make_cstring_view("shutting down")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  // Can't send DATA anymore.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_data(
                            channel_, 0, iree_async_span_list_empty(), 0));

  // Can still send ERROR.
  IREE_ASSERT_OK(iree_net_control_channel_send_error(
      channel_, iree_make_status(IREE_STATUS_INTERNAL, "fatal")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);

  // Everything fails in ERROR.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_data(
                            channel_, 0, iree_async_span_list_empty(), 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_ping(
                            channel_, iree_make_const_byte_span(nullptr, 0)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_control_channel_send_goaway(
                            channel_, 0, iree_string_view_empty()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_control_channel_send_error(
          channel_, iree_status_from_code(IREE_STATUS_INTERNAL)));
}

TEST_F(ControlChannelTest, RecvDrivenLifecycle) {
  CreateAndActivate();

  IREE_ASSERT_OK(
      mock_endpoint_->InjectMessage(MakeGoawayFrame(0, "peer shutting down")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(
      MakeErrorFrame(IREE_STATUS_UNAVAILABLE, "gone")));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

TEST_F(ControlChannelTest, PingStillWorksInDraining) {
  CreateAndActivate();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakeGoawayFrame(0)));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_DRAINING);

  mock_carrier_->sends.clear();
  IREE_ASSERT_OK(mock_endpoint_->InjectMessage(MakePingFrame({0x01})));
  EXPECT_EQ(mock_carrier_->sends.size(), 1u);
}

//===----------------------------------------------------------------------===//
// Transport error handling
//===----------------------------------------------------------------------===//

TEST_F(ControlChannelTest, TransportErrorTransitionsToError) {
  CreateAndActivate();
  mock_endpoint_->InjectError(
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
      mock_endpoint_->as_endpoint(), IREE_NET_FRAME_SENDER_MAX_SPANS,
      header_pool_->get(), iree_net_control_channel_options_default(),
      callbacks, iree_allocator_system(), &channel_));
  IREE_ASSERT_OK(iree_net_control_channel_activate(channel_));
  mock_endpoint_->InjectError(iree_make_status(IREE_STATUS_INTERNAL, "error"));
  EXPECT_EQ(iree_net_control_channel_state(channel_),
            IREE_NET_CONTROL_CHANNEL_STATE_ERROR);
}

}  // namespace
}  // namespace net
}  // namespace iree
