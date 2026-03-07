// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/queue/queue_channel.h"

#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/channel/queue/frame.h"
#include "iree/net/channel/util/frame_sender.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Test buffer pool
//===----------------------------------------------------------------------===//

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
    // Transfer sole ownership to the pool (pool retained the region).
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

struct CapturedSend {
  std::vector<uint8_t> data;
  uint64_t user_data;
};

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
    MockCarrier::Destroy,
    MockCarrier::SetRecvHandler,
    MockCarrier::Activate,
    MockCarrier::Deactivate,
    MockCarrier::QuerySendBudget,
    MockCarrier::Send,
    nullptr,  // shutdown
    nullptr,  // direct_write
    nullptr,  // direct_read
    nullptr,  // register_buffer
    nullptr,  // unregister_buffer
};

//===----------------------------------------------------------------------===//
// Mock message endpoint
//===----------------------------------------------------------------------===//

struct MockEndpoint {
  iree_net_message_endpoint_callbacks_t callbacks = {};
  MockCarrier* carrier = nullptr;
  bool activated = false;

  static void SetCallbacks(void* self,
                           iree_net_message_endpoint_callbacks_t callbacks) {
    static_cast<MockEndpoint*>(self)->callbacks = callbacks;
  }
  static iree_status_t Activate(void* self) {
    static_cast<MockEndpoint*>(self)->activated = true;
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

  iree_net_message_endpoint_t as_endpoint() { return {this, &vtable}; }

  iree_status_t InjectMessage(const std::vector<uint8_t>& data) {
    iree_const_byte_span_t message =
        iree_make_const_byte_span(data.data(), data.size());
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(const_cast<uint8_t*>(data.data()),
                                          data.size());
    return callbacks.on_message(callbacks.user_data, message, &lease);
  }

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

struct ReceivedCommand {
  uint32_t stream_id;
  std::vector<iree_async_frontier_entry_t> wait_entries;
  std::vector<iree_async_frontier_entry_t> signal_entries;
  std::vector<uint8_t> command_data;
};

struct TestContext {
  std::vector<ReceivedCommand> commands;
  std::vector<iree_status_code_t> transport_errors;
  std::vector<uint64_t> send_completions;
  std::vector<iree_status_code_t> send_completion_errors;

  static iree_status_t OnCommand(void* user_data, uint32_t stream_id,
                                 const iree_async_frontier_t* wait_frontier,
                                 const iree_async_frontier_t* signal_frontier,
                                 iree_const_byte_span_t command_data,
                                 iree_async_buffer_lease_t* lease) {
    auto* context = static_cast<TestContext*>(user_data);
    ReceivedCommand cmd;
    cmd.stream_id = stream_id;
    if (wait_frontier) {
      for (uint8_t i = 0; i < wait_frontier->entry_count; ++i) {
        cmd.wait_entries.push_back(wait_frontier->entries[i]);
      }
    }
    if (signal_frontier) {
      for (uint8_t i = 0; i < signal_frontier->entry_count; ++i) {
        cmd.signal_entries.push_back(signal_frontier->entries[i]);
      }
    }
    cmd.command_data.assign(command_data.data,
                            command_data.data + command_data.data_length);
    context->commands.push_back(std::move(cmd));
    return iree_ok_status();
  }

  static void OnTransportError(void* user_data, iree_status_t status) {
    auto* context = static_cast<TestContext*>(user_data);
    context->transport_errors.push_back(iree_status_code(status));
    iree_status_ignore(status);
  }

  static void OnSendComplete(void* user_data, uint64_t operation_user_data,
                             iree_status_t status) {
    auto* context = static_cast<TestContext*>(user_data);
    context->send_completions.push_back(operation_user_data);
    context->send_completion_errors.push_back(iree_status_code(status));
    iree_status_ignore(status);
  }

  iree_net_queue_channel_callbacks_t MakeCallbacks() {
    iree_net_queue_channel_callbacks_t callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.on_command = OnCommand;
    callbacks.on_transport_error = OnTransportError;
    callbacks.on_send_complete = OnSendComplete;
    callbacks.user_data = this;
    return callbacks;
  }
};

//===----------------------------------------------------------------------===//
// Message construction helpers
//===----------------------------------------------------------------------===//

// Builds a queue frame message with optional frontiers and command data.
static std::vector<uint8_t> BuildQueueFrame(
    iree_net_queue_frame_type_t type, iree_net_queue_frame_flags_t flags,
    uint32_t stream_id, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> message(IREE_NET_QUEUE_FRAME_HEADER_SIZE +
                               payload.size());
  iree_net_queue_frame_header_t header;
  iree_net_queue_frame_header_initialize(
      type, flags, static_cast<uint32_t>(payload.size()), stream_id, &header);
  memcpy(message.data(), &header, sizeof(header));
  if (!payload.empty()) {
    memcpy(message.data() + IREE_NET_QUEUE_FRAME_HEADER_SIZE, payload.data(),
           payload.size());
  }
  return message;
}

// Serializes a frontier to bytes. Entries must be sorted by axis.
static std::vector<uint8_t> SerializeFrontier(
    const std::vector<iree_async_frontier_entry_t>& entries) {
  iree_host_size_t size = sizeof(iree_async_frontier_t) +
                          entries.size() * sizeof(iree_async_frontier_entry_t);
  std::vector<uint8_t> bytes(size, 0);
  auto* frontier = reinterpret_cast<iree_async_frontier_t*>(bytes.data());
  frontier->entry_count = static_cast<uint8_t>(entries.size());
  memset(frontier->reserved, 0, sizeof(frontier->reserved));
  for (size_t i = 0; i < entries.size(); ++i) {
    frontier->entries[i] = entries[i];
  }
  return bytes;
}

// Concatenates byte vectors.
static std::vector<uint8_t> Concat(
    std::initializer_list<std::vector<uint8_t>> parts) {
  std::vector<uint8_t> result;
  for (const auto& part : parts) {
    result.insert(result.end(), part.begin(), part.end());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class QueueChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    carrier_ = MockCarrier::Create();
    endpoint_.carrier = carrier_.get();
    pool_ = std::make_unique<TestBufferPool>(/*buffer_count=*/16,
                                             /*buffer_size=*/1024);
  }

  void TearDown() override {
    if (channel_) iree_net_queue_channel_release(channel_);
    channel_ = nullptr;
  }

  // Creates and activates a channel with default callbacks.
  void CreateAndActivate() {
    IREE_ASSERT_OK(iree_net_queue_channel_create(
        endpoint_.as_endpoint(), /*max_send_spans=*/8, pool_->get(),
        context_.MakeCallbacks(), iree_allocator_system(), &channel_));
    IREE_ASSERT_OK(iree_net_queue_channel_activate(channel_));
  }

  std::unique_ptr<MockCarrier> carrier_;
  MockEndpoint endpoint_;
  std::unique_ptr<TestBufferPool> pool_;
  TestContext context_;
  iree_net_queue_channel_t* channel_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(QueueChannelTest, CreateAndRelease) {
  iree_net_queue_channel_t* channel = nullptr;
  IREE_ASSERT_OK(iree_net_queue_channel_create(
      endpoint_.as_endpoint(), 8, pool_->get(), context_.MakeCallbacks(),
      iree_allocator_system(), &channel));
  EXPECT_EQ(iree_net_queue_channel_state(channel),
            IREE_NET_QUEUE_CHANNEL_STATE_CREATED);
  iree_net_queue_channel_release(channel);
}

TEST_F(QueueChannelTest, ActivateTransitionsToOperational) {
  CreateAndActivate();
  EXPECT_EQ(iree_net_queue_channel_state(channel_),
            IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL);
  EXPECT_TRUE(endpoint_.activated);
}

TEST_F(QueueChannelTest, DoubleActivateFails) {
  CreateAndActivate();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_queue_channel_activate(channel_));
}

TEST_F(QueueChannelTest, RetainRelease) {
  CreateAndActivate();
  iree_net_queue_channel_retain(channel_);
  iree_net_queue_channel_release(channel_);
  // Channel still alive — state query should work.
  EXPECT_EQ(iree_net_queue_channel_state(channel_),
            IREE_NET_QUEUE_CHANNEL_STATE_OPERATIONAL);
}

TEST_F(QueueChannelTest, NullCallbackRejects) {
  iree_net_queue_channel_t* channel = nullptr;
  iree_net_queue_channel_callbacks_t callbacks = {};
  callbacks.on_command = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_queue_channel_create(
                            endpoint_.as_endpoint(), 8, pool_->get(), callbacks,
                            iree_allocator_system(), &channel));
}

//===----------------------------------------------------------------------===//
// Receive path: COMMAND frames
//===----------------------------------------------------------------------===//

TEST_F(QueueChannelTest, ReceiveCommandNoFrontiers) {
  CreateAndActivate();

  std::vector<uint8_t> command_data = {0x01, 0x02, 0x03, 0x04};
  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                 IREE_NET_QUEUE_FRAME_FLAG_NONE,
                                 /*stream_id=*/42, command_data);
  IREE_ASSERT_OK(endpoint_.InjectMessage(message));

  ASSERT_EQ(context_.commands.size(), 1u);
  EXPECT_EQ(context_.commands[0].stream_id, 42u);
  EXPECT_TRUE(context_.commands[0].wait_entries.empty());
  EXPECT_TRUE(context_.commands[0].signal_entries.empty());
  EXPECT_EQ(context_.commands[0].command_data, command_data);
}

TEST_F(QueueChannelTest, ReceiveCommandWithWaitFrontier) {
  CreateAndActivate();

  iree_async_axis_t axis_a = iree_async_axis_make_queue(1, 0, 0, 0);
  iree_async_axis_t axis_b = iree_async_axis_make_queue(1, 0, 1, 0);
  auto wait_bytes = SerializeFrontier({{axis_a, 5}, {axis_b, 10}});
  std::vector<uint8_t> command_data = {0xAA, 0xBB};

  auto payload = Concat({wait_bytes, command_data});
  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                 IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER,
                                 /*stream_id=*/1, payload);
  IREE_ASSERT_OK(endpoint_.InjectMessage(message));

  ASSERT_EQ(context_.commands.size(), 1u);
  ASSERT_EQ(context_.commands[0].wait_entries.size(), 2u);
  EXPECT_EQ(context_.commands[0].wait_entries[0].axis, axis_a);
  EXPECT_EQ(context_.commands[0].wait_entries[0].epoch, 5u);
  EXPECT_EQ(context_.commands[0].wait_entries[1].axis, axis_b);
  EXPECT_EQ(context_.commands[0].wait_entries[1].epoch, 10u);
  EXPECT_TRUE(context_.commands[0].signal_entries.empty());
  EXPECT_EQ(context_.commands[0].command_data, command_data);
}

TEST_F(QueueChannelTest, ReceiveCommandWithSignalFrontier) {
  CreateAndActivate();

  iree_async_axis_t axis_c = iree_async_axis_make_queue(1, 0, 2, 0);
  auto signal_bytes = SerializeFrontier({{axis_c, 7}});
  std::vector<uint8_t> command_data = {0xCC};

  auto payload = Concat({signal_bytes, command_data});
  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                 IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER,
                                 /*stream_id=*/2, payload);
  IREE_ASSERT_OK(endpoint_.InjectMessage(message));

  ASSERT_EQ(context_.commands.size(), 1u);
  EXPECT_TRUE(context_.commands[0].wait_entries.empty());
  ASSERT_EQ(context_.commands[0].signal_entries.size(), 1u);
  EXPECT_EQ(context_.commands[0].signal_entries[0].axis, axis_c);
  EXPECT_EQ(context_.commands[0].signal_entries[0].epoch, 7u);
  EXPECT_EQ(context_.commands[0].command_data, command_data);
}

TEST_F(QueueChannelTest, ReceiveCommandWithBothFrontiers) {
  CreateAndActivate();

  iree_async_axis_t axis_a = iree_async_axis_make_queue(1, 0, 0, 0);
  iree_async_axis_t axis_b = iree_async_axis_make_queue(1, 0, 1, 0);
  iree_async_axis_t axis_c = iree_async_axis_make_queue(1, 1, 0, 0);

  auto wait_bytes = SerializeFrontier({{axis_a, 3}});
  auto signal_bytes = SerializeFrontier({{axis_b, 6}, {axis_c, 9}});
  std::vector<uint8_t> command_data = {0xDE, 0xAD, 0xBE, 0xEF};

  auto payload = Concat({wait_bytes, signal_bytes, command_data});
  auto message = BuildQueueFrame(
      IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
      (iree_net_queue_frame_flags_t)(IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER |
                                     IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER),
      /*stream_id=*/7, payload);
  IREE_ASSERT_OK(endpoint_.InjectMessage(message));

  ASSERT_EQ(context_.commands.size(), 1u);
  EXPECT_EQ(context_.commands[0].stream_id, 7u);
  ASSERT_EQ(context_.commands[0].wait_entries.size(), 1u);
  EXPECT_EQ(context_.commands[0].wait_entries[0].axis, axis_a);
  EXPECT_EQ(context_.commands[0].wait_entries[0].epoch, 3u);
  ASSERT_EQ(context_.commands[0].signal_entries.size(), 2u);
  EXPECT_EQ(context_.commands[0].signal_entries[0].axis, axis_b);
  EXPECT_EQ(context_.commands[0].signal_entries[0].epoch, 6u);
  EXPECT_EQ(context_.commands[0].signal_entries[1].axis, axis_c);
  EXPECT_EQ(context_.commands[0].signal_entries[1].epoch, 9u);
  EXPECT_EQ(context_.commands[0].command_data, command_data);
}

TEST_F(QueueChannelTest, ReceiveCommandEmptyPayload) {
  CreateAndActivate();

  auto message =
      BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                      IREE_NET_QUEUE_FRAME_FLAG_NONE, /*stream_id=*/0, {});
  IREE_ASSERT_OK(endpoint_.InjectMessage(message));

  ASSERT_EQ(context_.commands.size(), 1u);
  EXPECT_EQ(context_.commands[0].stream_id, 0u);
  EXPECT_TRUE(context_.commands[0].command_data.empty());
}

TEST_F(QueueChannelTest, ReceiveCommandFrontierOnly) {
  // A frontier with no trailing command data is valid — the frontier IS the
  // command (e.g., a pure barrier with no payload).
  CreateAndActivate();

  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0);
  auto wait_bytes = SerializeFrontier({{axis, 100}});

  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                 IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER,
                                 /*stream_id=*/0, wait_bytes);
  IREE_ASSERT_OK(endpoint_.InjectMessage(message));

  ASSERT_EQ(context_.commands.size(), 1u);
  ASSERT_EQ(context_.commands[0].wait_entries.size(), 1u);
  EXPECT_EQ(context_.commands[0].wait_entries[0].epoch, 100u);
  EXPECT_TRUE(context_.commands[0].command_data.empty());
}

//===----------------------------------------------------------------------===//
// Receive path: error handling
//===----------------------------------------------------------------------===//

TEST_F(QueueChannelTest, ReceiveMessageTooShort) {
  CreateAndActivate();
  std::vector<uint8_t> truncated(10, 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(truncated));
}

TEST_F(QueueChannelTest, ReceiveBadMagic) {
  CreateAndActivate();
  std::vector<uint8_t> message(16, 0);
  // Bad magic in first 4 bytes.
  message[0] = 0xFF;
  message[1] = 0xFF;
  message[2] = 0xFF;
  message[3] = 0xFF;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(QueueChannelTest, ReceiveBadVersion) {
  CreateAndActivate();
  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                 IREE_NET_QUEUE_FRAME_FLAG_NONE, 0, {});
  message[4] = 99;  // Corrupt version field.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(QueueChannelTest, ReceivePayloadLengthMismatch) {
  CreateAndActivate();
  // Build a valid frame header claiming 100 bytes of payload, but only
  // provide 4 bytes.
  std::vector<uint8_t> message(IREE_NET_QUEUE_FRAME_HEADER_SIZE + 4, 0);
  iree_net_queue_frame_header_t header;
  iree_net_queue_frame_header_initialize(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                         IREE_NET_QUEUE_FRAME_FLAG_NONE,
                                         /*payload_length=*/100, 0, &header);
  memcpy(message.data(), &header, sizeof(header));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(QueueChannelTest, ReceiveTruncatedWaitFrontierHeader) {
  CreateAndActivate();
  // Set HAS_WAIT_FRONTIER but only provide 4 bytes of payload (need 8 for
  // the frontier header).
  std::vector<uint8_t> payload(4, 0);
  auto message =
      BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                      IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER, 0, payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(QueueChannelTest, ReceiveTruncatedWaitFrontierEntries) {
  CreateAndActivate();
  // Frontier header says 2 entries but only 1 entry's worth of data.
  std::vector<uint8_t> payload(
      sizeof(iree_async_frontier_t) + sizeof(iree_async_frontier_entry_t), 0);
  auto* frontier = reinterpret_cast<iree_async_frontier_t*>(payload.data());
  frontier->entry_count = 2;  // Claims 2 entries.
  auto message =
      BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                      IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER, 0, payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(QueueChannelTest, ReceiveUnknownFrameType) {
  CreateAndActivate();
  auto message = BuildQueueFrame(static_cast<iree_net_queue_frame_type_t>(0xFF),
                                 IREE_NET_QUEUE_FRAME_FLAG_NONE, 0, {});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        endpoint_.InjectMessage(message));
}

TEST_F(QueueChannelTest, ReceiveDataFrameUnimplemented) {
  CreateAndActivate();
  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_DATA,
                                 IREE_NET_QUEUE_FRAME_FLAG_NONE, 1, {0x00});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        endpoint_.InjectMessage(message));
}

//===----------------------------------------------------------------------===//
// Receive path: error state
//===----------------------------------------------------------------------===//

TEST_F(QueueChannelTest, TransportErrorTransitionsToError) {
  CreateAndActivate();
  endpoint_.InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport down"));
  EXPECT_EQ(iree_net_queue_channel_state(channel_),
            IREE_NET_QUEUE_CHANNEL_STATE_ERROR);
  ASSERT_EQ(context_.transport_errors.size(), 1u);
  EXPECT_EQ(context_.transport_errors[0], IREE_STATUS_UNAVAILABLE);
}

TEST_F(QueueChannelTest, ErrorStateRejectsMessages) {
  CreateAndActivate();
  endpoint_.InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport down"));

  auto message = BuildQueueFrame(IREE_NET_QUEUE_FRAME_TYPE_COMMAND,
                                 IREE_NET_QUEUE_FRAME_FLAG_NONE, 0, {0x01});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        endpoint_.InjectMessage(message));
}

//===----------------------------------------------------------------------===//
// Send path
//===----------------------------------------------------------------------===//

TEST_F(QueueChannelTest, SendCommandNoFrontiers) {
  CreateAndActivate();

  uint8_t command_data[] = {0x01, 0x02, 0x03, 0x04};
  iree_async_span_t span =
      iree_async_span_from_ptr(command_data, sizeof(command_data));
  iree_async_span_list_t payload = {&span, 1};

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      channel_, /*stream_id=*/10, /*wait_frontier=*/NULL,
      /*signal_frontier=*/NULL, payload, /*operation_user_data=*/42));

  ASSERT_EQ(carrier_->sends.size(), 1u);
  ASSERT_EQ(context_.send_completions.size(), 1u);
  EXPECT_EQ(context_.send_completions[0], 42u);

  // Verify the sent data: 16-byte header + command data.
  const auto& sent = carrier_->sends[0].data;
  ASSERT_GE(sent.size(),
            IREE_NET_QUEUE_FRAME_HEADER_SIZE + sizeof(command_data));

  iree_net_queue_frame_header_t sent_header;
  memcpy(&sent_header, sent.data(), sizeof(sent_header));
  EXPECT_EQ(sent_header.magic, IREE_NET_QUEUE_FRAME_MAGIC);
  EXPECT_EQ(sent_header.type, IREE_NET_QUEUE_FRAME_TYPE_COMMAND);
  EXPECT_EQ(sent_header.flags, IREE_NET_QUEUE_FRAME_FLAG_NONE);
  EXPECT_EQ(sent_header.stream_id, 10u);
  EXPECT_EQ(sent_header.payload_length, sizeof(command_data));

  // Command data follows header.
  EXPECT_EQ(memcmp(sent.data() + IREE_NET_QUEUE_FRAME_HEADER_SIZE, command_data,
                   sizeof(command_data)),
            0);
}

TEST_F(QueueChannelTest, SendCommandWithFrontiers) {
  CreateAndActivate();

  // Build wait frontier (1 entry).
  iree_async_axis_t axis_a = iree_async_axis_make_queue(1, 0, 0, 0);
  uint8_t wait_storage[sizeof(iree_async_frontier_t) +
                       sizeof(iree_async_frontier_entry_t)];
  auto* wait_frontier = reinterpret_cast<iree_async_frontier_t*>(wait_storage);
  iree_async_frontier_initialize(wait_frontier, 1);
  wait_frontier->entries[0] = {axis_a, 5};

  // Build signal frontier (1 entry).
  iree_async_axis_t axis_b = iree_async_axis_make_queue(1, 0, 1, 0);
  uint8_t signal_storage[sizeof(iree_async_frontier_t) +
                         sizeof(iree_async_frontier_entry_t)];
  auto* signal_frontier =
      reinterpret_cast<iree_async_frontier_t*>(signal_storage);
  iree_async_frontier_initialize(signal_frontier, 1);
  signal_frontier->entries[0] = {axis_b, 10};

  uint8_t command_data[] = {0xAA, 0xBB};
  iree_async_span_t span =
      iree_async_span_from_ptr(command_data, sizeof(command_data));
  iree_async_span_list_t payload = {&span, 1};

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      channel_, /*stream_id=*/3, wait_frontier, signal_frontier, payload,
      /*operation_user_data=*/99));

  ASSERT_EQ(carrier_->sends.size(), 1u);
  const auto& sent = carrier_->sends[0].data;

  // Parse the sent header.
  iree_net_queue_frame_header_t sent_header;
  memcpy(&sent_header, sent.data(), sizeof(sent_header));
  EXPECT_EQ(sent_header.type, IREE_NET_QUEUE_FRAME_TYPE_COMMAND);
  EXPECT_EQ(sent_header.flags,
            IREE_NET_QUEUE_FRAME_FLAG_HAS_WAIT_FRONTIER |
                IREE_NET_QUEUE_FRAME_FLAG_HAS_SIGNAL_FRONTIER);
  EXPECT_EQ(sent_header.stream_id, 3u);

  // Expected total payload: wait_frontier + signal_frontier + command_data.
  iree_host_size_t expected_frontier_size =
      sizeof(iree_async_frontier_t) + sizeof(iree_async_frontier_entry_t);
  EXPECT_EQ(sent_header.payload_length,
            2 * expected_frontier_size + sizeof(command_data));

  // Verify wait frontier in sent data.
  iree_host_size_t offset = IREE_NET_QUEUE_FRAME_HEADER_SIZE;
  const auto* sent_wait =
      reinterpret_cast<const iree_async_frontier_t*>(sent.data() + offset);
  EXPECT_EQ(sent_wait->entry_count, 1u);
  EXPECT_EQ(sent_wait->entries[0].axis, axis_a);
  EXPECT_EQ(sent_wait->entries[0].epoch, 5u);
  offset += expected_frontier_size;

  // Verify signal frontier.
  const auto* sent_signal =
      reinterpret_cast<const iree_async_frontier_t*>(sent.data() + offset);
  EXPECT_EQ(sent_signal->entry_count, 1u);
  EXPECT_EQ(sent_signal->entries[0].axis, axis_b);
  EXPECT_EQ(sent_signal->entries[0].epoch, 10u);
  offset += expected_frontier_size;

  // Verify command data.
  EXPECT_EQ(memcmp(sent.data() + offset, command_data, sizeof(command_data)),
            0);
}

TEST_F(QueueChannelTest, SendBeforeActivateFails) {
  iree_net_queue_channel_t* channel = nullptr;
  IREE_ASSERT_OK(iree_net_queue_channel_create(
      endpoint_.as_endpoint(), 8, pool_->get(), context_.MakeCallbacks(),
      iree_allocator_system(), &channel));

  iree_async_span_list_t empty_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_queue_channel_send_command(
                            channel, 0, NULL, NULL, empty_payload, 0));

  iree_net_queue_channel_release(channel);
}

TEST_F(QueueChannelTest, SendAfterErrorFails) {
  CreateAndActivate();
  endpoint_.InjectError(
      iree_make_status(IREE_STATUS_UNAVAILABLE, "transport down"));

  iree_async_span_list_t empty_payload = {nullptr, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_queue_channel_send_command(
                            channel_, 0, NULL, NULL, empty_payload, 0));
}

//===----------------------------------------------------------------------===//
// Round-trip: send then receive
//===----------------------------------------------------------------------===//

TEST_F(QueueChannelTest, RoundTripCommandWithFrontiers) {
  // Verify that a frame built by send_command can be parsed by the receive
  // path. We send a command, capture the wire bytes, then inject them back
  // through a second channel's receive path.
  CreateAndActivate();

  iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 0, 0);
  uint8_t wait_storage[sizeof(iree_async_frontier_t) +
                       sizeof(iree_async_frontier_entry_t)];
  auto* wait_frontier = reinterpret_cast<iree_async_frontier_t*>(wait_storage);
  iree_async_frontier_initialize(wait_frontier, 1);
  wait_frontier->entries[0] = {axis, 42};

  uint8_t command_data[] = {0xDE, 0xAD};
  iree_async_span_t span =
      iree_async_span_from_ptr(command_data, sizeof(command_data));
  iree_async_span_list_t payload = {&span, 1};

  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      channel_, /*stream_id=*/5, wait_frontier, /*signal_frontier=*/NULL,
      payload, /*operation_user_data=*/0));

  // Captured wire bytes from the send.
  ASSERT_EQ(carrier_->sends.size(), 1u);
  const auto& wire_bytes = carrier_->sends[0].data;

  // Create a second channel to receive the same bytes.
  TestContext recv_context;
  auto recv_carrier = MockCarrier::Create();
  MockEndpoint recv_endpoint;
  recv_endpoint.carrier = recv_carrier.get();

  iree_net_queue_channel_t* recv_channel = nullptr;
  IREE_ASSERT_OK(iree_net_queue_channel_create(
      recv_endpoint.as_endpoint(), 8, pool_->get(),
      recv_context.MakeCallbacks(), iree_allocator_system(), &recv_channel));
  IREE_ASSERT_OK(iree_net_queue_channel_activate(recv_channel));

  IREE_ASSERT_OK(recv_endpoint.InjectMessage(wire_bytes));

  ASSERT_EQ(recv_context.commands.size(), 1u);
  EXPECT_EQ(recv_context.commands[0].stream_id, 5u);
  ASSERT_EQ(recv_context.commands[0].wait_entries.size(), 1u);
  EXPECT_EQ(recv_context.commands[0].wait_entries[0].axis, axis);
  EXPECT_EQ(recv_context.commands[0].wait_entries[0].epoch, 42u);
  EXPECT_TRUE(recv_context.commands[0].signal_entries.empty());
  std::vector<uint8_t> expected_data = {0xDE, 0xAD};
  EXPECT_EQ(recv_context.commands[0].command_data, expected_data);

  iree_net_queue_channel_release(recv_channel);
}

}  // namespace
}  // namespace net
}  // namespace iree
