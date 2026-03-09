// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/framing_adapter.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/message_endpoint.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Test protocol: 4-byte little-endian length prefix
//===----------------------------------------------------------------------===//

static constexpr iree_host_size_t kHeaderSize = 4;

static iree_host_size_t TestFrameLength(void* user_data,
                                        iree_const_byte_span_t available) {
  if (available.data_length < kHeaderSize) return 0;
  uint32_t frame_size = available.data[0] | (available.data[1] << 8) |
                        (available.data[2] << 16) | (available.data[3] << 24);
  return static_cast<iree_host_size_t>(frame_size);
}

static std::vector<uint8_t> MakeFrame(const std::string& payload) {
  uint32_t frame_size = static_cast<uint32_t>(kHeaderSize + payload.size());
  std::vector<uint8_t> frame(frame_size);
  frame[0] = frame_size & 0xFF;
  frame[1] = (frame_size >> 8) & 0xFF;
  frame[2] = (frame_size >> 16) & 0xFF;
  frame[3] = (frame_size >> 24) & 0xFF;
  memcpy(frame.data() + kHeaderSize, payload.data(), payload.size());
  return frame;
}

//===----------------------------------------------------------------------===//
// Mock carrier for testing
//===----------------------------------------------------------------------===//

struct CapturedSend {
  std::vector<std::vector<uint8_t>> span_data;
  uint64_t user_data;
  iree_net_send_flags_t flags;
};

struct MockCarrier {
  iree_net_carrier_t base;
  std::vector<CapturedSend> sends;
  iree_status_code_t next_send_error = IREE_STATUS_OK;
  iree_host_size_t send_budget_bytes = SIZE_MAX;
  uint32_t send_budget_slots = UINT32_MAX;

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
    MockCarrier* mock = reinterpret_cast<MockCarrier*>(carrier);
    return {mock->send_budget_bytes, mock->send_budget_slots};
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
    captured.flags = params->flags;
    for (iree_host_size_t i = 0; i < params->data.count; ++i) {
      iree_async_span_t span = params->data.values[i];
      uint8_t* ptr = iree_async_span_ptr(span);
      captured.span_data.push_back(
          std::vector<uint8_t>(ptr, ptr + span.length));
    }
    mock->sends.push_back(std::move(captured));
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
    iree_net_carrier_callback_t callback = {nullptr, nullptr};
    iree_net_carrier_initialize(&kVtable, IREE_NET_CARRIER_CAPABILITY_RELIABLE,
                                0, 8, callback, iree_allocator_system(),
                                &mock->base);
    return mock;
  }

  // Injects recv data into the adapter's recv handler (simulates carrier recv).
  // The lease's span data is used by the accumulator; |data| provides length.
  iree_status_t InjectRecv(const std::vector<uint8_t>& data,
                           iree_async_buffer_lease_t* lease) {
    iree_async_span_t span = iree_async_span_from_ptr(
        const_cast<uint8_t*>(data.data()), data.size());
    return base.recv_handler.fn(base.recv_handler.user_data, span, lease);
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
// Mock lease for testing
//===----------------------------------------------------------------------===//

struct MockLease {
  std::vector<uint8_t> data;
  iree_async_buffer_lease_t lease;
  int release_count = 0;

  static void ReleaseCallback(void* user_data,
                              iree_async_buffer_index_t buffer_index) {
    MockLease* self = static_cast<MockLease*>(user_data);
    self->release_count++;
  }

  explicit MockLease(const std::vector<uint8_t>& buffer_data)
      : data(buffer_data), release_count(0) {
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(data.data(), data.size());
    lease.release.fn = ReleaseCallback;
    lease.release.user_data = this;
    lease.buffer_index = 0;
  }

  MockLease(std::initializer_list<std::vector<uint8_t>> frame_list)
      : release_count(0) {
    for (const auto& frame : frame_list) {
      data.insert(data.end(), frame.begin(), frame.end());
    }
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(data.data(), data.size());
    lease.release.fn = ReleaseCallback;
    lease.release.user_data = this;
    lease.buffer_index = 0;
  }
};

//===----------------------------------------------------------------------===//
// Test buffer pool (real pool backed by test memory)
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
    iree_async_region_release(region_);
  }

  ~TestBufferPool() {
    if (pool_) iree_async_buffer_pool_free(pool_);
  }

  iree_async_buffer_pool_t* get() { return pool_; }

  iree_host_size_t AvailableCount() const {
    return iree_async_buffer_pool_available(pool_);
  }

 private:
  static void DestroyRegion(iree_async_region_t* region) { free(region); }

  std::vector<uint8_t> buffer_memory_;
  iree_host_size_t buffer_count_;
  iree_host_size_t buffer_size_;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* pool_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Test context for tracking received messages
//===----------------------------------------------------------------------===//

struct ReceivedMessage {
  std::vector<uint8_t> data;
  bool had_lease;
};

struct TestContext {
  std::vector<ReceivedMessage> messages;
  iree_status_code_t next_error = IREE_STATUS_OK;
  bool deactivated = false;

  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    if (ctx->next_error != IREE_STATUS_OK) {
      iree_status_code_t error = ctx->next_error;
      ctx->next_error = IREE_STATUS_OK;
      return iree_make_status(error, "injected error");
    }
    ReceivedMessage received;
    received.data.assign(message.data, message.data + message.data_length);
    received.had_lease = (lease != nullptr);
    ctx->messages.push_back(std::move(received));
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    iree_status_ignore(status);
  }

  static void OnDeactivated(void* user_data) {
    TestContext* ctx = static_cast<TestContext*>(user_data);
    ctx->deactivated = true;
  }

  iree_net_message_endpoint_callbacks_t MakeCallbacks() {
    return {OnMessage, OnError, this};
  }
};

//===----------------------------------------------------------------------===//
// Test constants
//===----------------------------------------------------------------------===//

static constexpr iree_host_size_t kMaxFrameSize = 1024;
static constexpr iree_host_size_t kPoolBufferCount = 4;
static constexpr iree_host_size_t kPoolBufferSize = kMaxFrameSize;

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class FramingAdapterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_carrier_ = MockCarrier::Create();
    test_pool_ =
        std::make_unique<TestBufferPool>(kPoolBufferCount, kPoolBufferSize);
    iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
    IREE_ASSERT_OK(iree_net_framing_adapter_allocate(
        &mock_carrier_->base, frame_length, kMaxFrameSize, test_pool_->get(),
        iree_allocator_system(), &adapter_));
    endpoint_ = iree_net_framing_adapter_as_endpoint(adapter_);
  }

  void TearDown() override {
    if (adapter_) {
      iree_net_framing_adapter_free(adapter_);
      adapter_ = nullptr;
    }
  }

  void ActivateWithCallbacks() {
    iree_net_message_endpoint_set_callbacks(endpoint_, ctx_.MakeCallbacks());
    IREE_ASSERT_OK(iree_net_message_endpoint_activate(endpoint_));
  }

  // Injects recv data with a mock lease (simulates a carrier delivering data).
  iree_status_t InjectRecv(const std::vector<uint8_t>& data) {
    MockLease lease(data);
    return mock_carrier_->InjectRecv(data, &lease.lease);
  }

  iree_net_framing_adapter_t* adapter_ = nullptr;
  iree_net_message_endpoint_t endpoint_;
  std::unique_ptr<MockCarrier> mock_carrier_;
  std::unique_ptr<TestBufferPool> test_pool_;
  TestContext ctx_;
};

//===----------------------------------------------------------------------===//
// Allocation validation tests
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, AllocateAndFree) { EXPECT_NE(adapter_, nullptr); }

TEST_F(FramingAdapterTest, AllocateRequiresCarrier) {
  iree_net_framing_adapter_t* adapter = nullptr;
  iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_framing_adapter_allocate(nullptr, frame_length, kMaxFrameSize,
                                        test_pool_->get(),
                                        iree_allocator_system(), &adapter));
}

TEST_F(FramingAdapterTest, AllocateRequiresFrameLengthFn) {
  auto carrier = MockCarrier::Create();
  iree_net_framing_adapter_t* adapter = nullptr;
  iree_net_frame_length_callback_t frame_length = {nullptr, nullptr};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_framing_adapter_allocate(&carrier->base, frame_length,
                                        kMaxFrameSize, test_pool_->get(),
                                        iree_allocator_system(), &adapter));
}

TEST_F(FramingAdapterTest, AllocateRequiresPool) {
  auto carrier = MockCarrier::Create();
  iree_net_framing_adapter_t* adapter = nullptr;
  iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_framing_adapter_allocate(
                            &carrier->base, frame_length, kMaxFrameSize,
                            nullptr, iree_allocator_system(), &adapter));
}

TEST_F(FramingAdapterTest, AllocateRequiresNonZeroMaxFrameSize) {
  auto carrier = MockCarrier::Create();
  iree_net_framing_adapter_t* adapter = nullptr;
  iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_framing_adapter_allocate(
                            &carrier->base, frame_length, 0, test_pool_->get(),
                            iree_allocator_system(), &adapter));
}

TEST_F(FramingAdapterTest, AllocateRejectsActivatedCarrier) {
  auto carrier = MockCarrier::Create();
  iree_net_carrier_set_state(&carrier->base, IREE_NET_CARRIER_STATE_ACTIVE);
  iree_net_framing_adapter_t* adapter = nullptr;
  iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_framing_adapter_allocate(&carrier->base, frame_length,
                                        kMaxFrameSize, test_pool_->get(),
                                        iree_allocator_system(), &adapter));
}

TEST_F(FramingAdapterTest, AllocateRejectsPoolTooSmall) {
  auto carrier = MockCarrier::Create();
  iree_net_framing_adapter_t* adapter = nullptr;
  iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
  // Pool buffer size is kPoolBufferSize=1024, try max_frame_size=2048.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_framing_adapter_allocate(&carrier->base, frame_length, 2048,
                                        test_pool_->get(),
                                        iree_allocator_system(), &adapter));
}

//===----------------------------------------------------------------------===//
// Activation tests
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, ActivateRequiresCallbacks) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_message_endpoint_activate(endpoint_));
}

TEST_F(FramingAdapterTest, DoubleActivateFails) {
  ActivateWithCallbacks();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_message_endpoint_activate(endpoint_));
}

//===----------------------------------------------------------------------===//
// Receive path: zero-copy (frame fits in single buffer)
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, SingleCompleteFrame) {
  ActivateWithCallbacks();

  auto frame = MakeFrame("Hello");
  IREE_ASSERT_OK(InjectRecv(frame));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
  EXPECT_TRUE(ctx_.messages[0].had_lease);
}

TEST_F(FramingAdapterTest, MultipleFramesInOneBuffer) {
  ActivateWithCallbacks();

  auto frame1 = MakeFrame("First");
  auto frame2 = MakeFrame("Second");
  auto frame3 = MakeFrame("Third");

  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin(), frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.end());
  buffer.insert(buffer.end(), frame3.begin(), frame3.end());

  MockLease lease(buffer);
  IREE_ASSERT_OK(mock_carrier_->InjectRecv(buffer, &lease.lease));

  ASSERT_EQ(ctx_.messages.size(), 3u);
  EXPECT_EQ(ctx_.messages[0].data, frame1);
  EXPECT_EQ(ctx_.messages[1].data, frame2);
  EXPECT_EQ(ctx_.messages[2].data, frame3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(ctx_.messages[i].had_lease);
  }
}

TEST_F(FramingAdapterTest, EmptyPayloadFrame) {
  ActivateWithCallbacks();

  auto frame = MakeFrame("");
  ASSERT_EQ(frame.size(), kHeaderSize);

  IREE_ASSERT_OK(InjectRecv(frame));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
  EXPECT_TRUE(ctx_.messages[0].had_lease);
}

TEST_F(FramingAdapterTest, ZeroCopyPathDoesNotTouchPool) {
  ActivateWithCallbacks();

  iree_host_size_t available_before = test_pool_->AvailableCount();

  auto frame = MakeFrame("Zero-copy frame");
  IREE_ASSERT_OK(InjectRecv(frame));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
}

//===----------------------------------------------------------------------===//
// Receive path: copy path (frame spans multiple buffers)
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, FrameSpansTwoBuffers) {
  ActivateWithCallbacks();

  auto frame = MakeFrame("Split across buffers");
  iree_host_size_t split_point = frame.size() / 2;

  std::vector<uint8_t> first_half(frame.begin(), frame.begin() + split_point);
  IREE_ASSERT_OK(InjectRecv(first_half));
  EXPECT_EQ(ctx_.messages.size(), 0u);

  std::vector<uint8_t> second_half(frame.begin() + split_point, frame.end());
  IREE_ASSERT_OK(InjectRecv(second_half));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
  EXPECT_TRUE(ctx_.messages[0].had_lease);
}

TEST_F(FramingAdapterTest, PartialHeaderThenRest) {
  ActivateWithCallbacks();

  auto frame = MakeFrame("test");

  // Just 2 bytes of the 4-byte header.
  std::vector<uint8_t> partial_header(frame.begin(), frame.begin() + 2);
  IREE_ASSERT_OK(InjectRecv(partial_header));
  EXPECT_EQ(ctx_.messages.size(), 0u);

  std::vector<uint8_t> rest(frame.begin() + 2, frame.end());
  IREE_ASSERT_OK(InjectRecv(rest));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
}

TEST_F(FramingAdapterTest, ByteByByte) {
  ActivateWithCallbacks();

  auto frame = MakeFrame("byte by byte");
  for (size_t i = 0; i < frame.size(); ++i) {
    std::vector<uint8_t> single_byte = {frame[i]};
    IREE_ASSERT_OK(InjectRecv(single_byte));
  }

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
  EXPECT_TRUE(ctx_.messages[0].had_lease);
}

TEST_F(FramingAdapterTest, CopyPathAlwaysDeliversNonNullLease) {
  ActivateWithCallbacks();

  // Split at header boundary to force copy path.
  auto frame = MakeFrame("Must have lease");

  std::vector<uint8_t> header(frame.begin(), frame.begin() + kHeaderSize);
  IREE_ASSERT_OK(InjectRecv(header));

  std::vector<uint8_t> payload(frame.begin() + kHeaderSize, frame.end());
  IREE_ASSERT_OK(InjectRecv(payload));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_TRUE(ctx_.messages[0].had_lease);
}

TEST_F(FramingAdapterTest, CopyPathReturnsPoolBuffer) {
  ActivateWithCallbacks();

  iree_host_size_t available_before = test_pool_->AvailableCount();

  // Force copy path by splitting frame across two recvs.
  auto frame = MakeFrame("Pool buffer lifecycle");
  iree_host_size_t split_point = frame.size() / 2;

  std::vector<uint8_t> first_half(frame.begin(), frame.begin() + split_point);
  IREE_ASSERT_OK(InjectRecv(first_half));

  std::vector<uint8_t> second_half(frame.begin() + split_point, frame.end());
  IREE_ASSERT_OK(InjectRecv(second_half));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  // Adapter acquires a pool buffer, delivers, then releases it.
  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
}

TEST_F(FramingAdapterTest, MultipleCopyPathFramesRecyclePoolBuffers) {
  ActivateWithCallbacks();

  iree_host_size_t available_before = test_pool_->AvailableCount();

  // Send several frames, each split across two recvs.
  for (int i = 0; i < 8; ++i) {
    auto frame = MakeFrame("Frame " + std::to_string(i));
    iree_host_size_t split_point = frame.size() / 2;

    std::vector<uint8_t> first_half(frame.begin(), frame.begin() + split_point);
    IREE_ASSERT_OK(InjectRecv(first_half));

    std::vector<uint8_t> second_half(frame.begin() + split_point, frame.end());
    IREE_ASSERT_OK(InjectRecv(second_half));
  }

  // All 8 frames delivered, pool buffers recycled each time.
  ASSERT_EQ(ctx_.messages.size(), 8u);
  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
}

//===----------------------------------------------------------------------===//
// Receive path: mixed zero-copy and copy path
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, CompleteFrameThenPartial) {
  ActivateWithCallbacks();

  auto frame1 = MakeFrame("Complete");
  auto frame2 = MakeFrame("Partial frame here");
  iree_host_size_t partial_bytes = frame2.size() / 2;

  // Buffer contains complete frame1 + partial frame2.
  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin(), frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.begin() + partial_bytes);

  MockLease lease(buffer);
  IREE_ASSERT_OK(mock_carrier_->InjectRecv(buffer, &lease.lease));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame1);

  // Finish frame2.
  std::vector<uint8_t> rest(frame2.begin() + partial_bytes, frame2.end());
  IREE_ASSERT_OK(InjectRecv(rest));

  ASSERT_EQ(ctx_.messages.size(), 2u);
  EXPECT_EQ(ctx_.messages[1].data, frame2);
}

TEST_F(FramingAdapterTest, BufferedCompletionThenZeroCopy) {
  ActivateWithCallbacks();

  auto frame1 = MakeFrame("First");
  auto frame2 = MakeFrame("Second");

  // Partial frame1.
  iree_host_size_t split_point = frame1.size() / 2;
  std::vector<uint8_t> partial(frame1.begin(), frame1.begin() + split_point);
  IREE_ASSERT_OK(InjectRecv(partial));
  EXPECT_EQ(ctx_.messages.size(), 0u);

  // Rest of frame1 + complete frame2 in same buffer.
  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin() + split_point, frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.end());

  MockLease lease(buffer);
  IREE_ASSERT_OK(mock_carrier_->InjectRecv(buffer, &lease.lease));

  ASSERT_EQ(ctx_.messages.size(), 2u);
  EXPECT_EQ(ctx_.messages[0].data, frame1);
  EXPECT_EQ(ctx_.messages[1].data, frame2);
  EXPECT_TRUE(ctx_.messages[0].had_lease);
  EXPECT_TRUE(ctx_.messages[1].had_lease);
}

//===----------------------------------------------------------------------===//
// Send path
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, SendForwardsToCarrier) {
  ActivateWithCallbacks();

  std::vector<uint8_t> data = {0x01, 0x02, 0x03, 0x04};
  iree_async_span_t span = iree_async_span_from_ptr(data.data(), data.size());
  iree_net_message_endpoint_send_params_t params;
  params.data = iree_async_span_list_make(&span, 1);
  params.user_data = 42;

  IREE_ASSERT_OK(iree_net_message_endpoint_send(endpoint_, &params));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  ASSERT_EQ(mock_carrier_->sends[0].span_data.size(), 1u);
  EXPECT_EQ(mock_carrier_->sends[0].span_data[0], data);
  EXPECT_EQ(mock_carrier_->sends[0].user_data, 42u);
  EXPECT_EQ(mock_carrier_->sends[0].flags, IREE_NET_SEND_FLAG_NONE);
}

TEST_F(FramingAdapterTest, SendMultipleSpans) {
  ActivateWithCallbacks();

  std::vector<uint8_t> data1 = {0xAA, 0xBB};
  std::vector<uint8_t> data2 = {0xCC, 0xDD, 0xEE};
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(data1.data(), data1.size()),
      iree_async_span_from_ptr(data2.data(), data2.size()),
  };
  iree_net_message_endpoint_send_params_t params;
  params.data = iree_async_span_list_make(spans, 2);
  params.user_data = 99;

  IREE_ASSERT_OK(iree_net_message_endpoint_send(endpoint_, &params));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  ASSERT_EQ(mock_carrier_->sends[0].span_data.size(), 2u);
  EXPECT_EQ(mock_carrier_->sends[0].span_data[0], data1);
  EXPECT_EQ(mock_carrier_->sends[0].span_data[1], data2);
}

TEST_F(FramingAdapterTest, SendCarrierError) {
  ActivateWithCallbacks();

  mock_carrier_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;

  std::vector<uint8_t> data = {0x01};
  iree_async_span_t span = iree_async_span_from_ptr(data.data(), data.size());
  iree_net_message_endpoint_send_params_t params;
  params.data = iree_async_span_list_make(&span, 1);
  params.user_data = 0;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_message_endpoint_send(endpoint_, &params));
}

TEST_F(FramingAdapterTest, QuerySendBudgetDelegatesToCarrier) {
  mock_carrier_->send_budget_bytes = 4096;
  mock_carrier_->send_budget_slots = 16;

  iree_net_carrier_send_budget_t budget =
      iree_net_message_endpoint_query_send_budget(endpoint_);

  EXPECT_EQ(budget.bytes, 4096u);
  EXPECT_EQ(budget.slots, 16u);
}

//===----------------------------------------------------------------------===//
// Deactivation
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, DeactivationCallbackFires) {
  ActivateWithCallbacks();

  IREE_ASSERT_OK(iree_net_message_endpoint_deactivate(
      endpoint_, TestContext::OnDeactivated, &ctx_));

  EXPECT_TRUE(ctx_.deactivated);
}

TEST_F(FramingAdapterTest, DeactivationWithNullCallback) {
  ActivateWithCallbacks();

  IREE_ASSERT_OK(
      iree_net_message_endpoint_deactivate(endpoint_, nullptr, nullptr));

  // Should not crash.
}

//===----------------------------------------------------------------------===//
// Error propagation
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, HandlerErrorPropagates) {
  ActivateWithCallbacks();

  ctx_.next_error = IREE_STATUS_INTERNAL;

  auto frame = MakeFrame("error");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, InjectRecv(frame));

  EXPECT_EQ(ctx_.messages.size(), 0u);
}

TEST_F(FramingAdapterTest, HandlerErrorStopsProcessingMultipleFrames) {
  ActivateWithCallbacks();

  auto frame1 = MakeFrame("First");
  auto frame2 = MakeFrame("Second");

  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin(), frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.end());

  ctx_.next_error = IREE_STATUS_CANCELLED;

  MockLease lease(buffer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        mock_carrier_->InjectRecv(buffer, &lease.lease));

  // Error on first frame stops processing; second frame never delivered.
  EXPECT_EQ(ctx_.messages.size(), 0u);
}

//===----------------------------------------------------------------------===//
// Pool exhaustion during copy path
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, PoolExhaustedDuringCopyPath) {
  ActivateWithCallbacks();

  // Exhaust the reassembly pool.
  std::vector<iree_async_buffer_lease_t> leases(kPoolBufferCount);
  for (iree_host_size_t i = 0; i < kPoolBufferCount; ++i) {
    IREE_ASSERT_OK(
        iree_async_buffer_pool_acquire(test_pool_->get(), &leases[i]));
  }
  EXPECT_EQ(test_pool_->AvailableCount(), 0u);

  // Force copy path by splitting a frame.
  auto frame = MakeFrame("Copy path with exhausted pool");
  iree_host_size_t split_point = frame.size() / 2;

  std::vector<uint8_t> first_half(frame.begin(), frame.begin() + split_point);
  IREE_ASSERT_OK(InjectRecv(first_half));

  // Second recv completes the frame but pool acquire fails.
  std::vector<uint8_t> second_half(frame.begin() + split_point, frame.end());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        InjectRecv(second_half));

  EXPECT_EQ(ctx_.messages.size(), 0u);

  // Release leases to clean up.
  for (auto& lease : leases) {
    iree_async_buffer_lease_release(&lease);
  }
}

//===----------------------------------------------------------------------===//
// Callback swap (protocol handoff)
//===----------------------------------------------------------------------===//

struct SecondHandler {
  std::vector<ReceivedMessage> messages;

  static iree_status_t OnMessage(void* user_data,
                                 iree_const_byte_span_t message,
                                 iree_async_buffer_lease_t* lease) {
    SecondHandler* handler = static_cast<SecondHandler*>(user_data);
    ReceivedMessage received;
    received.data.assign(message.data, message.data + message.data_length);
    received.had_lease = (lease != nullptr);
    handler->messages.push_back(std::move(received));
    return iree_ok_status();
  }

  static void OnError(void* user_data, iree_status_t status) {
    iree_status_ignore(status);
  }

  iree_net_message_endpoint_callbacks_t MakeCallbacks() {
    return {OnMessage, OnError, this};
  }
};

TEST_F(FramingAdapterTest, CallbackSwapRedirectsMessages) {
  ActivateWithCallbacks();

  auto frame1 = MakeFrame("Before swap");
  IREE_ASSERT_OK(InjectRecv(frame1));
  ASSERT_EQ(ctx_.messages.size(), 1u);

  // Swap to a different handler.
  SecondHandler second_handler;
  iree_net_message_endpoint_set_callbacks(endpoint_,
                                          second_handler.MakeCallbacks());

  auto frame2 = MakeFrame("After swap");
  IREE_ASSERT_OK(InjectRecv(frame2));

  // First handler got frame1 only.
  EXPECT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame1);

  // Second handler got frame2 only.
  ASSERT_EQ(second_handler.messages.size(), 1u);
  EXPECT_EQ(second_handler.messages[0].data, frame2);
}

TEST_F(FramingAdapterTest, CallbackSwapMidFragment) {
  ActivateWithCallbacks();

  // Start a fragmented frame under the first handler.
  auto frame = MakeFrame("Fragment across handlers");
  iree_host_size_t split_point = frame.size() / 2;

  std::vector<uint8_t> first_half(frame.begin(), frame.begin() + split_point);
  IREE_ASSERT_OK(InjectRecv(first_half));
  EXPECT_EQ(ctx_.messages.size(), 0u);

  // Swap handler before the frame is complete.
  SecondHandler second_handler;
  iree_net_message_endpoint_set_callbacks(endpoint_,
                                          second_handler.MakeCallbacks());

  // Complete the frame under the second handler.
  std::vector<uint8_t> second_half(frame.begin() + split_point, frame.end());
  IREE_ASSERT_OK(InjectRecv(second_half));

  // First handler never got the frame.
  EXPECT_EQ(ctx_.messages.size(), 0u);

  // Second handler got the completed frame.
  ASSERT_EQ(second_handler.messages.size(), 1u);
  EXPECT_EQ(second_handler.messages[0].data, frame);
}

//===----------------------------------------------------------------------===//
// Frame size limits
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, OversizedFrameReturnsError) {
  ActivateWithCallbacks();

  // Frame larger than max_frame_size (header-encoded size exceeds limit).
  std::string payload(kMaxFrameSize, 'x');
  auto frame = MakeFrame(payload);
  ASSERT_GT(frame.size(), kMaxFrameSize);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, InjectRecv(frame));
  EXPECT_EQ(ctx_.messages.size(), 0u);
}

TEST_F(FramingAdapterTest, MaxSizeFrameSucceeds) {
  ActivateWithCallbacks();

  // Frame exactly at max_frame_size (header + payload = 1024).
  std::string payload(kMaxFrameSize - kHeaderSize, 'y');
  auto frame = MakeFrame(payload);
  ASSERT_EQ(frame.size(), kMaxFrameSize);

  IREE_ASSERT_OK(InjectRecv(frame));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
}

//===----------------------------------------------------------------------===//
// Endpoint view semantics
//===----------------------------------------------------------------------===//

TEST_F(FramingAdapterTest, EndpointIsCopyable) {
  // Endpoint is a value type (two pointers), can be copied freely.
  iree_net_message_endpoint_t copy = endpoint_;
  EXPECT_EQ(copy.self, endpoint_.self);
  EXPECT_EQ(copy.vtable, endpoint_.vtable);

  // Operations via the copy work the same as via the original.
  iree_net_message_endpoint_set_callbacks(copy, ctx_.MakeCallbacks());
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(copy));

  auto frame = MakeFrame("via copy");
  IREE_ASSERT_OK(InjectRecv(frame));

  ASSERT_EQ(ctx_.messages.size(), 1u);
  EXPECT_EQ(ctx_.messages[0].data, frame);
}

}  // namespace
}  // namespace net
}  // namespace iree
