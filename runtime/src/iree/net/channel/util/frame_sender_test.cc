// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/frame_sender.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Mock carrier for testing
//===----------------------------------------------------------------------===//

struct MockCarrier;

struct CapturedSend {
  std::vector<std::vector<uint8_t>> span_data;  // Copy of each span's data.
  uint64_t user_data;
  iree_net_send_flags_t flags;
};

struct MockCarrier {
  iree_net_carrier_t base;
  std::vector<CapturedSend> sends;
  iree_status_code_t next_send_error = IREE_STATUS_OK;
  bool auto_complete = true;  // Automatically fire completion on send.
  iree_host_size_t send_budget_bytes = SIZE_MAX;
  uint32_t send_budget_slots = UINT32_MAX;

  // Pending contexts for manual completion.
  std::vector<iree_net_frame_send_context_t*> pending_contexts;

  static void Destroy(iree_net_carrier_t* carrier) {
    // No-op for mock.
  }

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

    // Capture send data.
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

    if (mock->auto_complete) {
      // Fire completion immediately - don't track in pending_contexts.
      carrier->callback.fn(carrier->callback.user_data, params->user_data,
                           iree_ok_status(), 0, nullptr);
    } else {
      // Save context for manual completion later.
      mock->pending_contexts.push_back(
          reinterpret_cast<iree_net_frame_send_context_t*>(params->user_data));
    }

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

  static std::unique_ptr<MockCarrier> Create(
      iree_net_carrier_callback_t callback, iree_host_size_t max_iov = 8) {
    auto mock = std::make_unique<MockCarrier>();
    iree_net_carrier_initialize(&kVtable, IREE_NET_CARRIER_CAPABILITY_RELIABLE,
                                0, max_iov, callback, iree_allocator_system(),
                                &mock->base);
    return mock;
  }

  void FireCompletion(iree_status_t status) {
    if (!pending_contexts.empty()) {
      auto* context = pending_contexts.front();
      pending_contexts.erase(pending_contexts.begin());
      base.callback.fn(base.callback.user_data, (uint64_t)(uintptr_t)context,
                       status, 0, nullptr);
    }
  }

  void FireAllCompletions(iree_status_t status) {
    while (!pending_contexts.empty()) {
      FireCompletion(status);
    }
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
// Test buffer pool helper
//===----------------------------------------------------------------------===//

// Creates a real buffer pool backed by a test region.
// This avoids mocking the pool implementation.
class TestBufferPool {
 public:
  TestBufferPool(iree_host_size_t buffer_count, iree_host_size_t buffer_size)
      : buffer_count_(buffer_count), buffer_size_(buffer_size) {
    // Allocate backing memory.
    buffer_memory_.resize(buffer_count * buffer_size, 0);

    // Create a fake region with the required fields.
    // We allocate it ourselves and set up a destroy_fn that frees it.
    region_ =
        static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
    memset(region_, 0, sizeof(*region_));
    iree_atomic_ref_count_init(&region_->ref_count);
    region_->destroy_fn = DestroyRegion;
    region_->base_ptr = buffer_memory_.data();
    region_->length = buffer_memory_.size();
    region_->buffer_size = buffer_size;
    region_->buffer_count = static_cast<uint32_t>(buffer_count);

    // Create the real pool.
    iree_status_t status = iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &pool_);
    IREE_CHECK_OK(status);

    // Pool took a reference, we can release ours.
    iree_async_region_release(region_);
  }

  ~TestBufferPool() {
    if (pool_) {
      iree_async_buffer_pool_free(pool_);
    }
  }

  iree_async_buffer_pool_t* get() { return pool_; }

  iree_host_size_t AvailableCount() const {
    return iree_async_buffer_pool_available(pool_);
  }

 private:
  static void DestroyRegion(iree_async_region_t* region) {
    // Just free the region struct - the backing memory is owned by the test.
    free(region);
  }

  std::vector<uint8_t> buffer_memory_;
  iree_host_size_t buffer_count_;
  iree_host_size_t buffer_size_;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* pool_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Test context for tracking completions
//===----------------------------------------------------------------------===//

struct CompletionRecord {
  uint64_t operation_user_data;
  iree_status_code_t status_code;
};

struct TestContext {
  std::vector<CompletionRecord> completions;

  static void OnComplete(void* callback_user_data, uint64_t operation_user_data,
                         iree_status_t status) {
    TestContext* ctx = static_cast<TestContext*>(callback_user_data);
    CompletionRecord record;
    record.operation_user_data = operation_user_data;
    record.status_code = iree_status_code(status);
    iree_status_ignore(status);
    ctx->completions.push_back(record);
  }
};

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class FrameSenderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up carrier callback to dispatch to frame_sender completion handler.
    iree_net_carrier_callback_t carrier_callback;
    carrier_callback.fn = CarrierCompletionCallback;
    carrier_callback.user_data = nullptr;

    mock_carrier_ = MockCarrier::Create(carrier_callback);
    test_pool_ = std::make_unique<TestBufferPool>(4, 256);

    // Set up send completion callback.
    iree_net_frame_send_complete_callback_t complete_callback;
    complete_callback.fn = TestContext::OnComplete;
    complete_callback.user_data = &ctx_;

    IREE_ASSERT_OK(iree_net_frame_sender_initialize(
        &sender_, iree_net_frame_sender_carrier_submit, &mock_carrier_->base,
        mock_carrier_->base.max_iov, test_pool_->get(), complete_callback,
        iree_allocator_system(), iree_allocator_system()));
  }

  void TearDown() override {
    // Drain any remaining completions.
    mock_carrier_->FireAllCompletions(iree_ok_status());
    iree_net_frame_sender_deinitialize(&sender_);
  }

  // Carrier completion callback that routes to frame_sender.
  static void CarrierCompletionCallback(void* callback_user_data,
                                        uint64_t operation_user_data,
                                        iree_status_t status,
                                        iree_host_size_t bytes_transferred,
                                        iree_async_buffer_lease_t* recv_lease) {
    auto* context =
        reinterpret_cast<iree_net_frame_send_context_t*>(operation_user_data);
    iree_net_frame_sender_handle_completion(context, status);
  }

  iree_net_frame_sender_t sender_;
  std::unique_ptr<MockCarrier> mock_carrier_;
  std::unique_ptr<TestBufferPool> test_pool_;
  TestContext ctx_;
};

//===----------------------------------------------------------------------===//
// Lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(FrameSenderTest, InitializeSetsFields) {
  EXPECT_EQ(sender_.submit_fn, iree_net_frame_sender_carrier_submit);
  EXPECT_EQ(sender_.submit_fn_user_data, &mock_carrier_->base);
  EXPECT_EQ(sender_.header_pool, test_pool_->get());
  EXPECT_FALSE(sender_.has_batch_lease);
  EXPECT_EQ(sender_.batch_used, 0u);
  EXPECT_FALSE(iree_net_frame_sender_has_pending(&sender_));
}

TEST_F(FrameSenderTest, DeinitializeClearsFields) {
  iree_net_frame_sender_deinitialize(&sender_);

  EXPECT_EQ(sender_.submit_fn, nullptr);
  EXPECT_EQ(sender_.header_pool, nullptr);
  EXPECT_FALSE(sender_.has_batch_lease);

  // Re-initialize for TearDown to work.
  iree_net_frame_send_complete_callback_t complete_callback;
  complete_callback.fn = TestContext::OnComplete;
  complete_callback.user_data = &ctx_;
  IREE_ASSERT_OK(iree_net_frame_sender_initialize(
      &sender_, iree_net_frame_sender_carrier_submit, &mock_carrier_->base,
      mock_carrier_->base.max_iov, test_pool_->get(), complete_callback,
      iree_allocator_system(), iree_allocator_system()));
}

//===----------------------------------------------------------------------===//
// send() tests
//===----------------------------------------------------------------------===//

TEST_F(FrameSenderTest, SendWithEmptyPayload) {
  const uint8_t header_data[] = {0x01, 0x02, 0x03, 0x04};
  iree_const_byte_span_t header =
      iree_make_const_byte_span(header_data, sizeof(header_data));
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_ASSERT_OK(iree_net_frame_sender_send(&sender_, header, payload, 42));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  ASSERT_EQ(mock_carrier_->sends[0].span_data.size(), 1u);
  EXPECT_EQ(
      mock_carrier_->sends[0].span_data[0],
      std::vector<uint8_t>(header_data, header_data + sizeof(header_data)));

  // Completion should have fired.
  ASSERT_EQ(ctx_.completions.size(), 1u);
  EXPECT_EQ(ctx_.completions[0].operation_user_data, 42u);
  EXPECT_EQ(ctx_.completions[0].status_code, IREE_STATUS_OK);
}

TEST_F(FrameSenderTest, SendWithPayload) {
  mock_carrier_->auto_complete = false;

  const uint8_t header_data[] = {0x01, 0x02};
  iree_const_byte_span_t header =
      iree_make_const_byte_span(header_data, sizeof(header_data));

  // Create payload span.
  std::vector<uint8_t> payload_data = {0xAA, 0xBB, 0xCC, 0xDD};
  iree_async_span_t payload_span =
      iree_async_span_from_ptr(payload_data.data(), payload_data.size());
  iree_async_span_list_t payload = {&payload_span, 1};

  IREE_ASSERT_OK(iree_net_frame_sender_send(&sender_, header, payload, 100));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  ASSERT_EQ(mock_carrier_->sends[0].span_data.size(), 2u);  // Header + payload.
  EXPECT_EQ(
      mock_carrier_->sends[0].span_data[0],
      std::vector<uint8_t>(header_data, header_data + sizeof(header_data)));
  EXPECT_EQ(mock_carrier_->sends[0].span_data[1], payload_data);

  EXPECT_TRUE(iree_net_frame_sender_has_pending(&sender_));
  EXPECT_EQ(iree_net_frame_sender_pending_count(&sender_), 1);

  // Fire completion manually.
  mock_carrier_->FireCompletion(iree_ok_status());

  EXPECT_FALSE(iree_net_frame_sender_has_pending(&sender_));
  ASSERT_EQ(ctx_.completions.size(), 1u);
  EXPECT_EQ(ctx_.completions[0].operation_user_data, 100u);
}

TEST_F(FrameSenderTest, SendTooManySpans) {
  const uint8_t header_data[] = {0x01};
  iree_const_byte_span_t header =
      iree_make_const_byte_span(header_data, sizeof(header_data));

  // Create more spans than allowed (MAX_SPANS = 8, header takes 1).
  std::vector<uint8_t> dummy = {0x00};
  iree_async_span_t spans[IREE_NET_FRAME_SENDER_MAX_SPANS];
  for (int i = 0; i < IREE_NET_FRAME_SENDER_MAX_SPANS; ++i) {
    spans[i] = iree_async_span_from_ptr(dummy.data(), 1);
  }

  // 1 header + 8 payload = 9 spans, but max is 8.
  iree_async_span_list_t payload = {spans, IREE_NET_FRAME_SENDER_MAX_SPANS};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_frame_sender_send(&sender_, header, payload, 0));

  EXPECT_EQ(mock_carrier_->sends.size(), 0u);
  EXPECT_EQ(ctx_.completions.size(), 0u);  // No callback on failure.
}

TEST_F(FrameSenderTest, SendRespectsCarrierMaxIov) {
  // Create carrier with max_iov = 2.
  iree_net_carrier_callback_t carrier_callback;
  carrier_callback.fn = CarrierCompletionCallback;
  carrier_callback.user_data = nullptr;
  auto limited_carrier = MockCarrier::Create(carrier_callback, 2);

  iree_net_frame_sender_t limited_sender;
  iree_net_frame_send_complete_callback_t complete_callback;
  complete_callback.fn = TestContext::OnComplete;
  complete_callback.user_data = &ctx_;
  IREE_ASSERT_OK(iree_net_frame_sender_initialize(
      &limited_sender, iree_net_frame_sender_carrier_submit,
      &limited_carrier->base, limited_carrier->base.max_iov, test_pool_->get(),
      complete_callback, iree_allocator_system(), iree_allocator_system()));

  const uint8_t header_data[] = {0x01};
  iree_const_byte_span_t header =
      iree_make_const_byte_span(header_data, sizeof(header_data));

  // Try to send header + 2 payload spans (3 total, exceeds max_iov = 2).
  std::vector<uint8_t> dummy = {0x00};
  iree_async_span_t spans[2];
  for (int i = 0; i < 2; ++i) {
    spans[i] = iree_async_span_from_ptr(dummy.data(), 1);
  }
  iree_async_span_list_t payload = {spans, 2};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_frame_sender_send(&limited_sender, header, payload, 0));

  iree_net_frame_sender_deinitialize(&limited_sender);
}

TEST_F(FrameSenderTest, SendPoolExhausted) {
  // Exhaust the pool by acquiring all buffers without completing them.
  mock_carrier_->auto_complete = false;
  const uint8_t header_data[] = {0x01};
  iree_const_byte_span_t header =
      iree_make_const_byte_span(header_data, sizeof(header_data));
  iree_async_span_list_t payload = {nullptr, 0};

  // Pool has 4 buffers. Send 4 to exhaust it.
  for (int i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(iree_net_frame_sender_send(&sender_, header, payload, i));
  }

  // 5th send should fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_frame_sender_send(&sender_, header, payload, 99));

  EXPECT_EQ(mock_carrier_->sends.size(), 4u);

  // Complete all to release buffers for TearDown.
  mock_carrier_->FireAllCompletions(iree_ok_status());
}

TEST_F(FrameSenderTest, SendCarrierBackpressure) {
  iree_host_size_t available_before = test_pool_->AvailableCount();
  mock_carrier_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;

  const uint8_t header_data[] = {0x01};
  iree_const_byte_span_t header =
      iree_make_const_byte_span(header_data, sizeof(header_data));
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_frame_sender_send(&sender_, header, payload, 0));

  // Buffer should be released back to pool.
  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
  EXPECT_EQ(ctx_.completions.size(), 0u);
}

TEST_F(FrameSenderTest, SendHeaderExceedsPoolBuffer) {
  iree_host_size_t available_before = test_pool_->AvailableCount();

  // Create a header larger than the pool buffer size (256 bytes).
  std::vector<uint8_t> large_header(300, 0xAA);
  iree_const_byte_span_t header =
      iree_make_const_byte_span(large_header.data(), large_header.size());
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_frame_sender_send(&sender_, header, payload, 0));

  // Buffer should be released on error.
  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
}

//===----------------------------------------------------------------------===//
// queue() / flush() tests
//===----------------------------------------------------------------------===//

TEST_F(FrameSenderTest, QueueAccumulatesFrames) {
  const uint8_t frame1[] = {0x01, 0x02, 0x03};
  const uint8_t frame2[] = {0x04, 0x05};

  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(frame1, sizeof(frame1))));
  EXPECT_EQ(iree_net_frame_sender_queued_bytes(&sender_), 3u);
  EXPECT_TRUE(sender_.has_batch_lease);

  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(frame2, sizeof(frame2))));
  EXPECT_EQ(iree_net_frame_sender_queued_bytes(&sender_), 5u);

  // No carrier sends yet.
  EXPECT_EQ(mock_carrier_->sends.size(), 0u);
}

TEST_F(FrameSenderTest, FlushSendsBatch) {
  const uint8_t frame1[] = {0x01, 0x02};
  const uint8_t frame2[] = {0x03, 0x04, 0x05};

  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(frame1, sizeof(frame1))));
  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(frame2, sizeof(frame2))));
  IREE_ASSERT_OK(iree_net_frame_sender_flush(&sender_, 99));

  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  ASSERT_EQ(mock_carrier_->sends[0].span_data.size(), 1u);

  // Combined data.
  std::vector<uint8_t> expected = {0x01, 0x02, 0x03, 0x04, 0x05};
  EXPECT_EQ(mock_carrier_->sends[0].span_data[0], expected);

  // Completion with our user_data.
  ASSERT_EQ(ctx_.completions.size(), 1u);
  EXPECT_EQ(ctx_.completions[0].operation_user_data, 99u);

  // Batch state cleared.
  EXPECT_FALSE(sender_.has_batch_lease);
  EXPECT_EQ(sender_.batch_used, 0u);
}

TEST_F(FrameSenderTest, FlushEmptyIsNoOp) {
  IREE_ASSERT_OK(iree_net_frame_sender_flush(&sender_, 0));

  EXPECT_EQ(mock_carrier_->sends.size(), 0u);
  EXPECT_EQ(ctx_.completions.size(), 0u);
}

TEST_F(FrameSenderTest, QueueReturnsResourceExhaustedWhenBatchFull) {
  // Fill up the batch buffer (256 bytes).
  std::vector<uint8_t> data(200, 0xAA);
  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(data.data(), data.size())));

  // Try to queue more than remaining space.
  std::vector<uint8_t> overflow(100, 0xBB);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_frame_sender_queue(
                            &sender_, iree_make_const_byte_span(
                                          overflow.data(), overflow.size())));
}

TEST_F(FrameSenderTest, FlushFailurePreservesBatchData) {
  mock_carrier_->next_send_error = IREE_STATUS_RESOURCE_EXHAUSTED;

  const uint8_t frame[] = {0x01, 0x02, 0x03};
  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(frame, sizeof(frame))));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_frame_sender_flush(&sender_, 0));

  // Data should still be there for retry.
  EXPECT_TRUE(sender_.has_batch_lease);
  EXPECT_EQ(sender_.batch_used, 3u);
  EXPECT_EQ(ctx_.completions.size(), 0u);

  // Retry should work.
  IREE_ASSERT_OK(iree_net_frame_sender_flush(&sender_, 0));
  EXPECT_EQ(mock_carrier_->sends.size(), 1u);
}

//===----------------------------------------------------------------------===//
// Ordering tests (caller-managed)
//===----------------------------------------------------------------------===//

TEST_F(FrameSenderTest, SendDoesNotAutoFlush) {
  // send() is thread-safe and does NOT auto-flush batched data.
  // Caller must flush explicitly if ordering with queue() matters.
  const uint8_t queued_frame[] = {0x01, 0x02};
  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(queued_frame, sizeof(queued_frame))));

  const uint8_t header[] = {0xAA, 0xBB};
  iree_const_byte_span_t send_header =
      iree_make_const_byte_span(header, sizeof(header));
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_ASSERT_OK(iree_net_frame_sender_send(&sender_, send_header, payload, 0));

  // Only the send(), not a flush - batched data still pending.
  ASSERT_EQ(mock_carrier_->sends.size(), 1u);
  EXPECT_EQ(mock_carrier_->sends[0].span_data[0],
            std::vector<uint8_t>({0xAA, 0xBB}));

  // Batch still has data.
  EXPECT_TRUE(sender_.has_batch_lease);
  EXPECT_EQ(sender_.batch_used, 2u);
}

TEST_F(FrameSenderTest, ExplicitFlushThenSendMaintainsOrder) {
  // Demonstrates the correct pattern: flush() before send() for ordering.
  const uint8_t queued_frame[] = {0x01, 0x02};
  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(queued_frame, sizeof(queued_frame))));

  // Explicit flush first.
  IREE_ASSERT_OK(iree_net_frame_sender_flush(&sender_, 1));

  const uint8_t header[] = {0xAA, 0xBB};
  iree_const_byte_span_t send_header =
      iree_make_const_byte_span(header, sizeof(header));
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_ASSERT_OK(iree_net_frame_sender_send(&sender_, send_header, payload, 2));

  // Should have 2 sends in order: flush then send.
  ASSERT_EQ(mock_carrier_->sends.size(), 2u);
  EXPECT_EQ(mock_carrier_->sends[0].span_data[0],
            std::vector<uint8_t>({0x01, 0x02}));
  EXPECT_EQ(mock_carrier_->sends[1].span_data[0],
            std::vector<uint8_t>({0xAA, 0xBB}));

  // Completions in order.
  ASSERT_EQ(ctx_.completions.size(), 2u);
  EXPECT_EQ(ctx_.completions[0].operation_user_data, 1u);  // flush
  EXPECT_EQ(ctx_.completions[1].operation_user_data, 2u);  // send
}

//===----------------------------------------------------------------------===//
// Completion tests
//===----------------------------------------------------------------------===//

TEST_F(FrameSenderTest, CompletionReleasesBufferAndFiresCallback) {
  mock_carrier_->auto_complete = false;
  iree_host_size_t available_before = test_pool_->AvailableCount();

  const uint8_t header[] = {0x01};
  iree_const_byte_span_t send_header =
      iree_make_const_byte_span(header, sizeof(header));
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_ASSERT_OK(
      iree_net_frame_sender_send(&sender_, send_header, payload, 123));

  EXPECT_EQ(test_pool_->AvailableCount(), available_before - 1);
  EXPECT_EQ(ctx_.completions.size(), 0u);

  mock_carrier_->FireCompletion(iree_ok_status());

  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
  ASSERT_EQ(ctx_.completions.size(), 1u);
  EXPECT_EQ(ctx_.completions[0].operation_user_data, 123u);
  EXPECT_EQ(ctx_.completions[0].status_code, IREE_STATUS_OK);
}

TEST_F(FrameSenderTest, CompletionWithError) {
  mock_carrier_->auto_complete = false;

  const uint8_t header[] = {0x01};
  iree_const_byte_span_t send_header =
      iree_make_const_byte_span(header, sizeof(header));
  iree_async_span_list_t payload = {nullptr, 0};

  IREE_ASSERT_OK(
      iree_net_frame_sender_send(&sender_, send_header, payload, 456));

  mock_carrier_->FireCompletion(
      iree_make_status(IREE_STATUS_INTERNAL, "test error"));

  ASSERT_EQ(ctx_.completions.size(), 1u);
  EXPECT_EQ(ctx_.completions[0].operation_user_data, 456u);
  EXPECT_EQ(ctx_.completions[0].status_code, IREE_STATUS_INTERNAL);
}

TEST_F(FrameSenderTest, MultipleInFlightSends) {
  mock_carrier_->auto_complete = false;
  iree_host_size_t available_before = test_pool_->AvailableCount();  // 4

  for (int i = 0; i < 3; ++i) {
    const uint8_t header[] = {static_cast<uint8_t>(i)};
    iree_const_byte_span_t send_header =
        iree_make_const_byte_span(header, sizeof(header));
    iree_async_span_list_t payload = {nullptr, 0};
    IREE_ASSERT_OK(
        iree_net_frame_sender_send(&sender_, send_header, payload, i));
  }

  EXPECT_EQ(iree_net_frame_sender_pending_count(&sender_), 3);
  EXPECT_EQ(test_pool_->AvailableCount(), available_before - 3);  // 1 left.

  // Complete all in order.
  for (int i = 0; i < 3; ++i) {
    mock_carrier_->FireCompletion(iree_ok_status());
    EXPECT_EQ(iree_net_frame_sender_pending_count(&sender_), 2 - i);
  }

  EXPECT_EQ(test_pool_->AvailableCount(), available_before);
  ASSERT_EQ(ctx_.completions.size(), 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(ctx_.completions[i].operation_user_data,
              static_cast<uint64_t>(i));
  }
}

//===----------------------------------------------------------------------===//
// Query tests
//===----------------------------------------------------------------------===//

TEST_F(FrameSenderTest, QueuedBytesQuery) {
  EXPECT_EQ(iree_net_frame_sender_queued_bytes(&sender_), 0u);

  const uint8_t frame[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  IREE_ASSERT_OK(iree_net_frame_sender_queue(
      &sender_, iree_make_const_byte_span(frame, sizeof(frame))));

  EXPECT_EQ(iree_net_frame_sender_queued_bytes(&sender_), 5u);
}

TEST_F(FrameSenderTest, HasPendingAndPendingCount) {
  mock_carrier_->auto_complete = false;

  EXPECT_FALSE(iree_net_frame_sender_has_pending(&sender_));
  EXPECT_EQ(iree_net_frame_sender_pending_count(&sender_), 0);

  const uint8_t header[] = {0x01};
  iree_const_byte_span_t send_header =
      iree_make_const_byte_span(header, sizeof(header));
  iree_async_span_list_t payload = {nullptr, 0};
  IREE_ASSERT_OK(iree_net_frame_sender_send(&sender_, send_header, payload, 0));

  EXPECT_TRUE(iree_net_frame_sender_has_pending(&sender_));
  EXPECT_EQ(iree_net_frame_sender_pending_count(&sender_), 1);

  mock_carrier_->FireCompletion(iree_ok_status());

  EXPECT_FALSE(iree_net_frame_sender_has_pending(&sender_));
  EXPECT_EQ(iree_net_frame_sender_pending_count(&sender_), 0);
}

}  // namespace
}  // namespace net
}  // namespace iree
