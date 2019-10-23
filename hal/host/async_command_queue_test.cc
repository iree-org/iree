// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hal/host/async_command_queue.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "base/status.h"
#include "base/status_matchers.h"
#include "base/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hal/command_queue.h"
#include "hal/host/host_submission_queue.h"
#include "hal/testing/mock_command_buffer.h"
#include "hal/testing/mock_command_queue.h"

namespace iree {
namespace hal {
namespace {

using ::testing::_;

using testing::MockCommandBuffer;
using testing::MockCommandQueue;

struct AsyncCommandQueueTest : public ::testing::Test {
  MockCommandQueue* mock_target_queue;
  std::unique_ptr<CommandQueue> command_queue;

  void SetUp() override {
    auto mock_queue = absl::make_unique<MockCommandQueue>(
        "mock", CommandCategory::kTransfer | CommandCategory::kDispatch);
    mock_target_queue = mock_queue.get();
    command_queue = absl::make_unique<AsyncCommandQueue>(std::move(mock_queue));
  }

  void TearDown() override {
    command_queue.reset();
    mock_target_queue = nullptr;
  }
};

// Tests that submitting a command buffer and immediately waiting will not
// deadlock.
TEST_F(AsyncCommandQueueTest, BlockingSubmit) {
  ::testing::InSequence sequence;

  auto cmd_buffer = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);

  EXPECT_CALL(*mock_target_queue, Submit(_, _))
      .WillOnce(
          [&](absl::Span<const SubmissionBatch> batches, FenceValue fence) {
            CHECK_EQ(1, batches.size());
            CHECK_EQ(1, batches[0].command_buffers.size());
            CHECK_EQ(cmd_buffer.get(), batches[0].command_buffers[0]);
            CHECK_EQ(nullptr, fence.first);
            return OkStatus();
          });
  HostFence fence(0u);
  ASSERT_OK(command_queue->Submit({{}, {cmd_buffer.get()}, {}}, {&fence, 1u}));
  ASSERT_OK(HostFence::WaitForFences({{&fence, 1u}}, /*wait_all=*/true,
                                     absl::InfiniteFuture()));
}

// Tests that failure is propagated along the fence from the target queue.
TEST_F(AsyncCommandQueueTest, PropagateSubmitFailure) {
  ::testing::InSequence sequence;

  auto cmd_buffer = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);

  EXPECT_CALL(*mock_target_queue, Submit(_, _))
      .WillOnce(
          [](absl::Span<const SubmissionBatch> batches, FenceValue fence) {
            return DataLossErrorBuilder(IREE_LOC);
          });
  HostFence fence(0u);
  ASSERT_OK(command_queue->Submit({{}, {cmd_buffer.get()}, {}}, {&fence, 1u}));
  EXPECT_TRUE(IsDataLoss(HostFence::WaitForFences(
      {{&fence, 1u}}, /*wait_all=*/true, absl::InfiniteFuture())));
}

// Tests that waiting for idle is a no-op when nothing is queued.
TEST_F(AsyncCommandQueueTest, WaitIdleWhileIdle) {
  ASSERT_OK(command_queue->WaitIdle());
}

// Tests that waiting for idle will block when work is pending/in-flight.
TEST_F(AsyncCommandQueueTest, WaitIdleWithPending) {
  ::testing::InSequence sequence;

  auto cmd_buffer = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);

  EXPECT_CALL(*mock_target_queue, Submit(_, _))
      .WillOnce(
          [](absl::Span<const SubmissionBatch> batches, FenceValue fence) {
            Sleep(absl::Milliseconds(100));
            return OkStatus();
          });
  HostFence fence(0u);
  ASSERT_OK(command_queue->Submit({{}, {cmd_buffer.get()}, {}}, {&fence, 1u}));

  // This should block for a sec or two.
  ASSERT_OK(command_queue->WaitIdle());

  // Should have already expired.
  ASSERT_OK_AND_ASSIGN(uint64_t value, fence.QueryValue());
  ASSERT_EQ(1u, value);
}

// Tests that waiting for idle with multiple pending submissions will wait until
// all of them complete while still allowing incremental progress.
TEST_F(AsyncCommandQueueTest, WaitIdleAndProgress) {
  ::testing::InSequence sequence;

  EXPECT_CALL(*mock_target_queue, Submit(_, _))
      .WillRepeatedly(
          [](absl::Span<const SubmissionBatch> batches, FenceValue fence) {
            Sleep(absl::Milliseconds(100));
            return OkStatus();
          });

  auto cmd_buffer_0 = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);
  auto cmd_buffer_1 = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);

  HostFence fence_0(0u);
  ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer_0.get()}, {}}, {&fence_0, 1u}));
  HostFence fence_1(0u);
  ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer_1.get()}, {}}, {&fence_1, 1u}));

  // This should block for a sec or two.
  ASSERT_OK(command_queue->WaitIdle());

  // Both should have already expired.
  ASSERT_OK_AND_ASSIGN(uint64_t value_0, fence_0.QueryValue());
  ASSERT_EQ(1u, value_0);
  ASSERT_OK_AND_ASSIGN(uint64_t value_1, fence_1.QueryValue());
  ASSERT_EQ(1u, value_1);
}

// Tests that failures are sticky.
TEST_F(AsyncCommandQueueTest, StickyFailures) {
  ::testing::InSequence sequence;

  // Fail.
  EXPECT_CALL(*mock_target_queue, Submit(_, _))
      .WillOnce(
          [](absl::Span<const SubmissionBatch> batches, FenceValue fence) {
            Sleep(absl::Milliseconds(100));
            return DataLossErrorBuilder(IREE_LOC);
          });
  auto cmd_buffer_0 = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);
  HostFence fence_0(0u);
  ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer_0.get()}, {}}, {&fence_0, 1u}));
  EXPECT_TRUE(IsDataLoss(HostFence::WaitForFences(
      {{&fence_0, 1u}}, /*wait_all=*/true, absl::InfiniteFuture())));

  // Future flushes/waits/etc should also fail.
  EXPECT_TRUE(IsDataLoss(command_queue->WaitIdle()));

  // Future submits should fail asynchronously.
  auto cmd_buffer_1 = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);
  HostFence fence_1(0u);
  EXPECT_TRUE(IsDataLoss(
      command_queue->Submit({{}, {cmd_buffer_1.get()}, {}}, {&fence_1, 1u})));
}

// Tests that a failure with two submissions pending causes the second to
// bail as well.
TEST_F(AsyncCommandQueueTest, FailuresCascadeAcrossSubmits) {
  ::testing::InSequence sequence;

  // Fail.
  EXPECT_CALL(*mock_target_queue, Submit(_, _))
      .WillOnce(
          [](absl::Span<const SubmissionBatch> batches, FenceValue fence) {
            Sleep(absl::Milliseconds(100));
            return DataLossErrorBuilder(IREE_LOC);
          });

  auto cmd_buffer_0 = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);
  auto cmd_buffer_1 = make_ref<MockCommandBuffer>(
      nullptr, CommandBufferMode::kOneShot, CommandCategory::kTransfer);

  HostBinarySemaphore semaphore_0_1(false);
  HostFence fence_0(0u);
  ASSERT_OK(command_queue->Submit({{}, {cmd_buffer_0.get()}, {&semaphore_0_1}},
                                  {&fence_0, 1u}));
  HostFence fence_1(0u);
  ASSERT_OK(command_queue->Submit({{&semaphore_0_1}, {cmd_buffer_1.get()}, {}},
                                  {&fence_1, 1u}));

  EXPECT_TRUE(IsDataLoss(command_queue->WaitIdle()));

  EXPECT_TRUE(IsDataLoss(HostFence::WaitForFences(
      {{&fence_0, 1u}}, /*wait_all=*/true, absl::InfiniteFuture())));
  EXPECT_TRUE(IsDataLoss(HostFence::WaitForFences(
      {{&fence_1, 1u}}, /*wait_all=*/true, absl::InfiniteFuture())));

  // Future flushes/waits/etc should also fail.
  EXPECT_TRUE(IsDataLoss(command_queue->WaitIdle()));
}

}  // namespace
}  // namespace hal
}  // namespace iree
