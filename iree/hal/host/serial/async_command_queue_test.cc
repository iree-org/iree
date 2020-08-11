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

#include "iree/hal/host/serial/async_command_queue.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include "absl/memory/memory.h"
#include "iree/base/status.h"
#include "iree/base/time.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/host/serial/serial_submission_queue.h"
#include "iree/hal/testing/mock_command_buffer.h"
#include "iree/hal/testing/mock_command_queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace host {
namespace {

using ::testing::_;

using testing::MockCommandBuffer;
using testing::MockCommandQueue;

// Suspends execution of the calling thread for the given |duration_ms|.
// Depending on platform this may have an extremely coarse resolution (upwards
// of several to dozens of milliseconds).
inline void Sleep(std::chrono::milliseconds duration_ms) {
  std::this_thread::sleep_for(duration_ms);
}

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

  auto cmd_buffer = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                CommandCategory::kTransfer);

  EXPECT_CALL(*mock_target_queue, Submit(_))
      .WillOnce([&](absl::Span<const SubmissionBatch> batches) {
        CHECK_EQ(1, batches.size());
        CHECK_EQ(1, batches[0].command_buffers.size());
        CHECK_EQ(cmd_buffer.get(), batches[0].command_buffers[0]);
        return OkStatus();
      });
  CondVarSemaphore semaphore(0ull);
  IREE_ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer.get()}, {{&semaphore, 1ull}}}));
  IREE_ASSERT_OK(semaphore.Wait(1ull, InfiniteFuture()));
}

// Tests that failure is propagated along the fence from the target queue.
TEST_F(AsyncCommandQueueTest, PropagateSubmitFailure) {
  ::testing::InSequence sequence;

  auto cmd_buffer = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                CommandCategory::kTransfer);

  EXPECT_CALL(*mock_target_queue, Submit(_))
      .WillOnce([](absl::Span<const SubmissionBatch> batches) {
        return DataLossErrorBuilder(IREE_LOC);
      });
  CondVarSemaphore semaphore(0ull);
  IREE_ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer.get()}, {{&semaphore, 1ull}}}));
  EXPECT_TRUE(IsDataLoss(semaphore.Wait(1ull, InfiniteFuture())));
}

// Tests that waiting for idle is a no-op when nothing is queued.
TEST_F(AsyncCommandQueueTest, WaitIdleWhileIdle) {
  IREE_ASSERT_OK(command_queue->WaitIdle());
}

// Tests that waiting for idle will block when work is pending/in-flight.
TEST_F(AsyncCommandQueueTest, WaitIdleWithPending) {
  ::testing::InSequence sequence;

  auto cmd_buffer = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                CommandCategory::kTransfer);

  EXPECT_CALL(*mock_target_queue, Submit(_))
      .WillOnce([](absl::Span<const SubmissionBatch> batches) {
        Sleep(std::chrono::milliseconds(100));
        return OkStatus();
      });
  CondVarSemaphore semaphore(0ull);
  IREE_ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer.get()}, {{&semaphore, 1ull}}}));

  // This should block for a sec or two.
  IREE_ASSERT_OK(command_queue->WaitIdle());

  // Should have already expired.
  IREE_ASSERT_OK_AND_ASSIGN(uint64_t value, semaphore.Query());
  ASSERT_EQ(1ull, value);
}

// Tests that waiting for idle with multiple pending submissions will wait until
// all of them complete while still allowing incremental progress.
TEST_F(AsyncCommandQueueTest, WaitIdleAndProgress) {
  ::testing::InSequence sequence;

  EXPECT_CALL(*mock_target_queue, Submit(_))
      .WillRepeatedly([](absl::Span<const SubmissionBatch> batches) {
        Sleep(std::chrono::milliseconds(100));
        return OkStatus();
      });

  auto cmd_buffer_0 = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                  CommandCategory::kTransfer);
  auto cmd_buffer_1 = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                  CommandCategory::kTransfer);

  CondVarSemaphore semaphore_0(0u);
  IREE_ASSERT_OK(command_queue->Submit(
      {{}, {cmd_buffer_0.get()}, {{&semaphore_0, 1ull}}}));
  CondVarSemaphore semaphore_1(0u);
  IREE_ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer_1.get()}, {{&semaphore_1, 1u}}}));

  // This should block for a sec or two.
  IREE_ASSERT_OK(command_queue->WaitIdle());

  // Both should have already expired.
  IREE_ASSERT_OK_AND_ASSIGN(uint64_t value_0, semaphore_0.Query());
  ASSERT_EQ(1ull, value_0);
  IREE_ASSERT_OK_AND_ASSIGN(uint64_t value_1, semaphore_1.Query());
  ASSERT_EQ(1ull, value_1);
}

// Tests that failures are sticky.
TEST_F(AsyncCommandQueueTest, StickyFailures) {
  ::testing::InSequence sequence;

  // Fail.
  EXPECT_CALL(*mock_target_queue, Submit(_))
      .WillOnce([](absl::Span<const SubmissionBatch> batches) {
        Sleep(std::chrono::milliseconds(100));
        return DataLossErrorBuilder(IREE_LOC);
      });
  auto cmd_buffer_0 = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                  CommandCategory::kTransfer);
  CondVarSemaphore semaphore_0(0ull);
  IREE_ASSERT_OK(
      command_queue->Submit({{}, {cmd_buffer_0.get()}, {{&semaphore_0, 1u}}}));
  EXPECT_TRUE(IsDataLoss(semaphore_0.Wait(1ull, InfiniteFuture())));

  // Future flushes/waits/etc should also fail.
  EXPECT_TRUE(IsDataLoss(command_queue->WaitIdle()));

  // Future submits should fail asynchronously.
  auto cmd_buffer_1 = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                  CommandCategory::kTransfer);
  CondVarSemaphore semaphore_1(0ull);
  EXPECT_TRUE(IsDataLoss(command_queue->Submit(
      {{}, {cmd_buffer_1.get()}, {{&semaphore_1, 1ull}}})));
}

// Tests that a failure with two submissions pending causes the second to
// bail as well.
TEST_F(AsyncCommandQueueTest, FailuresCascadeAcrossSubmits) {
  ::testing::InSequence sequence;

  // Fail.
  EXPECT_CALL(*mock_target_queue, Submit(_))
      .WillOnce([](absl::Span<const SubmissionBatch> batches) {
        Sleep(std::chrono::milliseconds(100));
        return DataLossErrorBuilder(IREE_LOC);
      });

  auto cmd_buffer_0 = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                  CommandCategory::kTransfer);
  auto cmd_buffer_1 = make_ref<MockCommandBuffer>(CommandBufferMode::kOneShot,
                                                  CommandCategory::kTransfer);

  CondVarSemaphore semaphore_0(0ull);
  IREE_ASSERT_OK(command_queue->Submit(
      {{}, {cmd_buffer_0.get()}, {{&semaphore_0, 1ull}}}));
  CondVarSemaphore semaphore_1(0ull);
  IREE_ASSERT_OK(command_queue->Submit(
      {{{&semaphore_0, 1ull}}, {cmd_buffer_1.get()}, {{&semaphore_1, 1ull}}}));

  EXPECT_TRUE(IsDataLoss(command_queue->WaitIdle()));

  EXPECT_TRUE(IsDataLoss(semaphore_0.Wait(1ull, InfiniteFuture())));
  EXPECT_TRUE(IsDataLoss(semaphore_1.Wait(1ull, InfiniteFuture())));

  // Future flushes/waits/etc should also fail.
  EXPECT_TRUE(IsDataLoss(command_queue->WaitIdle()));
}

}  // namespace
}  // namespace host
}  // namespace hal
}  // namespace iree
