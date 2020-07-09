// Copyright 2020 Google LLC
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

#include "iree/base/status.h"
#include "iree/base/status_matchers.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/hal/driver_registry.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace cts {
namespace {

using ::iree::testing::status::IsOkAndHolds;
using ::testing::Eq;

class CommandQueueTest : public CtsTestBase {};

TEST_P(CommandQueueTest, EnumerateDeviceQueues) {
  // Log how many queues we have so future test cases have more context.
  // Most tests just use the first queue, but supporting multiple queues may be
  // relevant on some implementations.

  absl::Span<CommandQueue*> dispatch_queues = device_->dispatch_queues();
  LOG(INFO) << "Device has " << dispatch_queues.size() << " dispatch queue(s)";
  EXPECT_GE(dispatch_queues.size(), 1);
  for (auto* dispatch_queue : dispatch_queues) {
    EXPECT_TRUE(dispatch_queue->can_dispatch());
  }

  absl::Span<CommandQueue*> transfer_queues = device_->transfer_queues();
  LOG(INFO) << "Device has " << transfer_queues.size() << " transfer queue(s)";
  EXPECT_GE(transfer_queues.size(), 1);
  for (auto* transfer_queue : transfer_queues) {
    EXPECT_TRUE(transfer_queue->can_transfer());
  }
}

// Tests that waiting for idle is a no-op when nothing is queued.
TEST_P(CommandQueueTest, WaitIdleWhileIdle) {
  for (auto* dispatch_queue : device_->dispatch_queues()) {
    EXPECT_OK(dispatch_queue->WaitIdle());
  }
  for (auto* transfer_queue : device_->transfer_queues()) {
    EXPECT_OK(transfer_queue->WaitIdle());
  }
}

// Tests that submitting a command buffer and immediately waiting will not
// deadlock.
// Note: this test never completes with Vulkan timeline semaphore emulation.
TEST_P(CommandQueueTest, BlockingSubmit) {
  auto command_queue = device_->dispatch_queues()[0];

  ASSERT_OK_AND_ASSIGN(auto command_buffer, device_->CreateCommandBuffer(
                                                CommandBufferMode::kOneShot,
                                                CommandCategory::kDispatch));
  ASSERT_OK_AND_ASSIGN(auto semaphore, device_->CreateSemaphore(0ull));

  ASSERT_OK(command_queue->Submit(
      {{}, {command_buffer.get()}, {{semaphore.get(), 1ull}}}));
  ASSERT_OK(semaphore->Wait(1ull, absl::InfiniteFuture()));
}

// Tests waiting while work is pending/in-flight.
// Note: this test never completes with Vulkan timeline semaphore emulation.
TEST_P(CommandQueueTest, WaitTimeout) {
  auto command_queue = device_->dispatch_queues()[0];

  ASSERT_OK_AND_ASSIGN(auto command_buffer, device_->CreateCommandBuffer(
                                                CommandBufferMode::kOneShot,
                                                CommandCategory::kDispatch));
  ASSERT_OK_AND_ASSIGN(auto wait_semaphore, device_->CreateSemaphore(0ull));
  ASSERT_OK_AND_ASSIGN(auto signal_semaphore, device_->CreateSemaphore(0ull));

  ASSERT_OK(command_queue->Submit({{{wait_semaphore.get(), 1ull}},
                                   {command_buffer.get()},
                                   {{signal_semaphore.get(), 1ull}}}));

  // Work shouldn't start until the wait semaphore reaches its payload value.
  EXPECT_THAT(signal_semaphore->Query(), IsOkAndHolds(Eq(0ull)));
  EXPECT_TRUE(
      IsDeadlineExceeded(command_queue->WaitIdle(absl::Milliseconds(100))));

  // Signal the wait semaphore, work should begin and complete.
  ASSERT_OK(wait_semaphore->Signal(1ull));
  ASSERT_OK(signal_semaphore->Wait(1ull, absl::InfiniteFuture()));
}

// Tests using multiple wait and signal semaphores.
TEST_P(CommandQueueTest, WaitMultiple) {
  auto command_queue = device_->dispatch_queues()[0];

  ASSERT_OK_AND_ASSIGN(auto command_buffer, device_->CreateCommandBuffer(
                                                CommandBufferMode::kOneShot,
                                                CommandCategory::kDispatch));
  ASSERT_OK_AND_ASSIGN(auto wait_semaphore_1, device_->CreateSemaphore(0ull));
  ASSERT_OK_AND_ASSIGN(auto wait_semaphore_2, device_->CreateSemaphore(0ull));
  ASSERT_OK_AND_ASSIGN(auto signal_semaphore_1, device_->CreateSemaphore(0ull));
  ASSERT_OK_AND_ASSIGN(auto signal_semaphore_2, device_->CreateSemaphore(0ull));

  ASSERT_OK(command_queue->Submit(
      {{{wait_semaphore_1.get(), 1ull}, {wait_semaphore_2.get(), 1ull}},
       {command_buffer.get()},
       {{signal_semaphore_1.get(), 1ull}, {signal_semaphore_2.get(), 1ull}}}));

  // Work shouldn't start until the wait semaphore reaches its payload value.
  EXPECT_THAT(signal_semaphore_1->Query(), IsOkAndHolds(Eq(0ull)));
  EXPECT_THAT(signal_semaphore_2->Query(), IsOkAndHolds(Eq(0ull)));
  // Note: This fails with Vulkan timeline semaphore emulation (returns OK)
  EXPECT_TRUE(
      IsDeadlineExceeded(command_queue->WaitIdle(absl::Milliseconds(100))));

  // Signal the wait semaphores, work should only begin after each is set.
  ASSERT_OK(wait_semaphore_1->Signal(1ull));
  EXPECT_THAT(signal_semaphore_1->Query(), IsOkAndHolds(Eq(0ull)));
  EXPECT_THAT(signal_semaphore_2->Query(), IsOkAndHolds(Eq(0ull)));
  ASSERT_OK(wait_semaphore_2->Signal(1ull));

  ASSERT_OK(command_queue->WaitIdle());
}

// Disabled on Vulkan until tests pass when using timeline semaphore emulation.
INSTANTIATE_TEST_SUITE_P(AllDrivers, CommandQueueTest,
                         ::testing::Values("vmla", "llvm", "dylib"),
                         GenerateTestName());

}  // namespace
}  // namespace cts
}  // namespace hal
}  // namespace iree
