// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan-specific queue API tests

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/local/transient_buffer.h"

namespace iree::hal::cts {

namespace {

constexpr iree_device_size_t kBufferSize = 4096;

// Issues a queue_dealloca of |buffer| with no waits and a fresh signal
// semaphore and returns its status.
iree_status_t TryQueueDealloca(
    iree_hal_device_t* device, iree_hal_buffer_t* buffer,
    iree_hal_dealloca_flags_t flags = IREE_HAL_DEALLOCA_FLAG_NONE) {
  SemaphoreList empty_wait;
  SemaphoreList signal(device, {0}, {1});
  return iree_hal_device_queue_dealloca(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                        empty_wait, signal, buffer, flags);
}

}  // namespace

// The Vulkan queue accepts at most one queue_dealloca per transient buffer.
// The first accepted queue_dealloca sets the transient's dealloca_queued flag
// and every later queue_dealloca of the same buffer is rejected. A rejected
// call must not clear the flag: only the call that set it may, and only when
// its own submission fails.
class VulkanQueueDeallocaTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (IsSkipped() || HasFatalFailure()) return;
    gate_ = SemaphoreList(device_, {0}, {1});
    first_signal_ = SemaphoreList(device_, {0}, {1});
    ASSERT_NO_FATAL_FAILURE(AllocateCommittedTransient());
  }

  void TearDown() override {
    buffer_.reset();
    gate_ = SemaphoreList();
    first_signal_ = SemaphoreList();
    CtsTestBase<>::TearDown();
  }

  void AllocateCommittedTransient() {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    SemaphoreList empty_wait;
    SemaphoreList alloca_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        /*pool=*/NULL, params, kBufferSize, IREE_HAL_ALLOCA_FLAG_NONE,
        buffer_.out()));
    ASSERT_TRUE(iree_hal_local_transient_buffer_isa(buffer_));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    ASSERT_TRUE(iree_hal_local_transient_buffer_is_committed(buffer_));
  }

  // Queues the first dealloca behind |gate_| so it stays pending until
  // ReleaseFirstDealloca().
  void QueueFirstDealloca() {
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, gate_, first_signal_, buffer_,
        IREE_HAL_DEALLOCA_FLAG_NONE));
    ASSERT_TRUE(iree_hal_local_transient_buffer_is_dealloca_queued(buffer_));
  }

  // Expects the first dealloca to still be pending: queued but not executed.
  void ExpectFirstDeallocaPending() {
    EXPECT_TRUE(iree_hal_local_transient_buffer_is_dealloca_queued(buffer_));
    EXPECT_TRUE(iree_hal_local_transient_buffer_is_committed(buffer_));
  }

  // Lets the first dealloca execute and expects it to release the backing.
  void ReleaseFirstDealloca() {
    IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate_, /*frontier=*/NULL));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        first_signal_, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_FALSE(iree_hal_local_transient_buffer_is_committed(buffer_));
    EXPECT_EQ(iree_hal_local_transient_buffer_backing_buffer(buffer_), nullptr);
  }

  Ref<iree_hal_buffer_t> buffer_;
  SemaphoreList gate_;
  SemaphoreList first_signal_;
};

// A second dealloca while the first is pending is rejected and leaves the
// first one pending.
TEST_P(VulkanQueueDeallocaTest, SecondDeallocaRejectedWhilePending) {
  ASSERT_NO_FATAL_FAILURE(QueueFirstDealloca());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        TryQueueDealloca(device_, buffer_));
  ExpectFirstDeallocaPending();
  ASSERT_NO_FATAL_FAILURE(ReleaseFirstDealloca());
}

// Regression: rejecting the second dealloca used to clear the flag set by the
// first, so a third dealloca was accepted and the buffer was queued for
// release twice.
TEST_P(VulkanQueueDeallocaTest, RejectedDeallocaDoesNotAdmitAnother) {
  ASSERT_NO_FATAL_FAILURE(QueueFirstDealloca());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        TryQueueDealloca(device_, buffer_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        TryQueueDealloca(device_, buffer_));
  ExpectFirstDeallocaPending();
  ASSERT_NO_FATAL_FAILURE(ReleaseFirstDealloca());
}

// Regression: a dealloca rejected for unsupported flags while another is
// pending used to clear the pending one's.
TEST_P(VulkanQueueDeallocaTest, InvalidFlagsDeallocaDoesNotAdmitAnother) {
  ASSERT_NO_FATAL_FAILURE(QueueFirstDealloca());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      TryQueueDealloca(device_, buffer_, /*flags=*/1ull << 62));
  EXPECT_TRUE(iree_hal_local_transient_buffer_is_dealloca_queued(buffer_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        TryQueueDealloca(device_, buffer_));
  ExpectFirstDeallocaPending();
  ASSERT_NO_FATAL_FAILURE(ReleaseFirstDealloca());
}

// The flag stays set after the dealloca executes, so deallocating an already
// released buffer is rejected.
TEST_P(VulkanQueueDeallocaTest, DeallocaAfterCompletionRejected) {
  ASSERT_NO_FATAL_FAILURE(QueueFirstDealloca());
  ASSERT_NO_FATAL_FAILURE(ReleaseFirstDealloca());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        TryQueueDealloca(device_, buffer_));
}

// A dealloca that sets the flag and then fails to submit must clear it so the
// caller can retry. Here the signal payload is out of range, which is rejected
// after the flag has been set.
TEST_P(VulkanQueueDeallocaTest, FailedSubmissionAllowsRetry) {
  SemaphoreList empty_wait;
  SemaphoreList bad_signal(device_, {0}, {IREE_HAL_SEMAPHORE_MAX_VALUE + 1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_device_queue_dealloca(
                            device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait,
                            bad_signal, buffer_, IREE_HAL_DEALLOCA_FLAG_NONE));
  EXPECT_FALSE(iree_hal_local_transient_buffer_is_dealloca_queued(buffer_));
  EXPECT_TRUE(iree_hal_local_transient_buffer_is_committed(buffer_));

  ASSERT_NO_FATAL_FAILURE(QueueFirstDealloca());
  ASSERT_NO_FATAL_FAILURE(ReleaseFirstDealloca());
}

CTS_REGISTER_TEST_SUITE(VulkanQueueDeallocaTest);

}  // namespace iree::hal::cts
