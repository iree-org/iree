// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan-specific queue API tests

#include <cstdint>
#include <vector>

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

class VulkanQueueCacheMaintenanceTest : public CtsTestBase<> {
 protected:
  iree_status_t AllocateCoherentStagingBuffer(
      iree_device_size_t size, iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
    return iree_hal_allocator_allocate_buffer(device_allocator_, params, size,
                                               out_buffer);
  }

  int64_t QueryNativeReplayCache(iree_string_view_t key) {
    int64_t value = 0;
    IREE_EXPECT_OK(iree_hal_device_query_i64(
        device_, IREE_SV("vulkan.queue.native_replay_cache"), key, &value));
    return value;
  }
};

TEST_P(VulkanQueueCacheMaintenanceTest, ProducerWaitOrdersCoherentTransfer) {
  constexpr iree_device_size_t kSize = 16;
  const uint32_t source_data[4] = {11, 22, 33, 44};
  Ref<iree_hal_buffer_t> source;
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(AllocateCoherentStagingBuffer(kSize, source.out()));
  IREE_ASSERT_OK(AllocateCoherentStagingBuffer(kSize, target.out()));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(source, 0, source_data, sizeof(source_data)));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t source_ref =
      iree_hal_make_buffer_ref(source, 0, kSize);
  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_buffer_ref(target, 0, kSize);
  IREE_ASSERT_OK(
      iree_hal_command_buffer_invalidate_buffer(command_buffer, source_ref));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, source_ref, target_ref, IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_command_buffer_flush_buffer(command_buffer, target_ref));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList producer(device_, {0}, {1});
  SemaphoreList completion(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, producer, completion,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(completion));
  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(producer, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      completion, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(ReadBufferData<uint32_t>(target),
            (std::vector<uint32_t>{11, 22, 33, 44}));
}

TEST_P(VulkanQueueCacheMaintenanceTest, ReplaysIndirectMaintenanceWithNewBinding) {
  if (QueryNativeReplayCache(IREE_SV("max_instance_count")) == 0) {
    GTEST_SKIP() << "Vulkan native replay cache is disabled";
  }
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  if (QueryNativeReplayCache(IREE_SV("instance_count")) >=
      QueryNativeReplayCache(IREE_SV("max_instance_count"))) {
    GTEST_SKIP() << "Vulkan native replay cache is full";
  }
  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));

  constexpr iree_device_size_t kSize = 16;
  constexpr uint32_t kPattern = 0x12345678;
  const uint32_t zeros[4] = {0, 0, 0, 0};
  Ref<iree_hal_buffer_t> buffer_a;
  Ref<iree_hal_buffer_t> buffer_b;
  IREE_ASSERT_OK(AllocateCoherentStagingBuffer(kSize, buffer_a.out()));
  IREE_ASSERT_OK(AllocateCoherentStagingBuffer(kSize, buffer_b.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer_a, 0, zeros, sizeof(zeros)));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer_b, 0, zeros, sizeof(zeros)));

  constexpr iree_hal_queue_affinity_t kQueueAffinity = 1ull;
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, kQueueAffinity,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/4,
                                        /*length=*/8);
  IREE_ASSERT_OK(
      iree_hal_command_buffer_invalidate_buffer(command_buffer, target_ref));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, target_ref, &kPattern, sizeof(kPattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_command_buffer_flush_buffer(command_buffer, target_ref));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  for (int i = 0; i < 2; ++i) {
    iree_hal_buffer_t* buffer = i == 0 ? buffer_a.get() : buffer_b.get();
    iree_hal_buffer_binding_t binding = {buffer, 0, kSize};
    iree_hal_buffer_binding_table_t binding_table = {1, &binding};
    SemaphoreList completion(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, kQueueAffinity, iree_hal_semaphore_list_empty(), completion,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        completion, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(ReadBufferData<uint32_t>(buffer),
              (std::vector<uint32_t>{0, kPattern, kPattern, 0}));
  }
  EXPECT_EQ(initial_hit_count, QueryNativeReplayCache(IREE_SV("hit_count")));
  EXPECT_EQ(initial_create_count,
            QueryNativeReplayCache(IREE_SV("create_count")));
}

TEST_P(VulkanQueueCacheMaintenanceTest, ReplayFailureFailsCompletionSemaphore) {
  constexpr iree_device_size_t kSize = 16;
  Ref<iree_hal_buffer_t> coherent_buffer;
  IREE_ASSERT_OK(
      AllocateCoherentStagingBuffer(kSize, coherent_buffer.out()));

  iree_hal_buffer_params_t candidate_params = {0};
  candidate_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  candidate_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  candidate_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  Ref<iree_hal_buffer_t> candidate;
  iree_status_t allocation_status = iree_hal_allocator_allocate_buffer(
      device_allocator_, candidate_params, kSize, candidate.out());
  if (!iree_status_is_ok(allocation_status)) {
    iree_status_ignore(allocation_status);
  }
  const bool has_noncoherent_buffer =
      candidate.get() != nullptr &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(candidate),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT);

  // HAL validates short bindings before queue submission. For the range
  // fallback, let native replay report the error so semaphore propagation is
  // exercised on devices with only coherent memory.
  const iree_hal_command_buffer_mode_t mode =
      has_noncoherent_buffer
          ? IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT |
                IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, mode,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_invalidate_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/4,
                                        /*length=*/8)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_buffer_binding_t binding = {
      has_noncoherent_buffer ? candidate.get() : coherent_buffer.get(), 0,
      has_noncoherent_buffer ? kSize : 8};
  iree_hal_buffer_binding_table_t binding_table = {1, &binding};
  const iree_status_code_t expected_status =
      has_noncoherent_buffer ? IREE_STATUS_UNIMPLEMENTED
                             : IREE_STATUS_OUT_OF_RANGE;
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList completion(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, gate, completion, command_buffer,
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  EXPECT_FALSE(iree_hal_semaphore_list_poll(completion));
  IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, /*frontier=*/NULL));
  IREE_EXPECT_STATUS_IS(
      expected_status,
      iree_hal_semaphore_list_wait(completion, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(expected_status,
                        iree_hal_semaphore_query(completion.semaphores[0],
                                                 &value));
  EXPECT_GE(value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
}

CTS_REGISTER_TEST_SUITE(VulkanQueueCacheMaintenanceTest);

}  // namespace iree::hal::cts
