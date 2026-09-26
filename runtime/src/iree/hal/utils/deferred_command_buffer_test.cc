// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/deferred_command_buffer.h"

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace {

struct FakeBuffer {
  iree_hal_buffer_t base;
  int* destroy_count;
};

static void IREE_API_PTR DestroyBuffer(iree_hal_buffer_t* base_buffer) {
  auto* buffer = reinterpret_cast<FakeBuffer*>(base_buffer);
  ++*buffer->destroy_count;
  delete buffer;
}

static const iree_hal_buffer_vtable_t kBufferVTable = {
    /*.recycle=*/iree_hal_buffer_recycle,
    /*.destroy=*/DestroyBuffer,
};

struct FakeTargetCommandBuffer {
  iree_hal_command_buffer_t base;
  int flush_count = 0;
  int invalidate_count = 0;
  iree_hal_buffer_ref_t flush_ref = {};
  iree_hal_buffer_ref_t invalidate_ref = {};
  iree_status_code_t flush_status = IREE_STATUS_OK;
  iree_status_code_t invalidate_status = IREE_STATUS_OK;
};

static void IREE_API_PTR DestroyTarget(iree_hal_command_buffer_t* base) {
  delete reinterpret_cast<FakeTargetCommandBuffer*>(base);
}

static iree_status_t IREE_API_PTR
BeginOrEndTarget(iree_hal_command_buffer_t* /*base*/) {
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR Flush(iree_hal_command_buffer_t* base,
                                        iree_hal_buffer_ref_t ref) {
  auto* target = reinterpret_cast<FakeTargetCommandBuffer*>(base);
  ++target->flush_count;
  target->flush_ref = ref;
  return iree_status_from_code(target->flush_status);
}

static iree_status_t IREE_API_PTR Invalidate(iree_hal_command_buffer_t* base,
                                             iree_hal_buffer_ref_t ref) {
  auto* target = reinterpret_cast<FakeTargetCommandBuffer*>(base);
  ++target->invalidate_count;
  target->invalidate_ref = ref;
  return iree_status_from_code(target->invalidate_status);
}

class DeferredCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &block_pool_);
    target_vtable_.destroy = DestroyTarget;
    target_vtable_.begin = BeginOrEndTarget;
    target_vtable_.end = BeginOrEndTarget;
    target_vtable_.flush_buffer = Flush;
    target_vtable_.invalidate_buffer = Invalidate;
    target_ = new FakeTargetCommandBuffer{};
    iree_hal_command_buffer_initialize(
        /*device_allocator=*/nullptr,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, /*validation_state=*/nullptr, &target_vtable_,
        &target_->base);

    auto* buffer = new FakeBuffer{};
    buffer->destroy_count = &buffer_destroy_count_;
    iree_hal_buffer_initialize(
        /*placement=*/{}, /*allocated_buffer=*/&buffer->base,
        /*allocation_size=*/4096, /*byte_offset=*/0, /*byte_length=*/4096,
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_USAGE_TRANSFER, &kBufferVTable, &buffer->base);
    buffer_ = &buffer->base;

    IREE_ASSERT_OK(iree_hal_deferred_command_buffer_create(
        /*device_allocator=*/nullptr,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/4, &block_pool_, iree_allocator_system(),
        &command_buffer_));
  }

  void TearDown() override {
    iree_hal_command_buffer_release(command_buffer_);
    iree_hal_command_buffer_release(&target_->base);
    iree_hal_buffer_release(buffer_);
    EXPECT_EQ(buffer_destroy_count_, 1);
    // The pool must outlive the deferred command buffer and its resource set.
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using RecordFn = iree_status_t (*)(iree_hal_command_buffer_t*,
                                     iree_hal_buffer_ref_t);

  iree_status_t Record(RecordFn record, iree_hal_buffer_ref_t ref) {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer_));
    IREE_RETURN_IF_ERROR(record(command_buffer_, ref));
    ExpectCounts(0, 0);
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer_));
    ExpectCounts(0, 0);
    return iree_ok_status();
  }

  iree_status_t Apply(iree_hal_buffer_binding_table_t bindings =
                          iree_hal_buffer_binding_table_empty()) {
    return iree_hal_deferred_command_buffer_apply(command_buffer_,
                                                  &target_->base, bindings);
  }

  void ExpectCounts(int flush, int invalidate) {
    EXPECT_EQ(target_->flush_count, flush);
    EXPECT_EQ(target_->invalidate_count, invalidate);
  }

  static void ExpectRef(iree_hal_buffer_ref_t actual, iree_hal_buffer_t* buffer,
                        iree_device_size_t offset, iree_device_size_t length) {
    EXPECT_EQ(actual.buffer, buffer);
    EXPECT_EQ(actual.offset, offset);
    EXPECT_EQ(actual.length, length);
    EXPECT_EQ(static_cast<uint32_t>(actual.buffer_slot), 0u);
    EXPECT_EQ(static_cast<uint32_t>(actual.reserved), 0u);
  }

  void CheckRetention(RecordFn record, bool flush) {
    auto* recorded_buffer = buffer_;
    IREE_ASSERT_OK(
        Record(record, iree_hal_make_buffer_ref(recorded_buffer, 137, 251)));
    EXPECT_EQ(buffer_destroy_count_, 0);
    iree_hal_buffer_release(buffer_);
    buffer_ = nullptr;
    ASSERT_EQ(buffer_destroy_count_, 0);

    IREE_ASSERT_OK(Apply());
    ExpectCounts(flush ? 1 : 0, flush ? 0 : 1);
    ASSERT_EQ(buffer_destroy_count_, 0);
    ExpectRef(flush ? target_->flush_ref : target_->invalidate_ref,
              recorded_buffer, 137, 251);

    // Even one-shot apply keeps direct resources until deferred destruction.
    iree_hal_command_buffer_release(command_buffer_);
    command_buffer_ = nullptr;
    EXPECT_EQ(buffer_destroy_count_, 1);
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_hal_command_buffer_vtable_t target_vtable_ = {};
  FakeTargetCommandBuffer* target_ = nullptr;
  iree_hal_command_buffer_t* command_buffer_ = nullptr;
  int buffer_destroy_count_ = 0;
  iree_hal_buffer_t* buffer_ = nullptr;
};

TEST_F(DeferredCommandBufferTest, DirectFlushExecutesOnlyDuringApply) {
  // Also verifies that recording and replay preserve the exact direct range.
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_flush_buffer,
                        iree_hal_make_buffer_ref(buffer_, 137, 251)));
  IREE_ASSERT_OK(Apply());
  ExpectCounts(1, 0);
  ExpectRef(target_->flush_ref, buffer_, 137, 251);
}

TEST_F(DeferredCommandBufferTest, DirectInvalidateExecutesOnlyDuringApply) {
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_invalidate_buffer,
                        iree_hal_make_buffer_ref(buffer_, 193, 347)));
  IREE_ASSERT_OK(Apply());
  ExpectCounts(0, 1);
  ExpectRef(target_->invalidate_ref, buffer_, 193, 347);
}

TEST_F(DeferredCommandBufferTest, IndirectFlushResolvesBindingAtApply) {
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_flush_buffer,
                        iree_hal_make_indirect_buffer_ref(3, 37, 59)));
  // The binding is supplied only after recording has completed.
  iree_hal_buffer_binding_t bindings[4] = {};
  bindings[3] = {buffer_, 101, 503};
  IREE_ASSERT_OK(Apply({4, bindings}));
  ExpectCounts(1, 0);
  ExpectRef(target_->flush_ref, buffer_, 138, 59);
}

TEST_F(DeferredCommandBufferTest, IndirectInvalidateResolvesBindingAtApply) {
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_invalidate_buffer,
                        iree_hal_make_indirect_buffer_ref(3, 41, 61)));
  iree_hal_buffer_binding_t bindings[4] = {};
  bindings[3] = {buffer_, 103, 509};
  IREE_ASSERT_OK(Apply({4, bindings}));
  ExpectCounts(0, 1);
  ExpectRef(target_->invalidate_ref, buffer_, 144, 61);
}

TEST_F(DeferredCommandBufferTest, MissingIndirectFlushBindingFailsDuringApply) {
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_flush_buffer,
                        iree_hal_make_indirect_buffer_ref(3, 37, 59)));
  // An empty table resolves to a null ref successfully. A nonempty table with
  // an out-of-bounds slot makes resolve_ref return OUT_OF_RANGE.
  iree_hal_buffer_binding_t binding = {buffer_, 101, 503};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Apply({1, &binding}));
  ExpectCounts(0, 0);
}

TEST_F(DeferredCommandBufferTest,
       MissingIndirectInvalidateBindingFailsDuringApply) {
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_invalidate_buffer,
                        iree_hal_make_indirect_buffer_ref(3, 41, 61)));
  iree_hal_buffer_binding_t binding = {buffer_, 103, 509};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Apply({1, &binding}));
  ExpectCounts(0, 0);
}

TEST_F(DeferredCommandBufferTest, FlushTargetErrorPropagatesFromApply) {
  target_->flush_status = IREE_STATUS_ABORTED;
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_flush_buffer,
                        iree_hal_make_buffer_ref(buffer_, 137, 251)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, Apply());
  ExpectCounts(1, 0);
  ExpectRef(target_->flush_ref, buffer_, 137, 251);
}

TEST_F(DeferredCommandBufferTest, InvalidateTargetErrorPropagatesFromApply) {
  target_->invalidate_status = IREE_STATUS_DATA_LOSS;
  IREE_ASSERT_OK(Record(iree_hal_command_buffer_invalidate_buffer,
                        iree_hal_make_buffer_ref(buffer_, 193, 347)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Apply());
  ExpectCounts(0, 1);
  ExpectRef(target_->invalidate_ref, buffer_, 193, 347);
}

TEST_F(DeferredCommandBufferTest,
       DirectFlushBufferRetainedUntilDeferredDestruction) {
  CheckRetention(iree_hal_command_buffer_flush_buffer, /*flush=*/true);
}

TEST_F(DeferredCommandBufferTest,
       DirectInvalidateBufferRetainedUntilDeferredDestruction) {
  CheckRetention(iree_hal_command_buffer_invalidate_buffer, /*flush=*/false);
}

}  // namespace
}  // namespace hal
}  // namespace iree
