// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct FakeCommandBuffer {
  iree_hal_command_buffer_t base;
  int flush_count = 0;
  int invalidate_count = 0;
  iree_hal_buffer_ref_t flush_ref = {};
  iree_hal_buffer_ref_t invalidate_ref = {};
  iree_status_code_t flush_status = IREE_STATUS_OK;
  iree_status_code_t invalidate_status = IREE_STATUS_OK;
};

static void IREE_API_PTR Destroy(iree_hal_command_buffer_t* command_buffer) {
  // The fixture owns the fake; only the validation storage is heap allocated.
  iree_allocator_free(iree_allocator_system(), command_buffer->validation_state);
}

static iree_status_t IREE_API_PTR BeginOrEnd(
    iree_hal_command_buffer_t* /*command_buffer*/) {
  return iree_ok_status();
}

static iree_status_t IREE_API_PTR Flush(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t target_ref) {
  auto* fake = reinterpret_cast<FakeCommandBuffer*>(command_buffer);
  ++fake->flush_count;
  fake->flush_ref = target_ref;
  return iree_status_from_code(fake->flush_status);
}

static iree_status_t IREE_API_PTR Invalidate(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t target_ref) {
  auto* fake = reinterpret_cast<FakeCommandBuffer*>(command_buffer);
  ++fake->invalidate_count;
  fake->invalidate_ref = target_ref;
  return iree_status_from_code(fake->invalidate_status);
}

class CommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    vtable_.destroy = Destroy;
    vtable_.begin = BeginOrEnd;
    vtable_.end = BeginOrEnd;
    vtable_.flush_buffer = Flush;
    vtable_.invalidate_buffer = Invalidate;
  }

  void TearDown() override {
    if (initialized_) iree_hal_command_buffer_release(&fake_.base);
  }

  iree_status_t Initialize(
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      iree_hal_command_category_t categories =
          IREE_HAL_COMMAND_CATEGORY_TRANSFER) {
    void* validation_state = nullptr;
    iree_host_size_t validation_size =
        iree_hal_command_buffer_validation_state_size(
            mode, /*binding_capacity=*/0);
    if (validation_size) {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          iree_allocator_system(), validation_size, &validation_state));
    }
    iree_hal_command_buffer_initialize(
        /*device_allocator=*/nullptr, mode, categories,
        IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/0, validation_state,
        &vtable_, &fake_.base);
    initialized_ = true;
    return iree_ok_status();
  }

  static void ExpectBufferRefEq(iree_hal_buffer_ref_t expected,
                               iree_hal_buffer_ref_t actual) {
    EXPECT_EQ(static_cast<uint32_t>(expected.reserved),
              static_cast<uint32_t>(actual.reserved));
    EXPECT_EQ(static_cast<uint32_t>(expected.buffer_slot),
              static_cast<uint32_t>(actual.buffer_slot));
    EXPECT_EQ(expected.buffer, actual.buffer);
    EXPECT_EQ(expected.offset, actual.offset);
    EXPECT_EQ(expected.length, actual.length);
  }

  FakeCommandBuffer fake_ = {};
  iree_hal_command_buffer_vtable_t vtable_ = {};
  bool initialized_ = false;
};

TEST_F(CommandBufferTest, FlushDispatchesExactBufferRef) {
  IREE_ASSERT_OK(Initialize());
  // Unvalidated dispatch only: the buffer is an identity token, never accessed.
  iree_hal_buffer_t buffer = {};
  auto ref = iree_hal_make_buffer_ref(&buffer, 37, 59);
  IREE_ASSERT_OK(iree_hal_command_buffer_flush_buffer(&fake_.base, ref));
  EXPECT_EQ(fake_.flush_count, 1);
  EXPECT_EQ(fake_.invalidate_count, 0);
  ExpectBufferRefEq(ref, fake_.flush_ref);
}

TEST_F(CommandBufferTest, InvalidateDispatchesExactBufferRef) {
  IREE_ASSERT_OK(Initialize());
  // Unvalidated dispatch only: the buffer is an identity token, never accessed.
  iree_hal_buffer_t buffer = {};
  auto ref = iree_hal_make_buffer_ref(&buffer, 41, 61);
  IREE_ASSERT_OK(iree_hal_command_buffer_invalidate_buffer(&fake_.base, ref));
  EXPECT_EQ(fake_.invalidate_count, 1);
  EXPECT_EQ(fake_.flush_count, 0);
  ExpectBufferRefEq(ref, fake_.invalidate_ref);
}

TEST_F(CommandBufferTest, FlushPropagatesCallbackError) {
  IREE_ASSERT_OK(Initialize());
  fake_.flush_status = IREE_STATUS_ABORTED;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_hal_command_buffer_flush_buffer(
          &fake_.base, iree_hal_make_indirect_buffer_ref(3, 17, 29)));
  EXPECT_EQ(fake_.flush_count, 1);
}

TEST_F(CommandBufferTest, InvalidatePropagatesCallbackError) {
  IREE_ASSERT_OK(Initialize());
  fake_.invalidate_status = IREE_STATUS_DATA_LOSS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_command_buffer_invalidate_buffer(
          &fake_.base, iree_hal_make_indirect_buffer_ref(3, 17, 29)));
  EXPECT_EQ(fake_.invalidate_count, 1);
}

TEST_F(CommandBufferTest, ZeroLengthFlushIsNoOp) {
  IREE_ASSERT_OK(Initialize());
  fake_.flush_status = IREE_STATUS_ABORTED;
  IREE_EXPECT_OK(iree_hal_command_buffer_flush_buffer(
      &fake_.base, iree_hal_make_indirect_buffer_ref(3, 17, 0)));
  EXPECT_EQ(fake_.flush_count, 0);
}

TEST_F(CommandBufferTest, ZeroLengthInvalidateIsNoOp) {
  IREE_ASSERT_OK(Initialize());
  fake_.invalidate_status = IREE_STATUS_DATA_LOSS;
  IREE_EXPECT_OK(iree_hal_command_buffer_invalidate_buffer(
      &fake_.base, iree_hal_make_indirect_buffer_ref(3, 17, 0)));
  EXPECT_EQ(fake_.invalidate_count, 0);
}

TEST_F(CommandBufferTest, FlushWithoutBackendCallbackIsUnimplemented) {
  vtable_.flush_buffer = nullptr;
  IREE_ASSERT_OK(Initialize());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_command_buffer_flush_buffer(
          &fake_.base, iree_hal_make_indirect_buffer_ref(3, 17, 29)));
  EXPECT_EQ(fake_.flush_count, 0);
}

TEST_F(CommandBufferTest, InvalidateWithoutBackendCallbackIsUnimplemented) {
  vtable_.invalidate_buffer = nullptr;
  IREE_ASSERT_OK(Initialize());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_command_buffer_invalidate_buffer(
          &fake_.base, iree_hal_make_indirect_buffer_ref(3, 17, 29)));
  EXPECT_EQ(fake_.invalidate_count, 0);
}

TEST_F(CommandBufferTest, ValidationRejectsUnsupportedCommandCategory) {
  if (iree_hal_command_buffer_validation_state_size(
          /*mode=*/0, /*binding_capacity=*/0) == 0) {
    GTEST_SKIP() << "Command buffer validation is compiled out";
  }
  IREE_ASSERT_OK(Initialize(/*mode=*/0, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&fake_.base));
  auto ref = iree_hal_make_indirect_buffer_ref(0, 17, 29);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_command_buffer_flush_buffer(&fake_.base, ref));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_command_buffer_invalidate_buffer(&fake_.base, ref));
  EXPECT_EQ(fake_.flush_count, 0);
  EXPECT_EQ(fake_.invalidate_count, 0);
  IREE_EXPECT_OK(iree_hal_command_buffer_end(&fake_.base));
}

}  // namespace
}  // namespace iree
