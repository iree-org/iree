// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_command_buffer.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/block_processor.h"
#include "iree/task/process.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_host_size_t kBlockSize = 4096;

// Test fixture for block command buffer recording + execution.
// Creates a heap allocator and block pool for recording, then executes the
// recorded command buffer through the single-worker processor path.
class BlockCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        iree_make_cstring_view("test"), iree_allocator_system(),
        iree_allocator_system(), &device_allocator_));

    iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(),
                                     &block_pool_);
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_hal_allocator_release(device_allocator_);
  }

  // Allocates a host-visible buffer with persistent mapping support.
  iree_hal_buffer_t* AllocateBuffer(iree_host_size_t size) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(device_allocator_, params,
                                                     size, &buffer));
    // Zero the buffer for deterministic results.
    IREE_CHECK_OK(iree_hal_buffer_map_zero(buffer, 0, IREE_HAL_WHOLE_BUFFER));
    return buffer;
  }

  // Creates a block command buffer, calls begin(), returns it ready for
  // recording.
  iree_hal_command_buffer_t* CreateCommandBuffer() {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_CHECK_OK(iree_hal_block_command_buffer_create(
        device_allocator_, /*scope=*/nullptr, /*executor=*/nullptr,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*queue_affinity=*/0, /*binding_capacity=*/0, &block_pool_,
        iree_allocator_system(), &command_buffer));
    IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));
    return command_buffer;
  }

  // Ends recording and executes the recorded commands synchronously via
  // the single-worker processor path. Returns any execution error.
  iree_status_t EndAndExecute(iree_hal_command_buffer_t* command_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));

    const iree_hal_cmd_block_recording_t* recording =
        iree_hal_block_command_buffer_recording(command_buffer);

    // Allocate a processor context and run single-worker.
    iree_hal_cmd_block_processor_context_t* context = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_processor_context_allocate(
        recording, /*binding_table=*/nullptr, /*binding_table_length=*/0,
        /*worker_count=*/1, iree_allocator_system(), &context));

    iree_hal_cmd_block_processor_worker_state_t worker_state;
    memset(&worker_state, 0, sizeof(worker_state));
    iree_hal_cmd_block_processor_drain_result_t result;
    iree_hal_cmd_block_processor_drain(context, 0, &worker_state, &result);

    iree_status_t status =
        iree_hal_cmd_block_processor_context_consume_result(context);
    iree_hal_cmd_block_processor_context_free(context, iree_allocator_system());
    return status;
  }

  // Reads buffer contents into a byte vector.
  std::vector<uint8_t> ReadBuffer(iree_hal_buffer_t* buffer,
                                  iree_device_size_t offset,
                                  iree_device_size_t length) {
    std::vector<uint8_t> data(length);
    iree_hal_buffer_mapping_t mapping = {{0}};
    IREE_CHECK_OK(iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        offset, length, &mapping));
    memcpy(data.data(), mapping.contents.data, length);
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
    return data;
  }

  iree_hal_allocator_t* device_allocator_ = nullptr;
  iree_arena_block_pool_t block_pool_;
};

//===----------------------------------------------------------------------===//
// Fill buffer
//===----------------------------------------------------------------------===//

TEST_F(BlockCommandBufferTest, FillBuffer1Byte) {
  iree_hal_buffer_t* buffer = AllocateBuffer(256);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint8_t pattern = 0xAB;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 256};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  auto data = ReadBuffer(buffer, 0, 256);
  for (size_t i = 0; i < 256; ++i) {
    ASSERT_EQ(data[i], 0xAB) << "byte " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, FillBuffer4Byte) {
  iree_hal_buffer_t* buffer = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint32_t pattern = 0xDEADBEEF;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  auto data = ReadBuffer(buffer, 0, 64);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0xDEADBEEF) << "at offset " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, FillBufferSubregion) {
  iree_hal_buffer_t* buffer = AllocateBuffer(128);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Fill bytes [32..96) with pattern, leaving [0..32) and [96..128) as zero.
  uint32_t pattern = 0x12345678;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 32, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  // Leading region should be zero.
  auto before = ReadBuffer(buffer, 0, 32);
  for (size_t i = 0; i < 32; ++i) {
    ASSERT_EQ(before[i], 0) << "byte " << i;
  }

  // Filled region.
  auto filled = ReadBuffer(buffer, 32, 64);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value;
    memcpy(&value, filled.data() + i, 4);
    ASSERT_EQ(value, 0x12345678) << "at offset " << (32 + i);
  }

  // Trailing region should be zero.
  auto after = ReadBuffer(buffer, 96, 32);
  for (size_t i = 0; i < 32; ++i) {
    ASSERT_EQ(after[i], 0) << "byte " << (96 + i);
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

//===----------------------------------------------------------------------===//
// Update buffer
//===----------------------------------------------------------------------===//

TEST_F(BlockCommandBufferTest, UpdateBuffer) {
  iree_hal_buffer_t* buffer = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Write known pattern to the device buffer.
  const uint8_t source_data[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                                 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C,
                                 0x0D, 0x0E, 0x0F, 0x10};
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 8, .length = sizeof(source_data)};
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      cb, source_data, /*source_offset=*/0, target_ref,
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  // Bytes [0..8) should be zero.
  auto before = ReadBuffer(buffer, 0, 8);
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(before[i], 0) << "byte " << i;
  }

  // Bytes [8..24) should be our source data.
  auto updated = ReadBuffer(buffer, 8, sizeof(source_data));
  for (size_t i = 0; i < sizeof(source_data); ++i) {
    ASSERT_EQ(updated[i], source_data[i]) << "byte " << (8 + i);
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, UpdateBufferWithSourceOffset) {
  iree_hal_buffer_t* buffer = AllocateBuffer(32);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  const uint8_t source_data[] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
  // Copy bytes [2..6) of source to buffer [0..4).
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 4};
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      cb, source_data, /*source_offset=*/2, target_ref,
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  auto data = ReadBuffer(buffer, 0, 4);
  ASSERT_EQ(data[0], 0xCC);
  ASSERT_EQ(data[1], 0xDD);
  ASSERT_EQ(data[2], 0xEE);
  ASSERT_EQ(data[3], 0xFF);

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

//===----------------------------------------------------------------------===//
// Copy buffer
//===----------------------------------------------------------------------===//

TEST_F(BlockCommandBufferTest, CopyBuffer) {
  iree_hal_buffer_t* source = AllocateBuffer(64);
  iree_hal_buffer_t* target = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Fill source with a known pattern.
  uint32_t pattern = 0xCAFEBABE;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(source, 0, 64, &pattern, sizeof(pattern)));

  // Copy source → target via command buffer.
  iree_hal_buffer_ref_t source_ref = {
      .buffer = source, .offset = 0, .length = 64};
  iree_hal_buffer_ref_t target_ref = {
      .buffer = target, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(cb, source_ref, target_ref,
                                                     IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  auto data = ReadBuffer(target, 0, 64);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0xCAFEBABE) << "at offset " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(source);
}

TEST_F(BlockCommandBufferTest, CopyBufferSubregion) {
  iree_hal_buffer_t* source = AllocateBuffer(128);
  iree_hal_buffer_t* target = AllocateBuffer(128);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint32_t pattern = 0x11223344;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(source, 0, 128, &pattern, sizeof(pattern)));

  // Copy source[32..64) → target[64..96).
  iree_hal_buffer_ref_t source_ref = {
      .buffer = source, .offset = 32, .length = 32};
  iree_hal_buffer_ref_t target_ref = {
      .buffer = target, .offset = 64, .length = 32};
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(cb, source_ref, target_ref,
                                                     IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(EndAndExecute(cb));

  // Target [0..64) should be zero.
  auto before = ReadBuffer(target, 0, 64);
  for (size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(before[i], 0) << "byte " << i;
  }

  // Target [64..96) should have our pattern.
  auto copied = ReadBuffer(target, 64, 32);
  for (size_t i = 0; i < 32; i += 4) {
    uint32_t value;
    memcpy(&value, copied.data() + i, 4);
    ASSERT_EQ(value, 0x11223344) << "at offset " << (64 + i);
  }

  // Target [96..128) should be zero.
  auto after = ReadBuffer(target, 96, 32);
  for (size_t i = 0; i < 32; ++i) {
    ASSERT_EQ(after[i], 0) << "byte " << (96 + i);
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(target);
  iree_hal_buffer_release(source);
}

//===----------------------------------------------------------------------===//
// Mixed operations with barriers
//===----------------------------------------------------------------------===//

TEST_F(BlockCommandBufferTest, FillThenCopyWithBarrier) {
  iree_hal_buffer_t* buffer_a = AllocateBuffer(64);
  iree_hal_buffer_t* buffer_b = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Fill buffer_a.
  uint32_t pattern = 0xAAAAAAAA;
  iree_hal_buffer_ref_t fill_ref = {
      .buffer = buffer_a, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, fill_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Barrier between fill and copy.
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      cb, IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  // Copy buffer_a → buffer_b.
  iree_hal_buffer_ref_t source_ref = {
      .buffer = buffer_a, .offset = 0, .length = 64};
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer_b, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(cb, source_ref, target_ref,
                                                     IREE_HAL_COPY_FLAG_NONE));

  IREE_ASSERT_OK(EndAndExecute(cb));

  // Both buffers should contain the pattern.
  auto data_a = ReadBuffer(buffer_a, 0, 64);
  auto data_b = ReadBuffer(buffer_b, 0, 64);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value_a, value_b;
    memcpy(&value_a, data_a.data() + i, 4);
    memcpy(&value_b, data_b.data() + i, 4);
    ASSERT_EQ(value_a, 0xAAAAAAAA) << "buffer_a at " << i;
    ASSERT_EQ(value_b, 0xAAAAAAAA) << "buffer_b at " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer_b);
  iree_hal_buffer_release(buffer_a);
}

TEST_F(BlockCommandBufferTest, MultipleFillsInOneRegion) {
  iree_hal_buffer_t* buffer = AllocateBuffer(128);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Two fills in the same region (no barrier between them).
  uint32_t pattern_a = 0x11111111;
  iree_hal_buffer_ref_t ref_a = {.buffer = buffer, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, ref_a, &pattern_a, sizeof(pattern_a), IREE_HAL_FILL_FLAG_NONE));

  uint32_t pattern_b = 0x22222222;
  iree_hal_buffer_ref_t ref_b = {.buffer = buffer, .offset = 64, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, ref_b, &pattern_b, sizeof(pattern_b), IREE_HAL_FILL_FLAG_NONE));

  IREE_ASSERT_OK(EndAndExecute(cb));

  auto data = ReadBuffer(buffer, 0, 128);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0x11111111) << "first half at " << i;
  }
  for (size_t i = 64; i < 128; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0x22222222) << "second half at " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, UpdateThenFillWithBarrier) {
  iree_hal_buffer_t* buffer = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Update first 8 bytes.
  const uint8_t source[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  iree_hal_buffer_ref_t update_ref = {
      .buffer = buffer, .offset = 0, .length = 8};
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      cb, source, 0, update_ref, IREE_HAL_UPDATE_FLAG_NONE));

  // Barrier.
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      cb, IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  // Fill remaining bytes [8..64).
  uint32_t pattern = 0xFF;
  iree_hal_buffer_ref_t fill_ref = {
      .buffer = buffer, .offset = 8, .length = 56};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(cb, fill_ref, &pattern, 1,
                                                     IREE_HAL_FILL_FLAG_NONE));

  IREE_ASSERT_OK(EndAndExecute(cb));

  auto data = ReadBuffer(buffer, 0, 64);
  // First 8 bytes: our update.
  for (size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(data[i], source[i]) << "byte " << i;
  }
  // Remaining 56 bytes: fill with 0xFF.
  for (size_t i = 8; i < 64; ++i) {
    ASSERT_EQ(data[i], 0xFF) << "byte " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

//===----------------------------------------------------------------------===//
// Empty command buffer
//===----------------------------------------------------------------------===//

TEST_F(BlockCommandBufferTest, EmptyCommandBuffer) {
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();
  // Recording with no commands should succeed.
  IREE_ASSERT_OK(EndAndExecute(cb));
  iree_hal_command_buffer_release(cb);
}

//===----------------------------------------------------------------------===//
// ISA check
//===----------------------------------------------------------------------===//

TEST_F(BlockCommandBufferTest, IsaCheck) {
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();
  EXPECT_TRUE(iree_hal_block_command_buffer_isa(cb));
  iree_hal_command_buffer_release(cb);
}

//===----------------------------------------------------------------------===//
// Issue path: process-based execution
//===----------------------------------------------------------------------===//
// Tests the issue() path that creates a process wrapping the block processor.
// These tests drive the process manually (calling drain directly) without an
// executor, verifying the process→processor integration.

// Drains a process completely using a single worker. Calls drain() in a loop
// until completed, then triggers process completion and returns the error.
static iree_status_t DrainProcessSingleWorker(
    iree_hal_block_issue_context_t* issue_context) {
  iree_task_process_t* process = &issue_context->process;

  // Drain loop.
  iree_task_process_drain_result_t result;
  do {
    memset(&result, 0, sizeof(result));
    iree_status_t status = process->drain(process, 0, &result);
    if (!iree_status_is_ok(status)) {
      iree_task_process_report_error(process, status);
      break;
    }
  } while (!result.completed);

  // Snapshot release_fn before completion — completion_fn may invalidate the
  // process if release_fn is NULL (single-phase lifecycle).
  iree_task_process_release_fn_t release_fn = process->release_fn;

  // Complete the process (triggers eager completion callback).
  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(process, &activated_head, &activated_tail);

  // Call the deferred release callback (frees processor context). In the
  // executor this is called when the last active drainer exits; here we
  // are the only drainer and have already exited.
  if (release_fn) {
    release_fn(process);
  }

  return iree_ok_status();
}

// Multi-worker drain: spawns N threads that drain the process concurrently.
struct IssueWorkerArgs {
  iree_hal_block_issue_context_t* issue_context;
  uint32_t worker_index;
};

static int issue_worker_thread_entry(void* arg) {
  IssueWorkerArgs* worker_args = reinterpret_cast<IssueWorkerArgs*>(arg);
  iree_task_process_t* process = &worker_args->issue_context->process;

  iree_task_process_drain_result_t result;
  do {
    memset(&result, 0, sizeof(result));
    iree_status_t status =
        process->drain(process, worker_args->worker_index, &result);
    if (!iree_status_is_ok(status)) {
      iree_task_process_report_error(process, status);
      break;
    }
    if (!result.completed && !result.did_work) {
      iree_thread_yield();
    }
  } while (!result.completed);
  return 0;
}

static void DrainProcessMultiWorker(
    iree_hal_block_issue_context_t* issue_context, uint32_t worker_count) {
  std::vector<iree_thread_t*> threads(worker_count);
  std::vector<IssueWorkerArgs> args(worker_count);

  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_make_cstring_view("test-worker");

  for (uint32_t i = 0; i < worker_count; ++i) {
    args[i].issue_context = issue_context;
    args[i].worker_index = i;
    IREE_CHECK_OK(iree_thread_create(issue_worker_thread_entry, &args[i],
                                     params, iree_allocator_system(),
                                     &threads[i]));
  }

  for (uint32_t i = 0; i < worker_count; ++i) {
    iree_thread_release(threads[i]);
  }

  // All threads have exited. Snapshot release_fn before completion.
  iree_task_process_release_fn_t release_fn = issue_context->process.release_fn;

  // Complete the process (triggers eager completion callback).
  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&issue_context->process, &activated_head,
                             &activated_tail);

  // Call the deferred release callback (frees processor context).
  if (release_fn) {
    release_fn(&issue_context->process);
  }
}

TEST_F(BlockCommandBufferTest, IssueFillSingleWorker) {
  iree_hal_buffer_t* buffer = AllocateBuffer(256);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint32_t pattern = 0xDEADBEEF;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 256};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cb));

  // Issue: create a process wrapping the block processor.
  iree_hal_block_issue_context_t issue_context;
  IREE_ASSERT_OK(iree_hal_block_command_buffer_issue(
      cb, /*binding_table=*/nullptr, /*worker_count=*/1,
      iree_allocator_system(), &issue_context));

  // Drain the process with a single worker.
  DrainProcessSingleWorker(&issue_context);

  // Verify the fill executed.
  auto data = ReadBuffer(buffer, 0, 256);
  for (size_t i = 0; i < 256; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0xDEADBEEF) << "at offset " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, IssueFillCopyWithBarrier) {
  iree_hal_buffer_t* buffer_a = AllocateBuffer(64);
  iree_hal_buffer_t* buffer_b = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  // Fill buffer_a.
  uint32_t pattern = 0xCAFEBABE;
  iree_hal_buffer_ref_t fill_ref = {
      .buffer = buffer_a, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, fill_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Barrier.
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      cb, IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  // Copy buffer_a → buffer_b.
  iree_hal_buffer_ref_t source_ref = {
      .buffer = buffer_a, .offset = 0, .length = 64};
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer_b, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(cb, source_ref, target_ref,
                                                     IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cb));

  iree_hal_block_issue_context_t issue_context;
  IREE_ASSERT_OK(iree_hal_block_command_buffer_issue(
      cb, nullptr, 1, iree_allocator_system(), &issue_context));

  DrainProcessSingleWorker(&issue_context);

  // Both buffers should contain the pattern.
  auto data_a = ReadBuffer(buffer_a, 0, 64);
  auto data_b = ReadBuffer(buffer_b, 0, 64);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value_a, value_b;
    memcpy(&value_a, data_a.data() + i, 4);
    memcpy(&value_b, data_b.data() + i, 4);
    ASSERT_EQ(value_a, 0xCAFEBABE) << "buffer_a at " << i;
    ASSERT_EQ(value_b, 0xCAFEBABE) << "buffer_b at " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer_b);
  iree_hal_buffer_release(buffer_a);
}

TEST_F(BlockCommandBufferTest, IssueEmptyCommandBuffer) {
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cb));

  iree_hal_block_issue_context_t issue_context;
  IREE_ASSERT_OK(iree_hal_block_command_buffer_issue(
      cb, nullptr, 1, iree_allocator_system(), &issue_context));

  // The process should complete immediately (NULL processor context).
  iree_task_process_drain_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      issue_context.process.drain(&issue_context.process, 0, &result);
  IREE_EXPECT_OK(status);
  EXPECT_TRUE(result.completed);

  // Snapshot release_fn before completion.
  iree_task_process_release_fn_t release_fn = issue_context.process.release_fn;

  iree_task_process_t* activated_head = nullptr;
  iree_task_process_t* activated_tail = nullptr;
  iree_task_process_complete(&issue_context.process, &activated_head,
                             &activated_tail);

  // Deferred release (frees processor context — NULL for empty recordings,
  // but we still call it for protocol correctness).
  if (release_fn) {
    release_fn(&issue_context.process);
  }

  iree_hal_command_buffer_release(cb);
}

TEST_F(BlockCommandBufferTest, IssueFillMultiWorker) {
  iree_hal_buffer_t* buffer = AllocateBuffer(256);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint32_t pattern = 0xBAADF00D;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 256};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cb));

  const uint32_t worker_count = 4;
  iree_hal_block_issue_context_t issue_context;
  IREE_ASSERT_OK(iree_hal_block_command_buffer_issue(
      cb, nullptr, worker_count, iree_allocator_system(), &issue_context));

  DrainProcessMultiWorker(&issue_context, worker_count);

  auto data = ReadBuffer(buffer, 0, 256);
  for (size_t i = 0; i < 256; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0xBAADF00D) << "at offset " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, IssueDeinitWithoutDrain) {
  iree_hal_buffer_t* buffer = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint32_t pattern = 0x11111111;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cb));

  iree_hal_block_issue_context_t issue_context;
  IREE_ASSERT_OK(iree_hal_block_command_buffer_issue(
      cb, nullptr, 1, iree_allocator_system(), &issue_context));

  // Deinit without draining (abnormal cleanup path).
  iree_hal_block_issue_context_deinitialize(&issue_context);

  // Buffer should be unchanged (fill was never executed).
  auto data = ReadBuffer(buffer, 0, 64);
  for (size_t i = 0; i < 64; ++i) {
    ASSERT_EQ(data[i], 0) << "byte " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

TEST_F(BlockCommandBufferTest, IssueCompletionCallbackInvoked) {
  iree_hal_buffer_t* buffer = AllocateBuffer(64);
  iree_hal_command_buffer_t* cb = CreateCommandBuffer();

  uint32_t pattern = 0x55555555;
  iree_hal_buffer_ref_t target_ref = {
      .buffer = buffer, .offset = 0, .length = 64};
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      cb, target_ref, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cb));

  iree_hal_block_issue_context_t issue_context;
  IREE_ASSERT_OK(iree_hal_block_command_buffer_issue(
      cb, nullptr, 1, iree_allocator_system(), &issue_context));

  // Set a user completion callback that tracks invocation.
  static bool completion_called = false;
  static iree_status_t completion_status = iree_ok_status();
  completion_called = false;
  completion_status = iree_ok_status();
  issue_context.user_completion_fn = [](iree_task_process_t* process,
                                        iree_status_t status) {
    completion_called = true;
    completion_status = status;
  };

  DrainProcessSingleWorker(&issue_context);

  EXPECT_TRUE(completion_called);
  IREE_EXPECT_OK(completion_status);

  auto data = ReadBuffer(buffer, 0, 64);
  for (size_t i = 0; i < 64; i += 4) {
    uint32_t value;
    memcpy(&value, data.data() + i, 4);
    ASSERT_EQ(value, 0x55555555) << "at offset " << i;
  }

  iree_hal_command_buffer_release(cb);
  iree_hal_buffer_release(buffer);
}

}  // namespace
