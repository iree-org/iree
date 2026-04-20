// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_processor.h"

#include <cstring>
#include <vector>

#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/local_task/block_builder.h"
#include "iree/hal/drivers/local_task/block_isa.h"
#include "iree/hal/local/local_executable.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Worker thread entry
//===----------------------------------------------------------------------===//
// Thread entry point for multi-worker execution. Each thread receives a
// WorkerArgs pointing to the shared context and its worker index. Calls
// drain() in a loop until the recording completes.

struct WorkerArgs {
  // Shared processor context drained by all test workers.
  iree_hal_cmd_block_processor_context_t* context;
  // Aggregate tile count used by tests that inspect scheduling shape.
  iree_atomic_int64_t* total_tiles;
  // Worker index passed through to the processor.
  uint32_t worker_index;
};

static int worker_thread_entry(void* arg) {
  WorkerArgs* worker_args = reinterpret_cast<WorkerArgs*>(arg);
  // Drain loop.
  iree_hal_cmd_block_processor_worker_state_t worker_state;
  memset(&worker_state, 0, sizeof(worker_state));
  iree_hal_cmd_block_processor_drain_result_t result;
  do {
    iree_hal_cmd_block_processor_drain(worker_args->context,
                                       worker_args->worker_index, &worker_state,
                                       &result);
    if (result.tiles_executed != 0) {
      iree_atomic_fetch_add(worker_args->total_tiles,
                            (int64_t)result.tiles_executed,
                            iree_memory_order_relaxed);
    }
    if (!result.completed && result.tiles_executed == 0) {
      iree_thread_yield();
    }
  } while (!result.completed);
  return 0;
}

//===----------------------------------------------------------------------===//
// Test kernels
//===----------------------------------------------------------------------===//
// C functions matching iree_hal_executable_dispatch_v0_t. Each kernel
// exercises a specific aspect of the dispatch ABI.

// Atomically increments a uint32_t counter (binding 0) once per tile.
// Uses atomic_fetch_add so it's safe for multi-worker execution.
static int kernel_count_tiles(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  iree_atomic_int32_t* counter =
      reinterpret_cast<iree_atomic_int32_t*>(dispatch_state->binding_ptrs[0]);
  iree_atomic_fetch_add(counter, 1, iree_memory_order_relaxed);
  return 0;
}

// Writes workgroup_id_x into binding[0][workgroup_id_x].
// Safe for multi-worker: each tile writes to a unique index.
static int kernel_write_tile_id(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  uint32_t* output =
      reinterpret_cast<uint32_t*>(dispatch_state->binding_ptrs[0]);
  output[workgroup_state->workgroup_id_x] = workgroup_state->workgroup_id_x;
  return 0;
}

// Sums all push constants and atomically adds the result to binding[0][0]
// per tile.
static int kernel_sum_constants(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  uint32_t sum = 0;
  for (uint16_t i = 0; i < dispatch_state->constant_count; ++i) {
    sum += dispatch_state->constants[i];
  }
  iree_atomic_int32_t* output =
      reinterpret_cast<iree_atomic_int32_t*>(dispatch_state->binding_ptrs[0]);
  iree_atomic_fetch_add(output, (int32_t)sum, iree_memory_order_relaxed);
  return 0;
}

// Copies binding[0][workgroup_id_x] to binding[1][workgroup_id_x].
// Safe for multi-worker: each tile reads/writes a unique index.
static int kernel_copy_elements(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  const uint32_t* source =
      reinterpret_cast<const uint32_t*>(dispatch_state->binding_ptrs[0]);
  uint32_t* target =
      reinterpret_cast<uint32_t*>(dispatch_state->binding_ptrs[1]);
  target[workgroup_state->workgroup_id_x] =
      source[workgroup_state->workgroup_id_x];
  return 0;
}

// Returns an error code.
static int kernel_fail(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  return -1;
}

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

// Parameterized by worker_count: 1 = single-worker, >1 = multi-worker.
class BlockProcessorTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr iree_host_size_t kBlockSize = 4096;

  void SetUp() override {
    iree_arena_block_pool_initialize(kBlockSize, iree_allocator_system(),
                                     &block_pool_);
    // Minimal executable stub for tests. Only the environment field is
    // accessed by the processor (passed to the kernel function). Tests use
    // raw function pointers that ignore the environment.
    memset(&mock_executable_, 0, sizeof(mock_executable_));
  }

  void TearDown() override { iree_arena_block_pool_deinitialize(&block_pool_); }

  uint32_t worker_count() const { return GetParam(); }

  // Records a dispatch with the given kernel, workgroup count, and bindings.
  // Uses indirect fixups (binding table) for simplicity.
  struct DispatchDesc {
    iree_hal_executable_dispatch_v0_t function;
    uint32_t workgroup_count[3];
    uint8_t binding_count;
    uint16_t binding_data_base;
    uint8_t constant_count;
    const uint32_t* constants;
    uint8_t flags;
  };

  iree_status_t record_dispatch(iree_hal_cmd_block_builder_t* builder,
                                const DispatchDesc& desc,
                                const iree_hal_cmd_fixup_t* fixups,
                                uint16_t fixup_count) {
    const uint32_t tile_count = desc.workgroup_count[0] *
                                desc.workgroup_count[1] *
                                desc.workgroup_count[2];
    const iree_host_size_t cmd_size =
        iree_host_align(offsetof(iree_hal_cmd_dispatch_t, constants) +
                            desc.constant_count * sizeof(uint32_t),
                        8);

    iree_hal_cmd_dispatch_t* dispatch = NULL;
    iree_hal_cmd_fixup_t* out_fixups = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_builder_append_cmd(
        builder, IREE_HAL_CMD_DISPATCH, (iree_hal_cmd_flags_t)desc.flags,
        cmd_size, fixup_count, desc.binding_count, tile_count,
        (void**)&dispatch, &out_fixups));

    // Copy caller-provided fixups into the reserved storage.
    if (fixup_count > 0 && fixups) {
      memcpy(out_fixups, fixups, fixup_count * sizeof(iree_hal_cmd_fixup_t));
    }

    dispatch->function = desc.function;
    dispatch->executable = &mock_executable_;
    dispatch->export_ordinal = 0;
    dispatch->reserved = 0;
    dispatch->constant_count = desc.constant_count;
    dispatch->binding_count = desc.binding_count;
    dispatch->binding_data_base = desc.binding_data_base;
    dispatch->workgroup_size[0] = 1;
    dispatch->workgroup_size[1] = 1;
    dispatch->workgroup_size[2] = 1;
    dispatch->params.direct.workgroup_count[0] = desc.workgroup_count[0];
    dispatch->params.direct.workgroup_count[1] = desc.workgroup_count[1];
    dispatch->params.direct.workgroup_count[2] = desc.workgroup_count[2];
    dispatch->tile_count = tile_count;
    dispatch->tiles_per_reservation = 1;
    dispatch->local_memory_size = 0;

    if (desc.constant_count > 0 && desc.constants) {
      memcpy(dispatch->constants, desc.constants,
             desc.constant_count * sizeof(uint32_t));
    }

    return iree_ok_status();
  }

  // Executes a recording with the drain/return API. Creates the context,
  // spawns worker threads for multi-worker, runs worker 0 on the calling
  // thread via drain() loop, joins, and returns the result.
  iree_status_t execute(const iree_hal_cmd_block_recording_t* recording,
                        const iree_hal_cmd_binding_entry_t* binding_table,
                        iree_host_size_t binding_table_length,
                        uint64_t* out_total_tiles_executed = nullptr) {
    if (out_total_tiles_executed) *out_total_tiles_executed = 0;

    iree_hal_cmd_block_processor_context_t* context = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_cmd_block_processor_context_allocate(
        recording, binding_table, binding_table_length, worker_count(),
        iree_allocator_system(), &context));
    if (!context) return iree_ok_status();

    iree_atomic_int64_t total_tiles_executed = IREE_ATOMIC_VAR_INIT(0);
    if (worker_count() > 1) {
      // Spawn worker threads (1..N-1).
      std::vector<WorkerArgs> args(worker_count());
      std::vector<iree_thread_t*> threads(worker_count() - 1, nullptr);
      for (uint32_t i = 0; i < worker_count(); ++i) {
        args[i].context = context;
        args[i].total_tiles = &total_tiles_executed;
        args[i].worker_index = i;
      }
      for (uint32_t i = 1; i < worker_count(); ++i) {
        iree_thread_create_params_t params;
        memset(&params, 0, sizeof(params));
        params.name = iree_make_cstring_view("test_worker");
        iree_status_t status =
            iree_thread_create(worker_thread_entry, &args[i], params,
                               iree_allocator_system(), &threads[i - 1]);
        if (!iree_status_is_ok(status)) {
          // Abort: discard any execution result and free.
          status = iree_status_join(
              status,
              iree_hal_cmd_block_processor_context_consume_result(context));
          iree_hal_cmd_block_processor_context_free(context,
                                                    iree_allocator_system());
          for (uint32_t j = 1; j < i; ++j) {
            iree_thread_release(threads[j - 1]);
          }
          return status;
        }
      }

      // Worker 0 on the calling thread uses the same drain loop.
      iree_hal_cmd_block_processor_worker_state_t worker_state;
      memset(&worker_state, 0, sizeof(worker_state));
      iree_hal_cmd_block_processor_drain_result_t result;
      do {
        iree_hal_cmd_block_processor_drain(context, 0, &worker_state, &result);
        if (result.tiles_executed != 0) {
          iree_atomic_fetch_add(&total_tiles_executed,
                                (int64_t)result.tiles_executed,
                                iree_memory_order_relaxed);
        }
        if (!result.completed && result.tiles_executed == 0) {
          iree_thread_yield();
        }
      } while (!result.completed);

      // Join all spawned threads.
      for (uint32_t i = 1; i < worker_count(); ++i) {
        iree_thread_release(threads[i - 1]);
      }
    } else {
      // Single-worker: one drain() call completes everything.
      iree_hal_cmd_block_processor_worker_state_t worker_state;
      memset(&worker_state, 0, sizeof(worker_state));
      iree_hal_cmd_block_processor_drain_result_t result;
      iree_hal_cmd_block_processor_drain(context, 0, &worker_state, &result);
      if (result.tiles_executed != 0) {
        iree_atomic_fetch_add(&total_tiles_executed,
                              (int64_t)result.tiles_executed,
                              iree_memory_order_relaxed);
      }
    }

    iree_status_t result =
        iree_hal_cmd_block_processor_context_consume_result(context);
    if (out_total_tiles_executed) {
      *out_total_tiles_executed = (uint64_t)iree_atomic_load(
          &total_tiles_executed, iree_memory_order_relaxed);
    }
    iree_hal_cmd_block_processor_context_free(context, iree_allocator_system());
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  iree_hal_local_executable_t mock_executable_;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_P(BlockProcessorTest, EmptyRecording) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, NULL, 0));

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, SingleDispatch) {
  // Dispatch kernel_count_tiles with 8 workgroups.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_count_tiles;
  desc.workgroup_count[0] = 8;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 8);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, DispatchWritesTileIds) {
  // Dispatch kernel_write_tile_id with 16 workgroups.
  uint32_t output[16];
  memset(output, 0xFF, sizeof(output));

  iree_hal_cmd_binding_entry_t table[] = {
      {output, sizeof(output)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_write_tile_id;
  desc.workgroup_count[0] = 16;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(output[i], (uint32_t)i) << "output[" << i << "]";
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, DispatchWithConstants) {
  // Dispatch kernel_sum_constants with 3 push constants: 10 + 20 + 30 = 60.
  // 4 tiles → 4 × 60 = 240 accumulated.
  iree_atomic_int32_t result = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_cmd_binding_entry_t table[] = {
      {&result, sizeof(result)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  uint32_t constants[3] = {10, 20, 30};

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_sum_constants;
  desc.workgroup_count[0] = 4;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  desc.constant_count = 3;
  desc.constants = constants;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  EXPECT_EQ(iree_atomic_load(&result, iree_memory_order_relaxed), 240);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, FillCommand) {
  uint8_t buffer[256];
  memset(buffer, 0, sizeof(buffer));

  iree_hal_cmd_binding_entry_t table[] = {
      {buffer, sizeof(buffer)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  iree_hal_cmd_fill_t* fill = NULL;
  iree_hal_cmd_fixup_t* out_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 1, 1, 1, (void**)&fill, &out_fixups));
  memcpy(out_fixups, fixups, sizeof(fixups));
  fill->target_binding = 0;
  fill->pattern_length = 4;
  fill->params.direct.target_offset = 64;
  fill->params.direct.length = 128;
  fill->params.direct.pattern = 0xDEADBEEF;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  // Verify untouched region.
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(buffer[i], 0) << "byte " << i << " should be untouched";
  }
  // Verify filled region (128 bytes = 32 uint32_t values).
  const uint32_t* filled = reinterpret_cast<const uint32_t*>(buffer + 64);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(filled[i], 0xDEADBEEFu) << "filled[" << i << "]";
  }
  // Verify trailing untouched region.
  for (int i = 192; i < 256; ++i) {
    EXPECT_EQ(buffer[i], 0) << "byte " << i << " should be untouched";
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, LargeFillCommandUsesTransferTiles) {
  const iree_device_size_t fill_length =
      IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES * 3 + 17;
  const iree_device_size_t target_offset = 13;
  const iree_device_size_t trailing_length = 19;
  const iree_device_size_t buffer_length =
      target_offset + fill_length + trailing_length;
  std::vector<uint8_t> buffer((size_t)buffer_length, 0x11);

  iree_hal_cmd_binding_entry_t table[] = {
      {buffer.data(), buffer.size()},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  const uint32_t tile_count = iree_hal_cmd_transfer_tile_count(fill_length);
  iree_hal_cmd_fill_t* fill = NULL;
  iree_hal_cmd_fixup_t* out_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 1, 1, tile_count, (void**)&fill,
      &out_fixups));
  memcpy(out_fixups, fixups, sizeof(fixups));
  fill->target_binding = 0;
  fill->pattern_length = 1;
  fill->params.direct.target_offset = target_offset;
  fill->params.direct.length = fill_length;
  fill->params.direct.pattern = 0xA5;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  uint64_t total_tiles_executed = 0;
  IREE_ASSERT_OK(
      execute(&recording, table, IREE_ARRAYSIZE(table), &total_tiles_executed));

  EXPECT_EQ(total_tiles_executed, tile_count);
  for (iree_device_size_t i = 0; i < target_offset; ++i) {
    EXPECT_EQ(buffer[(size_t)i], 0x11) << "prefix byte " << i;
  }
  for (iree_device_size_t i = 0; i < fill_length; ++i) {
    EXPECT_EQ(buffer[(size_t)(target_offset + i)], 0xA5) << "filled byte " << i;
  }
  for (iree_device_size_t i = target_offset + fill_length; i < buffer_length;
       ++i) {
    EXPECT_EQ(buffer[(size_t)i], 0x11) << "trailing byte " << i;
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, CopyCommand) {
  uint32_t source[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  uint32_t target[8];
  memset(target, 0, sizeof(target));

  iree_hal_cmd_binding_entry_t table[] = {
      {source, sizeof(source)},
      {target, sizeof(target)},
  };

  iree_hal_cmd_fixup_t fixups[2];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;
  fixups[1].host_ptr = NULL;
  fixups[1].slot = 1;
  fixups[1].data_index = 1;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  iree_hal_cmd_copy_t* copy = NULL;
  iree_hal_cmd_fixup_t* out_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_COPY, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_copy_t), 2, 2, 1, (void**)&copy, &out_fixups));
  memcpy(out_fixups, fixups, sizeof(fixups));
  copy->source_binding = 0;
  copy->target_binding = 1;
  copy->params.direct.source_offset = 8;  // Skip first 2 uint32_ts.
  copy->params.direct.target_offset = 0;
  copy->params.direct.length = 16;  // Copy 4 uint32_ts.

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  // source[2..5] = {30, 40, 50, 60} copied to target[0..3].
  EXPECT_EQ(target[0], 30u);
  EXPECT_EQ(target[1], 40u);
  EXPECT_EQ(target[2], 50u);
  EXPECT_EQ(target[3], 60u);
  // Rest untouched.
  EXPECT_EQ(target[4], 0u);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, LargeCopyCommandUsesTransferTiles) {
  const iree_device_size_t copy_length =
      IREE_HAL_CMD_TRANSFER_TILE_LENGTH_BYTES * 2 + 123;
  const iree_device_size_t source_offset = 7;
  const iree_device_size_t target_offset = 11;
  const iree_device_size_t trailing_length = 23;
  const iree_device_size_t source_length = source_offset + copy_length;
  const iree_device_size_t target_length =
      target_offset + copy_length + trailing_length;
  std::vector<uint8_t> source((size_t)source_length);
  std::vector<uint8_t> target((size_t)target_length, 0xCC);
  for (iree_device_size_t i = 0; i < source_length; ++i) {
    source[(size_t)i] = (uint8_t)(i * 13 + 7);
  }

  iree_hal_cmd_binding_entry_t table[] = {
      {source.data(), source.size()},
      {target.data(), target.size()},
  };

  iree_hal_cmd_fixup_t fixups[2];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;
  fixups[1].host_ptr = NULL;
  fixups[1].slot = 1;
  fixups[1].data_index = 1;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  const uint32_t tile_count = iree_hal_cmd_transfer_tile_count(copy_length);
  iree_hal_cmd_copy_t* copy = NULL;
  iree_hal_cmd_fixup_t* out_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_COPY, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_copy_t), 2, 2, tile_count, (void**)&copy,
      &out_fixups));
  memcpy(out_fixups, fixups, sizeof(fixups));
  copy->source_binding = 0;
  copy->target_binding = 1;
  copy->params.direct.source_offset = source_offset;
  copy->params.direct.target_offset = target_offset;
  copy->params.direct.length = copy_length;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  uint64_t total_tiles_executed = 0;
  IREE_ASSERT_OK(
      execute(&recording, table, IREE_ARRAYSIZE(table), &total_tiles_executed));

  EXPECT_EQ(total_tiles_executed, tile_count);
  for (iree_device_size_t i = 0; i < target_offset; ++i) {
    EXPECT_EQ(target[(size_t)i], 0xCC) << "prefix byte " << i;
  }
  for (iree_device_size_t i = 0; i < copy_length; ++i) {
    EXPECT_EQ(target[(size_t)(target_offset + i)],
              source[(size_t)(source_offset + i)])
        << "copied byte " << i;
  }
  for (iree_device_size_t i = target_offset + copy_length; i < target_length;
       ++i) {
    EXPECT_EQ(target[(size_t)i], 0xCC) << "trailing byte " << i;
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, BarrierOrdering) {
  // Dispatch A: writes workgroup IDs to buffer A.
  // Barrier.
  // Dispatch B: copies buffer A to buffer B element-wise.
  uint32_t buffer_a[8];
  uint32_t buffer_b[8];
  memset(buffer_a, 0, sizeof(buffer_a));
  memset(buffer_b, 0xFF, sizeof(buffer_b));

  iree_hal_cmd_binding_entry_t table[] = {
      {buffer_a, sizeof(buffer_a)},
      {buffer_b, sizeof(buffer_b)},
  };

  // Fixups for dispatch A: binding 0 → slot 0.
  iree_hal_cmd_fixup_t fixups_a[1];
  memset(fixups_a, 0, sizeof(fixups_a));
  fixups_a[0].host_ptr = NULL;
  fixups_a[0].slot = 0;
  fixups_a[0].data_index = 0;

  // Fixups for dispatch B: binding 0 → slot 0, binding 1 → slot 1.
  iree_hal_cmd_fixup_t fixups_b[2];
  memset(fixups_b, 0, sizeof(fixups_b));
  fixups_b[0].host_ptr = NULL;
  fixups_b[0].slot = 0;
  fixups_b[0].data_index = 1;
  fixups_b[1].host_ptr = NULL;
  fixups_b[1].slot = 1;
  fixups_b[1].data_index = 2;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Dispatch A: write tile IDs to buffer_a.
  DispatchDesc desc_a = {};
  desc_a.function = kernel_write_tile_id;
  desc_a.workgroup_count[0] = 8;
  desc_a.workgroup_count[1] = 1;
  desc_a.workgroup_count[2] = 1;
  desc_a.binding_count = 1;
  desc_a.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc_a, fixups_a, 1));

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Dispatch B: copy buffer_a to buffer_b.
  DispatchDesc desc_b = {};
  desc_b.function = kernel_copy_elements;
  desc_b.workgroup_count[0] = 8;
  desc_b.workgroup_count[1] = 1;
  desc_b.workgroup_count[2] = 1;
  desc_b.binding_count = 2;
  desc_b.binding_data_base = 1;
  IREE_ASSERT_OK(record_dispatch(&builder, desc_b, fixups_b, 2));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(buffer_a[i], (uint32_t)i) << "buffer_a[" << i << "]";
    EXPECT_EQ(buffer_b[i], (uint32_t)i) << "buffer_b[" << i << "]";
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, DirectFixup) {
  // Use direct inline fixups (host_ptr) instead of the binding table.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = &counter;
  fixups[0].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
  fixups[0].offset = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_count_tiles;
  desc.workgroup_count[0] = 5;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  // No binding table — all fixups are direct.
  IREE_ASSERT_OK(execute(&recording, NULL, 0));

  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 5);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, KernelFailurePropagates) {
  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_fail;
  desc.workgroup_count[0] = 1;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, NULL, 0));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, execute(&recording, NULL, 0));

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, MultiBlockExecution) {
  // Record 200 dispatches (will cause block splitting) and verify all execute.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  for (int i = 0; i < 200; ++i) {
    DispatchDesc desc = {};
    desc.function = kernel_count_tiles;
    desc.workgroup_count[0] = 1;
    desc.workgroup_count[1] = 1;
    desc.workgroup_count[2] = 1;
    desc.binding_count = 1;
    desc.binding_data_base = 0;
    IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  EXPECT_GT(recording.block_count, 1);

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 200);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, IndirectDispatch) {
  // Dispatch with INDIRECT flag: workgroup count read from a buffer.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_dispatch_params_t params;
  params.workgroup_count[0] = 12;
  params.workgroup_count[1] = 1;
  params.workgroup_count[2] = 1;

  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
      {&params, sizeof(params)},
  };

  // Fixup 0: counter binding, fixup 1: params binding.
  iree_hal_cmd_fixup_t fixups[2];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;
  fixups[1].host_ptr = NULL;
  fixups[1].slot = 1;
  fixups[1].data_index = 1;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Record an indirect dispatch.
  const iree_host_size_t cmd_size =
      iree_host_align(sizeof(iree_hal_cmd_dispatch_t), 8);
  iree_hal_cmd_dispatch_t* dispatch = NULL;
  iree_hal_cmd_fixup_t* out_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_DISPATCH, IREE_HAL_CMD_FLAG_INDIRECT, cmd_size, 2,
      2,
      12,  // tile_count hint
      (void**)&dispatch, &out_fixups));
  memcpy(out_fixups, fixups, sizeof(fixups));

  dispatch->function = kernel_count_tiles;
  dispatch->executable = &mock_executable_;
  dispatch->export_ordinal = 0;
  dispatch->reserved = 0;
  dispatch->constant_count = 0;
  dispatch->binding_count = 2;
  dispatch->binding_data_base = 0;
  dispatch->workgroup_size[0] = 1;
  dispatch->workgroup_size[1] = 1;
  dispatch->workgroup_size[2] = 1;
  dispatch->params.indirect.params_binding = 1;
  dispatch->params.indirect.params_offset = 0;
  dispatch->params.indirect.tile_count_hint = 12;
  dispatch->tile_count = 12;
  dispatch->tiles_per_reservation = 1;
  dispatch->local_memory_size = 0;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 12);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, PredicatedDispatchSkipped) {
  // Indirect predicated dispatch with zero workgroup counts → should be
  // skipped entirely.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_dispatch_params_t params;
  params.workgroup_count[0] = 0;
  params.workgroup_count[1] = 0;
  params.workgroup_count[2] = 0;

  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
      {&params, sizeof(params)},
  };

  iree_hal_cmd_fixup_t fixups[2];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;
  fixups[1].host_ptr = NULL;
  fixups[1].slot = 1;
  fixups[1].data_index = 1;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  const iree_host_size_t cmd_size =
      iree_host_align(sizeof(iree_hal_cmd_dispatch_t), 8);
  iree_hal_cmd_dispatch_t* dispatch = NULL;
  iree_hal_cmd_fixup_t* out_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_DISPATCH,
      IREE_HAL_CMD_FLAG_INDIRECT | IREE_HAL_CMD_FLAG_PREDICATED, cmd_size, 2, 2,
      0, (void**)&dispatch, &out_fixups));
  memcpy(out_fixups, fixups, sizeof(fixups));

  dispatch->function = kernel_count_tiles;
  dispatch->executable = &mock_executable_;
  dispatch->export_ordinal = 0;
  dispatch->reserved = 0;
  dispatch->constant_count = 0;
  dispatch->binding_count = 2;
  dispatch->binding_data_base = 0;
  dispatch->workgroup_size[0] = 1;
  dispatch->workgroup_size[1] = 1;
  dispatch->workgroup_size[2] = 1;
  dispatch->params.indirect.params_binding = 1;
  dispatch->params.indirect.params_offset = 0;
  dispatch->params.indirect.tile_count_hint = 0;
  dispatch->tile_count = 0;
  dispatch->tiles_per_reservation = 1;
  dispatch->local_memory_size = 0;

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 0);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, ThreeDimensionalDispatch) {
  // 4×3×2 = 24 tiles. Verify the tile count and decomposition.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_count_tiles;
  desc.workgroup_count[0] = 4;
  desc.workgroup_count[1] = 3;
  desc.workgroup_count[2] = 2;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 24);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, MixedCommandSequence) {
  // Fill → dispatch → copy: exercises all three work command types in
  // sequence with shared bindings.
  uint32_t source[4];
  uint32_t target[4];
  memset(source, 0, sizeof(source));
  memset(target, 0, sizeof(target));

  iree_hal_cmd_binding_entry_t table[] = {
      {source, sizeof(source)},
      {target, sizeof(target)},
  };

  // Fixups: slot 0 → data_index 0, slot 1 → data_index 1.
  iree_hal_cmd_fixup_t fixups[2];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;
  fixups[1].host_ptr = NULL;
  fixups[1].slot = 1;
  fixups[1].data_index = 1;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Step 1: Fill source with 0xAA pattern.
  iree_hal_cmd_fill_t* fill = NULL;
  iree_hal_cmd_fixup_t* fill_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_FILL, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_fill_t), 1, 1, 1, (void**)&fill, &fill_fixups));
  memcpy(fill_fixups, &fixups[0], sizeof(fixups[0]));
  fill->target_binding = 0;
  fill->pattern_length = 1;
  fill->params.direct.target_offset = 0;
  fill->params.direct.length = sizeof(source);
  fill->params.direct.pattern = 0xAA;

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Step 2: Dispatch writes tile IDs into source (overwrites fill).
  DispatchDesc desc = {};
  desc.function = kernel_write_tile_id;
  desc.workgroup_count[0] = 4;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, NULL, 0));

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Step 3: Copy source to target.
  iree_hal_cmd_copy_t* copy = NULL;
  iree_hal_cmd_fixup_t* copy_fixups = NULL;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_append_cmd(
      &builder, IREE_HAL_CMD_COPY, IREE_HAL_CMD_FLAG_NONE,
      sizeof(iree_hal_cmd_copy_t), 1, 1, 1, (void**)&copy, &copy_fixups));
  memcpy(copy_fixups, &fixups[1], sizeof(fixups[1]));
  copy->source_binding = 0;
  copy->target_binding = 1;
  copy->params.direct.source_offset = 0;
  copy->params.direct.target_offset = 0;
  copy->params.direct.length = sizeof(source);

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  // target should contain {0, 1, 2, 3} (tile IDs written by dispatch).
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(source[i], (uint32_t)i) << "source[" << i << "]";
    EXPECT_EQ(target[i], (uint32_t)i) << "target[" << i << "]";
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, LargeDispatchMultiWorker) {
  // 1024 tiles across workers — exercises actual concurrent tile claiming.
  uint32_t output[1024];
  memset(output, 0xFF, sizeof(output));

  iree_hal_cmd_binding_entry_t table[] = {
      {output, sizeof(output)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  DispatchDesc desc = {};
  desc.function = kernel_write_tile_id;
  desc.workgroup_count[0] = 1024;
  desc.workgroup_count[1] = 1;
  desc.workgroup_count[2] = 1;
  desc.binding_count = 1;
  desc.binding_data_base = 0;
  IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  // Every tile should have written its ID exactly once.
  for (int i = 0; i < 1024; ++i) {
    EXPECT_EQ(output[i], (uint32_t)i) << "output[" << i << "]";
  }

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, MultiRegionMultiDispatch) {
  // Three regions, each with multiple dispatches. Verifies that workers
  // correctly handle region transitions with multiple dispatches per region.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Region 0: 3 dispatches × 10 tiles = 30 tiles.
  for (int i = 0; i < 3; ++i) {
    DispatchDesc desc = {};
    desc.function = kernel_count_tiles;
    desc.workgroup_count[0] = 10;
    desc.workgroup_count[1] = 1;
    desc.workgroup_count[2] = 1;
    desc.binding_count = 1;
    desc.binding_data_base = 0;
    IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));
  }

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Region 1: 2 dispatches × 20 tiles = 40 tiles.
  for (int i = 0; i < 2; ++i) {
    DispatchDesc desc = {};
    desc.function = kernel_count_tiles;
    desc.workgroup_count[0] = 20;
    desc.workgroup_count[1] = 1;
    desc.workgroup_count[2] = 1;
    desc.binding_count = 1;
    desc.binding_data_base = 0;
    IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));
  }

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Region 2: 1 dispatch × 50 tiles = 50 tiles.
  {
    DispatchDesc desc = {};
    desc.function = kernel_count_tiles;
    desc.workgroup_count[0] = 50;
    desc.workgroup_count[1] = 1;
    desc.workgroup_count[2] = 1;
    desc.binding_count = 1;
    desc.binding_data_base = 0;
    IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  // Total tiles: 30 + 40 + 50 = 120.
  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 120);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

TEST_P(BlockProcessorTest, NarrowToWideTransition) {
  // Region 0: 1 tile (narrow — only 1 worker gets work, others signal
  // arrival immediately). Region 1: 128 tiles (wide — all workers
  // participate). Exercises the cold-start path: non-contributing workers
  // signal arrival instantly, completer's wait resolves near-immediately.
  iree_atomic_int32_t counter = IREE_ATOMIC_VAR_INIT(0);
  iree_hal_cmd_binding_entry_t table[] = {
      {&counter, sizeof(counter)},
  };

  iree_hal_cmd_fixup_t fixups[1];
  memset(fixups, 0, sizeof(fixups));
  fixups[0].host_ptr = NULL;
  fixups[0].slot = 0;
  fixups[0].data_index = 0;

  iree_hal_cmd_block_builder_t builder;
  iree_hal_cmd_block_builder_initialize(&block_pool_, &builder);
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_begin(&builder));

  // Region 0: 1 tile.
  {
    DispatchDesc desc = {};
    desc.function = kernel_count_tiles;
    desc.workgroup_count[0] = 1;
    desc.workgroup_count[1] = 1;
    desc.workgroup_count[2] = 1;
    desc.binding_count = 1;
    desc.binding_data_base = 0;
    IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));
  }

  IREE_ASSERT_OK(iree_hal_cmd_block_builder_barrier(&builder));

  // Region 1: 128 tiles.
  {
    DispatchDesc desc = {};
    desc.function = kernel_count_tiles;
    desc.workgroup_count[0] = 128;
    desc.workgroup_count[1] = 1;
    desc.workgroup_count[2] = 1;
    desc.binding_count = 1;
    desc.binding_data_base = 0;
    IREE_ASSERT_OK(record_dispatch(&builder, desc, fixups, 1));
  }

  iree_hal_cmd_block_recording_t recording;
  IREE_ASSERT_OK(iree_hal_cmd_block_builder_end(&builder, &recording));

  IREE_ASSERT_OK(execute(&recording, table, IREE_ARRAYSIZE(table)));

  // Total: 1 + 128 = 129.
  EXPECT_EQ(iree_atomic_load(&counter, iree_memory_order_relaxed), 129);

  iree_hal_cmd_block_recording_release(&recording);
  iree_hal_cmd_block_builder_deinitialize(&builder);
}

// Run all tests with 1, 2, 4, and 8 workers.
INSTANTIATE_TEST_SUITE_P(WorkerCounts, BlockProcessorTest,
                         ::testing::Values(1, 2, 4, 8), [](const auto& info) {
                           return "Workers" + std::to_string(info.param);
                         });

}  // namespace
