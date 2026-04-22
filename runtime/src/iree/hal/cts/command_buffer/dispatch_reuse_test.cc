// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for reusable command buffers with indirect binding tables.
//
// This is the hot path for model execution: a command buffer is recorded once
// with indirect bindings, then re-submitted multiple times with different
// binding tables (different transient allocations, different input/output
// buffers, etc.). The existing dispatch tests only use ONE_SHOT mode with a
// single submission — they miss entire categories of bugs around command buffer
// reuse, binding table lifecycle, and multi-submission ordering.
//
// Test scenarios:
//   - Record once, submit multiple times with different binding tables
//   - Chained submissions with semaphore ordering (alloca→execute→dealloca)
//   - Multiple dispatches in a single reusable command buffer
//   - Large workgroup counts through reusable command buffers

#include <cstdint>
#include <numeric>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;
using ::testing::Each;

class DispatchReuseTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    // Load the workgroup-ID kernel: writes workgroup_id[0] to buffer[wg_id].
    {
      iree_hal_executable_params_t params;
      iree_hal_executable_params_initialize(&params);
      params.caching_mode =
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
      params.executable_format = iree_make_cstring_view(executable_format());
      params.executable_data = executable_data(iree_make_cstring_view(
          "command_buffer_dispatch_multi_workgroup_test.bin"));
      IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
          executable_cache_, &params, &workgroup_id_executable_));
    }

    // Load the absf kernel: output[i] = abs(input[i]).
    {
      iree_hal_executable_params_t params;
      iree_hal_executable_params_initialize(&params);
      params.caching_mode =
          IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
      params.executable_format = iree_make_cstring_view(executable_format());
      params.executable_data = executable_data(
          iree_make_cstring_view("command_buffer_dispatch_test.bin"));
      IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
          executable_cache_, &params, &absf_executable_));
    }
  }

  void TearDown() override {
    iree_hal_executable_release(absf_executable_);
    absf_executable_ = nullptr;
    iree_hal_executable_release(workgroup_id_executable_);
    workgroup_id_executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  // Records a reusable command buffer that dispatches the workgroup-ID kernel
  // with the given workgroup count. Uses a single indirect binding (slot 0)
  // for the output buffer.
  void RecordWorkgroupIdDispatch(
      iree_host_size_t workgroup_count,
      iree_hal_command_buffer_t** out_command_buffer) {
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/1, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

    iree_hal_buffer_ref_t binding_refs[1] = {{
        /*binding=*/0,
        /*buffer_slot=*/0,
        /*buffer=*/nullptr,
        /*offset=*/0,
        /*length=*/workgroup_count * sizeof(uint32_t),
    }};
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/1,
        /*.values=*/binding_refs,
    };

    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer, workgroup_id_executable_, /*entry_point=*/0,
        iree_hal_make_static_dispatch_config(workgroup_count, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

    IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
        command_buffer,
        /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH |
            IREE_HAL_EXECUTION_STAGE_TRANSFER |
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
            IREE_HAL_EXECUTION_STAGE_DISPATCH |
            IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
        /*memory_barriers=*/nullptr,
        /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
    *out_command_buffer = command_buffer;
  }

  // Submits a command buffer with a binding table and waits for completion.
  void SubmitWithBindingsAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table) {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* workgroup_id_executable_ = nullptr;
  iree_hal_executable_t* absf_executable_ = nullptr;
};

// Records a command buffer once and submits it multiple times with different
// output buffers via the binding table. Each submission writes workgroup IDs
// to a different buffer. This is the fundamental reuse pattern for model
// execution.
TEST_P(DispatchReuseTest, ResubmitWithDifferentBindings) {
  static constexpr iree_host_size_t kWorkgroupCount = 64;
  const iree_device_size_t buffer_size = kWorkgroupCount * sizeof(uint32_t);

  // Record once with indirect bindings.
  Ref<iree_hal_command_buffer_t> command_buffer;
  RecordWorkgroupIdDispatch(kWorkgroupCount, command_buffer.out());

  // Submit 3 times, each with a different output buffer.
  for (int iteration = 0; iteration < 3; ++iteration) {
    Ref<iree_hal_buffer_t> output;
    CreateZeroedDeviceBuffer(buffer_size, output.out());

    iree_hal_buffer_binding_t bindings[1] = {{
        /*buffer=*/output,
        /*offset=*/0,
        /*length=*/buffer_size,
    }};
    iree_hal_buffer_binding_table_t binding_table = {
        /*.count=*/1,
        /*.bindings=*/bindings,
    };

    SubmitWithBindingsAndWait(command_buffer, binding_table);

    auto data = ReadBufferData<uint32_t>(output);
    std::vector<uint32_t> expected(kWorkgroupCount);
    std::iota(expected.begin(), expected.end(), 0u);
    EXPECT_THAT(data, ContainerEq(expected))
        << "Mismatch on iteration " << iteration;
  }
}

// Submits a reusable command buffer with large workgroup counts. The existing
// dispatch tests max out at 32 workgroups — real models use thousands.
TEST_P(DispatchReuseTest, LargeWorkgroupCount) {
  static constexpr iree_host_size_t kWorkgroupCount = 1024;
  const iree_device_size_t buffer_size = kWorkgroupCount * sizeof(uint32_t);

  Ref<iree_hal_command_buffer_t> command_buffer;
  RecordWorkgroupIdDispatch(kWorkgroupCount, command_buffer.out());

  Ref<iree_hal_buffer_t> output;
  CreateZeroedDeviceBuffer(buffer_size, output.out());

  iree_hal_buffer_binding_t bindings[1] = {{output, 0, buffer_size}};
  iree_hal_buffer_binding_table_t binding_table = {1, bindings};
  SubmitWithBindingsAndWait(command_buffer, binding_table);

  auto data = ReadBufferData<uint32_t>(output);
  std::vector<uint32_t> expected(kWorkgroupCount);
  std::iota(expected.begin(), expected.end(), 0u);
  EXPECT_THAT(data, ContainerEq(expected));
}

// Records one dispatch with a direct input buffer and an indirect output
// buffer. This is distinct from the all-direct/all-indirect cases: replay must
// combine the baked static pointer source and the queue_execute binding-table
// source in the same HAL ABI binding prefix.
TEST_P(DispatchReuseTest, MixedDirectAndIndirectBindings) {
  static constexpr iree_device_size_t kElementCount = 4;
  static constexpr iree_device_size_t kByteLength =
      kElementCount * sizeof(float);

  Ref<iree_hal_buffer_t> input;
  CreateFilledDeviceBuffer<float>(kByteLength, -2.5f, input.out());

  Ref<iree_hal_buffer_t> output;
  CreateFilledDeviceBuffer<float>(kByteLength, -9.0f, output.out());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      {/*binding=*/0, /*buffer_slot=*/0, input,
       /*offset=*/1 * sizeof(float), /*length=*/2 * sizeof(float)},
      {/*binding=*/1, /*buffer_slot=*/0, /*buffer=*/nullptr,
       /*offset=*/1 * sizeof(float), /*length=*/2 * sizeof(float)},
  };
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, absf_executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_buffer_binding_t table_bindings[1] = {{
      /*buffer=*/output,
      /*offset=*/0,
      /*length=*/IREE_HAL_WHOLE_BUFFER,
  }};
  iree_hal_buffer_binding_table_t binding_table = {1, table_bindings};
  SubmitWithBindingsAndWait(command_buffer, binding_table);

  auto data = ReadBufferData<float>(output);
  EXPECT_THAT(data, ::testing::ElementsAre(-9.0f, 2.5f, 2.5f, -9.0f));
}

// Submits an indirect dispatch command buffer behind an unresolved wait, then
// releases the command buffer and an input binding before the wait is signaled.
// queue_execute must copy the binding table metadata and retain the bound
// resources until the signal semaphore publishes completion.
TEST_P(DispatchReuseTest, DeferredExecuteRetainsDispatchBindingTable) {
  static constexpr iree_device_size_t kElementCount = 4;
  static constexpr iree_device_size_t kByteLength =
      kElementCount * sizeof(float);

  iree_hal_buffer_t* input = nullptr;
  CreateFilledDeviceBuffer<float>(kByteLength, -2.5f, &input);

  Ref<iree_hal_buffer_t> output;
  CreateFilledDeviceBuffer<float>(kByteLength, -9.0f, output.out());

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t binding_refs[2] = {
      {/*binding=*/0, /*buffer_slot=*/0, /*buffer=*/nullptr,
       /*offset=*/1 * sizeof(float), /*length=*/2 * sizeof(float)},
      {/*binding=*/1, /*buffer_slot=*/1, /*buffer=*/nullptr,
       /*offset=*/1 * sizeof(float), /*length=*/2 * sizeof(float)},
  };
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, absf_executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList wait(device_, {0}, {1});
  SemaphoreList signal(device_, {0}, {1});
  iree_hal_buffer_binding_t table_bindings[2] = {
      {input, 0, IREE_HAL_WHOLE_BUFFER},
      {output, 0, IREE_HAL_WHOLE_BUFFER},
  };
  iree_hal_buffer_binding_table_t binding_table = {
      IREE_ARRAYSIZE(table_bindings),
      table_bindings,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, wait, signal, command_buffer,
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(input);
  table_bindings[0] = {nullptr, 0, 0};
  table_bindings[1] = {nullptr, 0, 0};

  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(wait.semaphores[0], 1, /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  auto data = ReadBufferData<float>(output);
  EXPECT_THAT(data, ::testing::ElementsAre(-9.0f, 2.5f, 2.5f, -9.0f));
}

// Simulates the hot path: alloca → execute → dealloca, all chained via
// semaphores. The command buffer is reusable; each iteration gets a fresh
// transient buffer.
TEST_P(DispatchReuseTest, AllocaExecuteDeallocaCycle) {
  static constexpr iree_host_size_t kWorkgroupCount = 128;
  const iree_device_size_t buffer_size = kWorkgroupCount * sizeof(uint32_t);

  // Record the reusable command buffer.
  Ref<iree_hal_command_buffer_t> command_buffer;
  RecordWorkgroupIdDispatch(kWorkgroupCount, command_buffer.out());

  iree_hal_buffer_params_t alloca_params = {0};
  alloca_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  alloca_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;

  // Run the alloca → execute → verify → dealloca cycle 3 times.
  // The dealloca must be deferred until AFTER host readback because the
  // dealloca decommits the transient buffer's backing. Since the dealloca
  // waits on the same semaphore (execute_signal) that gates our readback,
  // it can race with us and decommit before we read.
  for (int iteration = 0; iteration < 3; ++iteration) {
    // Step 1: alloca.
    SemaphoreList alloca_signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    iree_hal_buffer_t* transient_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_device_queue_alloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
        /*pool=*/NULL, alloca_params, buffer_size, IREE_HAL_ALLOCA_FLAG_NONE,
        &transient_buffer));
    Ref<iree_hal_buffer_t> transient(transient_buffer);

    // Step 2: execute — waits on alloca, signals execute.
    SemaphoreList execute_signal(device_, {0}, {1});
    iree_hal_buffer_binding_t bindings[1] = {{transient, 0, buffer_size}};
    iree_hal_buffer_binding_table_t binding_table = {1, bindings};
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_signal, execute_signal,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

    // Wait for execute to complete, then verify.
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    auto data = ReadBufferData<uint32_t>(transient);
    std::vector<uint32_t> expected(kWorkgroupCount);
    std::iota(expected.begin(), expected.end(), 0u);
    EXPECT_THAT(data, ContainerEq(expected))
        << "Mismatch on iteration " << iteration;

    // Step 3: dealloca — AFTER readback to avoid decommit race.
    SemaphoreList dealloca_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, execute_signal, dealloca_signal,
        transient, IREE_HAL_DEALLOCA_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
}

// Pipelines multiple alloca → execute sequences before dealloca. This is the
// pattern when a model has multiple compute phases with separate transient
// allocations.
TEST_P(DispatchReuseTest, PipelinedAllocaExecuteDealloca) {
  static constexpr iree_host_size_t kWorkgroupCount = 64;
  const iree_device_size_t buffer_size = kWorkgroupCount * sizeof(uint32_t);

  Ref<iree_hal_command_buffer_t> command_buffer;
  RecordWorkgroupIdDispatch(kWorkgroupCount, command_buffer.out());

  iree_hal_buffer_params_t alloca_params = {0};
  alloca_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  alloca_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;

  SemaphoreList empty_wait;

  // Phase 1: alloca + execute for buffer A.
  SemaphoreList alloca_a_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw_a = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_a_signal,
      /*pool=*/NULL, alloca_params, buffer_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &raw_a));
  Ref<iree_hal_buffer_t> buffer_a(raw_a);

  SemaphoreList execute_a_signal(device_, {0}, {1});
  iree_hal_buffer_binding_t bindings_a[1] = {{buffer_a, 0, buffer_size}};
  iree_hal_buffer_binding_table_t table_a = {1, bindings_a};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_a_signal, execute_a_signal,
      command_buffer, table_a, IREE_HAL_EXECUTE_FLAG_NONE));

  // Phase 2: alloca + execute for buffer B (independent of phase 1).
  SemaphoreList alloca_b_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw_b = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_b_signal,
      /*pool=*/NULL, alloca_params, buffer_size, IREE_HAL_ALLOCA_FLAG_NONE,
      &raw_b));
  Ref<iree_hal_buffer_t> buffer_b(raw_b);

  SemaphoreList execute_b_signal(device_, {0}, {1});
  iree_hal_buffer_binding_t bindings_b[1] = {{buffer_b, 0, buffer_size}};
  iree_hal_buffer_binding_table_t table_b = {1, bindings_b};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, alloca_b_signal, execute_b_signal,
      command_buffer, table_b, IREE_HAL_EXECUTE_FLAG_NONE));

  // Wait for both executes, then verify results.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_a_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_b_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> expected(kWorkgroupCount);
  std::iota(expected.begin(), expected.end(), 0u);

  auto data_a = ReadBufferData<uint32_t>(buffer_a);
  EXPECT_THAT(data_a, ContainerEq(expected)) << "Buffer A mismatch";

  auto data_b = ReadBufferData<uint32_t>(buffer_b);
  EXPECT_THAT(data_b, ContainerEq(expected)) << "Buffer B mismatch";

  // Dealloca AFTER readback to avoid decommit race (the dealloca waits on
  // execute_signal, which is the same condition we waited on for readback —
  // if we enqueue dealloca before reading, it can race and decommit before
  // our d2h transfer maps the buffer).
  SemaphoreList dealloca_a_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, execute_a_signal, dealloca_a_signal,
      buffer_a, IREE_HAL_DEALLOCA_FLAG_NONE));
  SemaphoreList dealloca_b_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, execute_b_signal, dealloca_b_signal,
      buffer_b, IREE_HAL_DEALLOCA_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_a_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_b_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

// Records a reusable command buffer with multiple dispatches separated by
// barriers, then submits it. This exercises the multi-dispatch recording path
// that real models use (hundreds of dispatches per command buffer).
TEST_P(DispatchReuseTest, MultipleDispatchesInSingleCommandBuffer) {
  static constexpr iree_host_size_t kWorkgroupCount = 32;
  const iree_device_size_t buffer_size = kWorkgroupCount * sizeof(uint32_t);

  // Record two dispatches writing to different binding slots,
  // separated by a barrier.
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  // Dispatch 1: write workgroup IDs to slot 0.
  {
    iree_hal_buffer_ref_t refs[1] = {{0, 0, nullptr, 0, buffer_size}};
    iree_hal_buffer_ref_list_t bindings = {1, refs};
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer, workgroup_id_executable_, /*entry_point=*/0,
        iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  }

  // Barrier between dispatches.
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  // Dispatch 2: write workgroup IDs to slot 1.
  {
    iree_hal_buffer_ref_t refs[1] = {{0, 1, nullptr, 0, buffer_size}};
    iree_hal_buffer_ref_list_t bindings = {1, refs};
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer, workgroup_id_executable_, /*entry_point=*/0,
        iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  }

  // Final barrier.
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  // Submit with two different output buffers in the binding table.
  Ref<iree_hal_buffer_t> output_a;
  CreateZeroedDeviceBuffer(buffer_size, output_a.out());
  Ref<iree_hal_buffer_t> output_b;
  CreateZeroedDeviceBuffer(buffer_size, output_b.out());

  iree_hal_buffer_binding_t table_bindings[2] = {
      {output_a, 0, buffer_size},
      {output_b, 0, buffer_size},
  };
  iree_hal_buffer_binding_table_t binding_table = {2, table_bindings};

  SubmitWithBindingsAndWait(command_buffer, binding_table);

  // Both buffers should contain workgroup IDs.
  std::vector<uint32_t> expected(kWorkgroupCount);
  std::iota(expected.begin(), expected.end(), 0u);

  auto data_a = ReadBufferData<uint32_t>(output_a);
  EXPECT_THAT(data_a, ContainerEq(expected)) << "Output A mismatch";

  auto data_b = ReadBufferData<uint32_t>(output_b);
  EXPECT_THAT(data_b, ContainerEq(expected)) << "Output B mismatch";
}

// Resubmits a multi-dispatch command buffer multiple times with different
// binding tables. This is the full hot-path pattern: record N dispatches
// once, then iterate alloca→execute→dealloca with different transient
// buffers each time.
TEST_P(DispatchReuseTest, MultiDispatchMultiResubmit) {
  static constexpr iree_host_size_t kWorkgroupCount = 32;
  const iree_device_size_t buffer_size = kWorkgroupCount * sizeof(uint32_t);

  // Record command buffer with two dispatches to different slots.
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  iree_hal_buffer_ref_t refs_0[1] = {{0, 0, nullptr, 0, buffer_size}};
  iree_hal_buffer_ref_list_t bindings_0 = {1, refs_0};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, workgroup_id_executable_, 0,
      iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
      iree_const_byte_span_empty(), bindings_0, IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  iree_hal_buffer_ref_t refs_1[1] = {{0, 1, nullptr, 0, buffer_size}};
  iree_hal_buffer_ref_list_t bindings_1 = {1, refs_1};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, workgroup_id_executable_, 0,
      iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
      iree_const_byte_span_empty(), bindings_1, IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 0, nullptr));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  std::vector<uint32_t> expected(kWorkgroupCount);
  std::iota(expected.begin(), expected.end(), 0u);

  // Submit 5 times with different buffer pairs each time.
  for (int iteration = 0; iteration < 5; ++iteration) {
    Ref<iree_hal_buffer_t> out_a;
    CreateZeroedDeviceBuffer(buffer_size, out_a.out());
    Ref<iree_hal_buffer_t> out_b;
    CreateZeroedDeviceBuffer(buffer_size, out_b.out());

    iree_hal_buffer_binding_t table_bindings[2] = {
        {out_a, 0, buffer_size},
        {out_b, 0, buffer_size},
    };
    iree_hal_buffer_binding_table_t binding_table = {2, table_bindings};

    SubmitWithBindingsAndWait(command_buffer, binding_table);

    auto data_a = ReadBufferData<uint32_t>(out_a);
    EXPECT_THAT(data_a, ContainerEq(expected))
        << "Output A mismatch on iteration " << iteration;

    auto data_b = ReadBufferData<uint32_t>(out_b);
    EXPECT_THAT(data_b, ContainerEq(expected))
        << "Output B mismatch on iteration " << iteration;
  }
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(DispatchReuseTest);

}  // namespace iree::hal::cts
