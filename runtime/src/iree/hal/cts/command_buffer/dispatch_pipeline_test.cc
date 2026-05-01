// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for data flow pipelines through dispatch operations.
//
// Exercises the patterns real models use: fill buffers with data, dispatch
// kernels that transform the data, chain multiple dispatches via barriers,
// and verify results flow correctly through the pipeline. Uses the
// scale_and_offset kernel (output[i] = input[i] * scale + offset) which
// produces verifiable transforms at each stage.
//
// The existing dispatch tests verify that individual dispatches produce
// correct output. These tests verify that DATA FLOWS correctly between
// operations: fill → dispatch, dispatch → dispatch, and through the full
// alloca → fill → dispatch → dealloca lifecycle.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class DispatchPipelineTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kElementCount = 4;
  static constexpr iree_device_size_t kBufferSize =
      kElementCount * sizeof(uint32_t);

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    // Load the scale_and_offset kernel:
    //   output[i] = input[i] * scale + offset
    //   2 push constants (scale, offset), 2 bindings (input, output)
    iree_hal_executable_params_t params;
    iree_hal_executable_params_initialize(&params);
    params.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    params.executable_format = iree_make_cstring_view(executable_format());
    params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));
    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &params, &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  // Records a scale_and_offset dispatch into |cmd| using indirect bindings.
  // Binding slot |input_slot| = input buffer, |output_slot| = output buffer.
  void RecordScaleAndOffset(iree_hal_command_buffer_t* cmd, uint32_t scale,
                            uint32_t offset, iree_host_size_t input_slot,
                            iree_host_size_t output_slot) {
    iree_hal_buffer_ref_t refs[2] = {
        {/*binding=*/0, /*buffer_slot=*/(uint8_t)input_slot, /*buffer=*/nullptr,
         /*offset=*/0, /*length=*/kBufferSize},
        {/*binding=*/1, /*buffer_slot=*/(uint8_t)output_slot,
         /*buffer=*/nullptr, /*offset=*/0, /*length=*/kBufferSize},
    };
    iree_hal_buffer_ref_list_t bindings = {2, refs};

    std::vector<uint32_t> constant_data = {scale, offset};
    iree_const_byte_span_t constants = iree_make_const_byte_span(
        constant_data.data(), constant_data.size() * sizeof(uint32_t));

    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        cmd, executable_, /*entry_point=*/0,
        iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
        IREE_HAL_DISPATCH_FLAG_NONE));
  }

  // Records a full barrier between dispatch/transfer stages.
  void RecordBarrier(iree_hal_command_buffer_t* cmd) {
    const iree_hal_memory_barrier_t memory_barrier = {
        IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
            IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
            IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
        IREE_HAL_ACCESS_SCOPE_DISPATCH_READ | IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
        cmd,
        IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
            IREE_HAL_EXECUTION_STAGE_DISPATCH |
            IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, nullptr));
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

// Fills an input buffer via queue_update, dispatches scale_and_offset,
// and verifies the output. Tests data flow from host → buffer → dispatch
// → buffer → host.
TEST_P(DispatchPipelineTest, UpdateThenDispatch) {
  std::vector<uint32_t> input_data = {1, 2, 3, 4};
  Ref<iree_hal_buffer_t> input;
  IREE_ASSERT_OK(
      CreateDeviceBufferWithData(input_data.data(), kBufferSize, input.out()));
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, output.out()));

  // Record: dispatch scale=2, offset=5.
  Ref<iree_hal_command_buffer_t> cmd;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, cmd.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd));
  RecordScaleAndOffset(cmd, /*scale=*/2, /*offset=*/5, /*input_slot=*/0,
                       /*output_slot=*/1);
  RecordBarrier(cmd);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd));

  iree_hal_buffer_binding_t table[2] = {
      {input, 0, kBufferSize},
      {output, 0, kBufferSize},
  };
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      cmd, iree_hal_buffer_binding_table_t{2, table}));

  // output[i] = input[i] * 2 + 5 = [7, 9, 11, 13]
  auto data = ReadBufferData<uint32_t>(output);
  EXPECT_THAT(data, ContainerEq(std::vector<uint32_t>{7, 9, 11, 13}));
}

// Chains two dispatches in a single command buffer with a barrier between
// them. The first dispatch's output feeds into the second dispatch's input.
// This is the fundamental multi-dispatch pattern used in real models.
TEST_P(DispatchPipelineTest, ChainedDispatches) {
  std::vector<uint32_t> input_data = {1, 2, 3, 4};
  Ref<iree_hal_buffer_t> input;
  IREE_ASSERT_OK(
      CreateDeviceBufferWithData(input_data.data(), kBufferSize, input.out()));
  Ref<iree_hal_buffer_t> intermediate;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, intermediate.out()));
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, output.out()));

  // Record: dispatch A → barrier → dispatch B.
  // Dispatch A: intermediate[i] = input[i] * 2 + 0 = [2, 4, 6, 8]
  // Dispatch B: output[i] = intermediate[i] * 1 + 10 = [12, 14, 16, 18]
  Ref<iree_hal_command_buffer_t> cmd;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/3, cmd.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd));

  // Dispatch A: slots 0→1 (input→intermediate).
  RecordScaleAndOffset(cmd, /*scale=*/2, /*offset=*/0, /*input_slot=*/0,
                       /*output_slot=*/1);
  RecordBarrier(cmd);

  // Dispatch B: slots 1→2 (intermediate→output).
  RecordScaleAndOffset(cmd, /*scale=*/1, /*offset=*/10, /*input_slot=*/1,
                       /*output_slot=*/2);
  RecordBarrier(cmd);

  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd));

  iree_hal_buffer_binding_t table[3] = {
      {input, 0, kBufferSize},
      {intermediate, 0, kBufferSize},
      {output, 0, kBufferSize},
  };
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      cmd, iree_hal_buffer_binding_table_t{3, table}));

  auto data = ReadBufferData<uint32_t>(output);
  EXPECT_THAT(data, ContainerEq(std::vector<uint32_t>{12, 14, 16, 18}));

  // Also verify the intermediate buffer to ensure the chain worked correctly.
  auto intermediate_data = ReadBufferData<uint32_t>(intermediate);
  EXPECT_THAT(intermediate_data,
              ContainerEq(std::vector<uint32_t>{2, 4, 6, 8}));
}

// Three-stage pipeline: fill → dispatch → dispatch, all chained with barriers
// in a single command buffer. Exercises transfer→dispatch and dispatch→dispatch
// transitions.
TEST_P(DispatchPipelineTest, FillDispatchDispatchPipeline) {
  Ref<iree_hal_buffer_t> input;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, input.out()));
  Ref<iree_hal_buffer_t> intermediate;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, intermediate.out()));
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, output.out()));

  // Record: fill → barrier → dispatch A → barrier → dispatch B.
  Ref<iree_hal_command_buffer_t> cmd;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      IREE_HAL_QUEUE_AFFINITY_ANY, /*binding_capacity=*/3, cmd.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd));

  // Fill input with a recognizable pattern via update_buffer.
  // We use update_buffer because it embeds host data directly.
  std::vector<uint32_t> fill_data = {10, 20, 30, 40};
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      cmd, fill_data.data(), /*source_offset=*/0,
      iree_hal_make_buffer_ref(input, 0, kBufferSize),
      IREE_HAL_UPDATE_FLAG_NONE));
  RecordBarrier(cmd);

  // Dispatch A: intermediate = input * 3 + 0 = [30, 60, 90, 120]
  RecordScaleAndOffset(cmd, /*scale=*/3, /*offset=*/0, /*input_slot=*/0,
                       /*output_slot=*/1);
  RecordBarrier(cmd);

  // Dispatch B: output = intermediate * 1 + 7 = [37, 67, 97, 127]
  RecordScaleAndOffset(cmd, /*scale=*/1, /*offset=*/7, /*input_slot=*/1,
                       /*output_slot=*/2);
  RecordBarrier(cmd);

  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd));

  iree_hal_buffer_binding_t table[3] = {
      {input, 0, kBufferSize},
      {intermediate, 0, kBufferSize},
      {output, 0, kBufferSize},
  };
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      cmd, iree_hal_buffer_binding_table_t{3, table}));

  auto data = ReadBufferData<uint32_t>(output);
  EXPECT_THAT(data, ContainerEq(std::vector<uint32_t>{37, 67, 97, 127}));
}

// Full transient lifecycle pipeline: alloca → fill → dispatch → readback
// → dealloca. The transient buffer is used as both the dispatch input
// (filled from host data) and scratch space. Results go to a persistent
// output buffer.
TEST_P(DispatchPipelineTest, TransientInputPipeline) {
  // Allocate a transient buffer for input.
  SemaphoreList alloca_signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  iree_hal_buffer_params_t alloca_params = {0};
  alloca_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  alloca_params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* raw = nullptr;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, alloca_signal,
      /*pool=*/NULL, alloca_params, kBufferSize, IREE_HAL_ALLOCA_FLAG_NONE,
      &raw));
  Ref<iree_hal_buffer_t> transient_input(raw);

  // Wait for alloca, then fill the transient input.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      alloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> input_data = {5, 10, 15, 20};
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device_, input_data.data(), transient_input, 0, kBufferSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  // Persistent output buffer.
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, output.out()));

  // Dispatch: output = transient_input * 4 + 1 = [21, 41, 61, 81]
  Ref<iree_hal_command_buffer_t> cmd;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, cmd.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd));
  RecordScaleAndOffset(cmd, /*scale=*/4, /*offset=*/1, /*input_slot=*/0,
                       /*output_slot=*/1);
  RecordBarrier(cmd);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd));

  iree_hal_buffer_binding_t table[2] = {
      {transient_input, 0, kBufferSize},
      {output, 0, kBufferSize},
  };
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(
      cmd, iree_hal_buffer_binding_table_t{2, table}));

  // Verify output (persistent buffer — no transient access issues).
  auto data = ReadBufferData<uint32_t>(output);
  EXPECT_THAT(data, ContainerEq(std::vector<uint32_t>{21, 41, 61, 81}));

  // Dealloca after readback.
  SemaphoreList dealloca_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dealloca_signal,
      transient_input, IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

// Reusable command buffer pipeline: records a two-stage dispatch chain once,
// then re-submits with different input data via binding tables. Verifies
// each submission produces correct results independently.
TEST_P(DispatchPipelineTest, ReusablePipelineWithDifferentInputs) {
  // Record a reusable two-stage pipeline.
  // Stage 1: intermediate = input * 2 + 0
  // Stage 2: output = intermediate * 1 + 100
  // Net: output = input * 2 + 100
  Ref<iree_hal_command_buffer_t> cmd;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/3, cmd.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd));
  RecordScaleAndOffset(cmd, 2, 0, 0, 1);
  RecordBarrier(cmd);
  RecordScaleAndOffset(cmd, 1, 100, 1, 2);
  RecordBarrier(cmd);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd));

  // Submit with different inputs.
  struct TestCase {
    std::vector<uint32_t> input;
    std::vector<uint32_t> expected;
  };
  TestCase cases[] = {
      {{1, 2, 3, 4}, {102, 104, 106, 108}},
      {{10, 20, 30, 40}, {120, 140, 160, 180}},
      {{0, 0, 0, 0}, {100, 100, 100, 100}},
  };

  for (size_t i = 0; i < IREE_ARRAYSIZE(cases); ++i) {
    Ref<iree_hal_buffer_t> input;
    IREE_ASSERT_OK(CreateDeviceBufferWithData(cases[i].input.data(),
                                              kBufferSize, input.out()));
    Ref<iree_hal_buffer_t> intermediate;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, intermediate.out()));
    Ref<iree_hal_buffer_t> output;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, output.out()));

    iree_hal_buffer_binding_t table[3] = {
        {input, 0, kBufferSize},
        {intermediate, 0, kBufferSize},
        {output, 0, kBufferSize},
    };

    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, cmd,
        iree_hal_buffer_binding_table_t{3, table}, IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));

    auto data = ReadBufferData<uint32_t>(output);
    EXPECT_THAT(data, ContainerEq(cases[i].expected))
        << "Mismatch on test case " << i;
  }
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(DispatchPipelineTest);

}  // namespace iree::hal::cts
