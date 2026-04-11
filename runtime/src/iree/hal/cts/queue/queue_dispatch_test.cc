// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HAL queue_dispatch operations.

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class QueueDispatchTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(executable_format());
    executable_params.executable_data = executable_data(iree_make_cstring_view(
        "command_buffer_dispatch_constants_bindings_test.bin"));

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

static void MakeScaleAndOffsetBindings(iree_hal_buffer_t* input_buffer,
                                       iree_hal_buffer_t* output_buffer,
                                       iree_hal_buffer_ref_t binding_refs[2]) {
  binding_refs[0] = iree_hal_make_buffer_ref(
      input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
  binding_refs[1] = iree_hal_make_buffer_ref(
      output_buffer, /*offset=*/0, iree_hal_buffer_byte_length(output_buffer));
}

// Dispatches scale_and_offset directly on the queue:
// output[i] = input[i] * scale + offset.
TEST_P(QueueDispatchTest, DispatchWithConstantsAndBindings) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

// Zero-workgroup dispatches are no-ops that still participate in semaphore
// ordering and signal completion.
TEST_P(QueueDispatchTest, NoopDispatchSignalsAndDoesNotTouchBuffers) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                           output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList empty_wait;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(0, 0, 0), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{99, 99, 99, 99}));
}

// Deferred zero-workgroup dispatches still wait for their dependency before
// signaling completion, but never execute the kernel body.
TEST_P(QueueDispatchTest, DeferredNoopDispatch) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateFilledDeviceBuffer(4 * sizeof(uint32_t), uint32_t{99},
                           output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_wait(device_, {0}, {1});
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, dispatch_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(0, 0, 0), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));

  uint64_t dispatch_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(dispatch_signal.semaphores[0], &dispatch_value));
  EXPECT_EQ(0u, dispatch_value);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(dispatch_wait.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{99, 99, 99, 99}));
}

// Wait-before-signal dispatches must be queued without head-of-line blocking
// and replayed after the wait semaphore advances.
TEST_P(QueueDispatchTest, DeferredWaitBeforeSignalDispatch) {
  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    CreateDeviceBufferWithData(input_data.data(),
                               input_data.size() * sizeof(input_data[0]),
                               input_buffer.out());
  }

  Ref<iree_hal_buffer_t> output_buffer;
  CreateZeroedDeviceBuffer(4 * sizeof(uint32_t), output_buffer.out());

  iree_hal_buffer_ref_t binding_refs[2];
  MakeScaleAndOffsetBindings(input_buffer, output_buffer, binding_refs);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  const uint32_t constant_data[] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));

  SemaphoreList dispatch_wait(device_, {0}, {1});
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, dispatch_wait, dispatch_signal,
      executable_, /*export_ordinal=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));

  uint64_t dispatch_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(dispatch_signal.semaphores[0], &dispatch_value));
  EXPECT_EQ(0u, dispatch_value);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(dispatch_wait.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<uint32_t>{13, 16, 19, 22}));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(QueueDispatchTest);

}  // namespace iree::hal::cts
