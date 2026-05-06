// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class VulkanQueueDescriptorCacheTest : public CtsTestBase<> {
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

TEST_P(VulkanQueueDescriptorCacheTest, DeferredUnalignedFillsExceedOneBlock) {
  constexpr iree_host_size_t kSubmissionCount = 5000;
  constexpr uint8_t kPattern = 0x5A;

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kSubmissionCount, target_buffer.out()));

  SemaphoreList gate(device_, {0}, {1});
  std::vector<uint64_t> initial_values(kSubmissionCount, 0);
  std::vector<uint64_t> payload_values(kSubmissionCount, 1);
  SemaphoreList signals(device_, std::move(initial_values),
                        std::move(payload_values));

  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &signals.semaphores[i],
        .payload_values = &signals.payload_values[i],
    };
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, gate, signal_list,
        target_buffer.get(), i, /*length=*/1, &kPattern, sizeof(kPattern),
        IREE_HAL_FILL_FLAG_NONE));
  }

  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate.semaphores[0], 1, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint8_t> bytes =
      ReadBufferBytes(target_buffer.get(), /*offset=*/0, kSubmissionCount);
  ASSERT_EQ(kSubmissionCount, bytes.size());
  for (uint8_t byte : bytes) {
    EXPECT_EQ(kPattern, byte);
  }
}

TEST_P(VulkanQueueDescriptorCacheTest, DeferredDispatchesExceedOneBlock) {
  constexpr iree_host_size_t kSubmissionCount = 5000;
  constexpr iree_host_size_t kElementCount = 4;
  constexpr uint32_t kScale = 3;
  constexpr uint32_t kOffset = 10;
  constexpr iree_device_size_t kDispatchByteLength =
      kElementCount * sizeof(uint32_t);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(
      kSubmissionCount * kDispatchByteLength, output_buffer.out()));

  SemaphoreList gate(device_, {0}, {1});
  std::vector<uint64_t> initial_values(kSubmissionCount, 0);
  std::vector<uint64_t> payload_values(kSubmissionCount, 1);
  SemaphoreList signals(device_, std::move(initial_values),
                        std::move(payload_values));

  const uint32_t constant_data[] = {kScale, kOffset};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));
  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                                 kDispatchByteLength),
        iree_hal_make_buffer_ref(output_buffer, i * kDispatchByteLength,
                                 kDispatchByteLength),
    };
    iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &signals.semaphores[i],
        .payload_values = &signals.payload_values[i],
    };
    IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, gate, signal_list, executable_,
        /*export_ordinal=*/0, iree_hal_make_static_dispatch_config(1, 1, 1),
        constants, bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  }

  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate.semaphores[0], 1, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  std::vector<uint32_t> expected(kSubmissionCount * kElementCount);
  const uint32_t expected_dispatch_data[] = {13, 16, 19, 22};
  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    std::copy(expected_dispatch_data,
              expected_dispatch_data + IREE_ARRAYSIZE(expected_dispatch_data),
              expected.begin() + i * kElementCount);
  }
  EXPECT_THAT(output_data, ContainerEq(expected));
}

TEST_P(VulkanQueueDescriptorCacheTest,
       DeferredCommandBufferDispatchesExceedOneBlock) {
  constexpr iree_host_size_t kSubmissionCount = 5000;
  constexpr iree_host_size_t kElementCount = 4;
  constexpr uint32_t kScale = 3;
  constexpr uint32_t kOffset = 10;
  constexpr iree_device_size_t kDispatchByteLength =
      kElementCount * sizeof(uint32_t);

  Ref<iree_hal_buffer_t> input_buffer;
  {
    std::vector<uint32_t> input_data = {1, 2, 3, 4};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        input_data.data(), input_data.size() * sizeof(input_data[0]),
        input_buffer.out()));
  }

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(
      kSubmissionCount * kDispatchByteLength, output_buffer.out()));

  const uint32_t constant_data[] = {kScale, kOffset};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_data, sizeof(constant_data));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                        kDispatchByteLength),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/0,
                                        kDispatchByteLength),
  };
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0,
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  SemaphoreList gate(device_, {0}, {1});
  std::vector<uint64_t> initial_values(kSubmissionCount, 0);
  std::vector<uint64_t> payload_values(kSubmissionCount, 1);
  SemaphoreList signals(device_, std::move(initial_values),
                        std::move(payload_values));

  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    iree_hal_buffer_binding_t binding_table_entries[2] = {
        {
            /*buffer=*/input_buffer,
            /*offset=*/0,
            /*length=*/kDispatchByteLength,
        },
        {
            /*buffer=*/output_buffer,
            /*offset=*/i * kDispatchByteLength,
            /*length=*/kDispatchByteLength,
        },
    };
    iree_hal_buffer_binding_table_t binding_table = {
        /*.count=*/IREE_ARRAYSIZE(binding_table_entries),
        /*.bindings=*/binding_table_entries,
    };
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &signals.semaphores[i],
        .payload_values = &signals.payload_values[i],
    };
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, gate, signal_list, command_buffer,
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  }

  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate.semaphores[0], 1, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint32_t> output_data = ReadBufferData<uint32_t>(output_buffer);
  std::vector<uint32_t> expected(kSubmissionCount * kElementCount);
  const uint32_t expected_dispatch_data[] = {13, 16, 19, 22};
  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    std::copy(expected_dispatch_data,
              expected_dispatch_data + IREE_ARRAYSIZE(expected_dispatch_data),
              expected.begin() + i * kElementCount);
  }
  EXPECT_THAT(output_data, ContainerEq(expected));
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(VulkanQueueDescriptorCacheTest);

}  // namespace iree::hal::cts
