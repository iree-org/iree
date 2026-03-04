// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

class CommandBufferFillBufferTest : public CtsTestBase<> {
 protected:
  // Runs a fill buffer test using the current recording mode.
  // In direct mode: uses inline buffer references (binding_capacity = 0).
  // In indirect mode: uses binding table slots (binding_capacity = 1).
  // Results are written to |out_data|, which is resized to |buffer_size|.
  void RunFillBufferTest(iree_device_size_t buffer_size,
                         iree_device_size_t target_offset,
                         iree_device_size_t fill_length, const void* pattern,
                         iree_host_size_t pattern_length,
                         std::vector<uint8_t>& out_data) {
    iree_hal_buffer_t* device_buffer = NULL;
    CreateZeroedDeviceBuffer(buffer_size, &device_buffer);

    const bool indirect = recording_mode() == RecordingMode::kIndirect;
    const iree_host_size_t binding_capacity = indirect ? 1 : 0;

    iree_hal_command_buffer_t* command_buffer = NULL;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        binding_capacity, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

    iree_hal_buffer_ref_t target_ref;
    if (indirect) {
      target_ref = iree_hal_make_indirect_buffer_ref(
          /*binding=*/0, target_offset, fill_length);
    } else {
      target_ref =
          iree_hal_make_buffer_ref(device_buffer, target_offset, fill_length);
    }

    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, target_ref, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    if (indirect) {
      const iree_hal_buffer_binding_t bindings[] = {
          {device_buffer, 0, IREE_HAL_WHOLE_BUFFER},
      };
      IREE_ASSERT_OK(SubmitCommandBufferAndWait(
          command_buffer,
          iree_hal_buffer_binding_table_t{IREE_ARRAYSIZE(bindings), bindings}));
    } else {
      IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
    }

    out_data.resize(buffer_size);
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device_, device_buffer, /*source_offset=*/0, out_data.data(),
        buffer_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(device_buffer);
  }
};

TEST_P(CommandBufferFillBufferTest, Pattern1_Size1_Offset0_Length1) {
  iree_device_size_t buffer_size = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/1, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern1_Size5_Offset0_Length5) {
  iree_device_size_t buffer_size = 5;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/5, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern1_Size16_Offset0_Length1) {
  iree_device_size_t buffer_size = 16;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/1, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern1_Size16_Offset0_Length3) {
  iree_device_size_t buffer_size = 16;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/3, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern1_Size16_Offset0_Length8) {
  iree_device_size_t buffer_size = 16;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/8, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern1_Size16_Offset2_Length8) {
  iree_device_size_t buffer_size = 16;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/2,
                    /*fill_length=*/8, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern2_Size2_Offset0_Length2) {
  iree_device_size_t buffer_size = 2;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/2, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern2_Size16_Offset0_Length8) {
  iree_device_size_t buffer_size = 16;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/8, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern2_Size16_Offset0_Length10) {
  iree_device_size_t buffer_size = 16;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/10, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern2_Size16_Offset2_Length8) {
  iree_device_size_t buffer_size = 16;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/2,
                    /*fill_length=*/8, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern4_Size4_Offset0_Length4) {
  iree_device_size_t buffer_size = 4;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/4, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern4_Size4_Offset16_Length4) {
  iree_device_size_t buffer_size = 20;
  iree_device_size_t target_offset = 16;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer(buffer_size, 0);
  *reinterpret_cast<uint32_t*>(&reference_buffer[target_offset]) = pattern;
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, target_offset, /*fill_length=*/4, &pattern,
                    sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern4_Size16_Offset0_Length8) {
  iree_device_size_t buffer_size = 16;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/0,
                    /*fill_length=*/8, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

TEST_P(CommandBufferFillBufferTest, Pattern4_Size16_Offset8_Length8) {
  iree_device_size_t buffer_size = 16;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_data;
  RunFillBufferTest(buffer_size, /*target_offset=*/8,
                    /*fill_length=*/8, &pattern, sizeof(pattern), actual_data);
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferFillBufferTest);

}  // namespace iree::hal::cts
