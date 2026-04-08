// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for queue-ordered transfer operations: queue_fill, queue_update,
// queue_copy. These are the "glue code" operations used outside command buffers
// for staging data, initializing buffers, and copying between buffers.
//
// Each operation is queue-ordered via semaphores and executes without a
// command buffer. The tests verify data correctness, offset handling, pattern
// sizes, and semaphore ordering between chained operations.

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;
using ::testing::Each;

class QueueTransferTest : public CtsTestBase<> {
 protected:
  // Submits a queue_fill and waits for completion.
  void QueueFillAndWait(iree_hal_buffer_t* target_buffer,
                        iree_device_size_t target_offset,
                        iree_device_size_t length, const void* pattern,
                        iree_host_size_t pattern_length) {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_fill(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, target_buffer,
        target_offset, length, pattern, pattern_length,
        IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Submits a queue_update and waits for completion.
  void QueueUpdateAndWait(const void* source_buffer,
                          iree_host_size_t source_offset,
                          iree_hal_buffer_t* target_buffer,
                          iree_device_size_t target_offset,
                          iree_device_size_t length) {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_update(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, source_buffer,
        source_offset, target_buffer, target_offset, length,
        IREE_HAL_UPDATE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  // Submits a queue_copy and waits for completion.
  void QueueCopyAndWait(iree_hal_buffer_t* source_buffer,
                        iree_device_size_t source_offset,
                        iree_hal_buffer_t* target_buffer,
                        iree_device_size_t target_offset,
                        iree_device_size_t length) {
    SemaphoreList signal(device_, {0}, {1});
    SemaphoreList empty_wait;
    IREE_ASSERT_OK(iree_hal_device_queue_copy(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal, source_buffer,
        source_offset, target_buffer, target_offset, length,
        IREE_HAL_COPY_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));
  }
};

//===----------------------------------------------------------------------===//
// queue_fill tests
//===----------------------------------------------------------------------===//

// Fills an entire buffer with a 1-byte pattern.
TEST_P(QueueTransferTest, FillEntireBuffer_1Byte) {
  const iree_device_size_t buffer_size = 1024;
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  uint8_t pattern = 0xAB;
  QueueFillAndWait(buffer, 0, buffer_size, &pattern, sizeof(pattern));

  auto data = ReadBufferData<uint8_t>(buffer);
  EXPECT_EQ(data.size(), buffer_size);
  EXPECT_THAT(data, Each(0xAB));
}

// Fills an entire buffer with a 2-byte pattern.
TEST_P(QueueTransferTest, FillEntireBuffer_2Byte) {
  const iree_device_size_t buffer_size = 1024;
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  uint16_t pattern = 0xBEEF;
  QueueFillAndWait(buffer, 0, buffer_size, &pattern, sizeof(pattern));

  auto data = ReadBufferData<uint16_t>(buffer);
  EXPECT_EQ(data.size(), buffer_size / sizeof(uint16_t));
  EXPECT_THAT(data, Each(0xBEEF));
}

// Fills an entire buffer with a 4-byte pattern.
TEST_P(QueueTransferTest, FillEntireBuffer_4Byte) {
  const iree_device_size_t buffer_size = 4096;
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  uint32_t pattern = 0xDEADCAFE;
  QueueFillAndWait(buffer, 0, buffer_size, &pattern, sizeof(pattern));

  auto data = ReadBufferData<uint32_t>(buffer);
  EXPECT_EQ(data.size(), buffer_size / sizeof(uint32_t));
  EXPECT_THAT(data, Each(0xDEADCAFE));
}

// Fills a subrange of a buffer, verifying boundaries are untouched.
TEST_P(QueueTransferTest, FillSubrange) {
  const iree_device_size_t buffer_size = 256;
  const iree_device_size_t fill_offset = 64;
  const iree_device_size_t fill_length = 128;
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  uint32_t pattern = 0xCAFEF00D;
  QueueFillAndWait(buffer, fill_offset, fill_length, &pattern, sizeof(pattern));

  // Verify the filled region.
  auto filled = ReadBufferBytes(buffer, fill_offset, fill_length);
  for (iree_device_size_t i = 0; i + 3 < fill_length; i += 4) {
    uint32_t value;
    memcpy(&value, filled.data() + i, sizeof(value));
    EXPECT_EQ(value, 0xCAFEF00D) << "Mismatch at offset " << (fill_offset + i);
  }

  // Verify the regions before and after the fill are still zero.
  auto before = ReadBufferBytes(buffer, 0, fill_offset);
  EXPECT_THAT(before, Each(0x00)) << "Data before fill region was modified";

  auto after = ReadBufferBytes(buffer, fill_offset + fill_length,
                               buffer_size - fill_offset - fill_length);
  EXPECT_THAT(after, Each(0x00)) << "Data after fill region was modified";
}

// Fills a large buffer to exercise non-trivial transfer sizes.
TEST_P(QueueTransferTest, FillLargeBuffer) {
  const iree_device_size_t buffer_size = 256 * 1024;  // 256KB
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  uint32_t pattern = 0x12345678;
  QueueFillAndWait(buffer, 0, buffer_size, &pattern, sizeof(pattern));

  // Spot-check: first element, last element, middle element.
  auto data = ReadBufferData<uint32_t>(buffer);
  EXPECT_EQ(data.front(), 0x12345678);
  EXPECT_EQ(data.back(), 0x12345678);
  EXPECT_EQ(data[data.size() / 2], 0x12345678);
  EXPECT_THAT(data, Each(0x12345678));
}

TEST_P(QueueTransferTest, FillSizeAlignmentAndPatternClasses) {
  struct FillCase {
    iree_host_size_t pattern_length = 0;
    iree_device_size_t target_offset = 0;
    iree_device_size_t fill_length = 0;
    uint32_t pattern = 0;
  };
  const FillCase cases[] = {
      {/*.pattern_length=*/1, /*.target_offset=*/0,
       /*.fill_length=*/4, /*.pattern=*/0x000000A5u},
      {/*.pattern_length=*/1, /*.target_offset=*/3,
       /*.fill_length=*/31, /*.pattern=*/0x0000005Au},
      {/*.pattern_length=*/1, /*.target_offset=*/17,
       /*.fill_length=*/32, /*.pattern=*/0x000000C3u},
      {/*.pattern_length=*/1, /*.target_offset=*/1,
       /*.fill_length=*/33, /*.pattern=*/0x0000003Cu},
      {/*.pattern_length=*/1, /*.target_offset=*/16,
       /*.fill_length=*/64, /*.pattern=*/0x000000D7u},
      {/*.pattern_length=*/1, /*.target_offset=*/7,
       /*.fill_length=*/1024, /*.pattern=*/0x00000019u},
      {/*.pattern_length=*/1, /*.target_offset=*/31,
       /*.fill_length=*/64 * 1024, /*.pattern=*/0x000000E1u},
      {/*.pattern_length=*/2, /*.target_offset=*/0,
       /*.fill_length=*/4, /*.pattern=*/0x0000BEEFu},
      {/*.pattern_length=*/2, /*.target_offset=*/2,
       /*.fill_length=*/30, /*.pattern=*/0x0000CAFEu},
      {/*.pattern_length=*/2, /*.target_offset=*/6,
       /*.fill_length=*/32, /*.pattern=*/0x00001234u},
      {/*.pattern_length=*/2, /*.target_offset=*/18,
       /*.fill_length=*/34, /*.pattern=*/0x0000A1B2u},
      {/*.pattern_length=*/2, /*.target_offset=*/4,
       /*.fill_length=*/1024, /*.pattern=*/0x00000F0Eu},
      {/*.pattern_length=*/2, /*.target_offset=*/14,
       /*.fill_length=*/64 * 1024, /*.pattern=*/0x000055AAu},
      {/*.pattern_length=*/4, /*.target_offset=*/0,
       /*.fill_length=*/4, /*.pattern=*/0xDEADCAFEu},
      {/*.pattern_length=*/4, /*.target_offset=*/4,
       /*.fill_length=*/28, /*.pattern=*/0xCAFEF00Du},
      {/*.pattern_length=*/4, /*.target_offset=*/8,
       /*.fill_length=*/32, /*.pattern=*/0x12345678u},
      {/*.pattern_length=*/4, /*.target_offset=*/20,
       /*.fill_length=*/36, /*.pattern=*/0xA5A55A5Au},
      {/*.pattern_length=*/4, /*.target_offset=*/4,
       /*.fill_length=*/1024, /*.pattern=*/0x0F1E2D3Cu},
      {/*.pattern_length=*/4, /*.target_offset=*/12,
       /*.fill_length=*/64 * 1024, /*.pattern=*/0x55AA33CCu},
  };

  for (const FillCase& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "pattern_length=" << test_case.pattern_length
                 << " target_offset=" << test_case.target_offset
                 << " fill_length=" << test_case.fill_length);
    ASSERT_EQ(test_case.target_offset % test_case.pattern_length, 0);
    ASSERT_EQ(test_case.fill_length % test_case.pattern_length, 0);

    const iree_device_size_t buffer_size =
        test_case.target_offset + test_case.fill_length + 16;
    Ref<iree_hal_buffer_t> buffer;
    CreateZeroedDeviceBuffer(buffer_size, buffer.out());

    QueueFillAndWait(buffer, test_case.target_offset, test_case.fill_length,
                     &test_case.pattern, test_case.pattern_length);

    auto data = ReadBufferBytes(buffer, 0, buffer_size);
    auto expected = MakeFilledBytes(buffer_size, test_case.target_offset,
                                    test_case.fill_length, test_case.pattern,
                                    test_case.pattern_length);
    EXPECT_THAT(data, ContainerEq(expected));
  }
}

//===----------------------------------------------------------------------===//
// queue_update tests
//===----------------------------------------------------------------------===//

// Updates an entire buffer from host data.
TEST_P(QueueTransferTest, UpdateEntireBuffer) {
  const iree_device_size_t element_count = 64;
  const iree_device_size_t buffer_size = element_count * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  // Create host data with a recognizable sequence.
  std::vector<uint32_t> host_data(element_count);
  std::iota(host_data.begin(), host_data.end(), 100u);

  QueueUpdateAndWait(host_data.data(), 0, buffer, 0, buffer_size);

  auto readback = ReadBufferData<uint32_t>(buffer);
  EXPECT_THAT(readback, ContainerEq(host_data));
}

// Updates a subrange of a buffer, verifying boundaries are untouched.
TEST_P(QueueTransferTest, UpdateSubrange) {
  const iree_device_size_t buffer_size = 256;
  const iree_device_size_t update_offset = 32;
  const iree_device_size_t update_length = 64;
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  // Source data: sequential bytes.
  std::vector<uint8_t> source(update_length);
  std::iota(source.begin(), source.end(), static_cast<uint8_t>(1));

  QueueUpdateAndWait(source.data(), 0, buffer, update_offset, update_length);

  // Verify the updated region matches source.
  auto updated = ReadBufferBytes(buffer, update_offset, update_length);
  EXPECT_THAT(updated, ContainerEq(source));

  // Verify boundaries are untouched.
  auto before = ReadBufferBytes(buffer, 0, update_offset);
  EXPECT_THAT(before, Each(0x00));

  auto after = ReadBufferBytes(buffer, update_offset + update_length,
                               buffer_size - update_offset - update_length);
  EXPECT_THAT(after, Each(0x00));
}

// Updates from a host buffer with a source offset.
TEST_P(QueueTransferTest, UpdateWithSourceOffset) {
  const iree_device_size_t buffer_size = 64;
  Ref<iree_hal_buffer_t> buffer;
  CreateZeroedDeviceBuffer(buffer_size, buffer.out());

  // Source has a header we want to skip.
  std::vector<uint8_t> source(128);
  std::iota(source.begin(), source.end(), static_cast<uint8_t>(0));

  const iree_host_size_t source_offset = 64;
  QueueUpdateAndWait(source.data(), source_offset, buffer, 0, buffer_size);

  auto readback = ReadBufferBytes(buffer, 0, buffer_size);
  std::vector<uint8_t> expected(source.begin() + source_offset,
                                source.begin() + source_offset + buffer_size);
  EXPECT_THAT(readback, ContainerEq(expected));
}

//===----------------------------------------------------------------------===//
// queue_copy tests
//===----------------------------------------------------------------------===//

// Copies an entire buffer from source to destination.
TEST_P(QueueTransferTest, CopyEntireBuffer) {
  const iree_device_size_t buffer_size = 512;
  Ref<iree_hal_buffer_t> source;
  CreateFilledDeviceBuffer<uint32_t>(buffer_size, 0xAAAAAAAAu, source.out());
  Ref<iree_hal_buffer_t> target;
  CreateZeroedDeviceBuffer(buffer_size, target.out());

  QueueCopyAndWait(source, 0, target, 0, buffer_size);

  auto data = ReadBufferData<uint32_t>(target);
  EXPECT_THAT(data, Each(0xAAAAAAAAu));
}

// Copies a subrange with both source and target offsets.
TEST_P(QueueTransferTest, CopyWithOffsets) {
  const iree_device_size_t buffer_size = 256;
  const iree_device_size_t source_offset = 64;
  const iree_device_size_t target_offset = 128;
  const iree_device_size_t copy_length = 64;

  // Source: sequential uint32 values.
  std::vector<uint32_t> source_data(buffer_size / sizeof(uint32_t));
  std::iota(source_data.begin(), source_data.end(), 0u);
  Ref<iree_hal_buffer_t> source;
  CreateDeviceBufferWithData(source_data.data(), buffer_size, source.out());

  Ref<iree_hal_buffer_t> target;
  CreateZeroedDeviceBuffer(buffer_size, target.out());

  QueueCopyAndWait(source, source_offset, target, target_offset, copy_length);

  // Verify the copied region matches the source subrange.
  auto copied = ReadBufferBytes(target, target_offset, copy_length);
  std::vector<uint8_t> expected(copy_length);
  memcpy(expected.data(),
         reinterpret_cast<const uint8_t*>(source_data.data()) + source_offset,
         copy_length);
  EXPECT_THAT(copied, ContainerEq(expected));

  // Verify boundaries are untouched.
  auto before = ReadBufferBytes(target, 0, target_offset);
  EXPECT_THAT(before, Each(0x00));
  auto after = ReadBufferBytes(target, target_offset + copy_length,
                               buffer_size - target_offset - copy_length);
  EXPECT_THAT(after, Each(0x00));
}

// Copies a large buffer to exercise non-trivial DMA paths.
TEST_P(QueueTransferTest, CopyLargeBuffer) {
  const iree_device_size_t buffer_size = 256 * 1024;  // 256KB
  Ref<iree_hal_buffer_t> source;
  CreateFilledDeviceBuffer<uint32_t>(buffer_size, 0xFEEDFACEu, source.out());
  Ref<iree_hal_buffer_t> target;
  CreateZeroedDeviceBuffer(buffer_size, target.out());

  QueueCopyAndWait(source, 0, target, 0, buffer_size);

  auto data = ReadBufferData<uint32_t>(target);
  EXPECT_EQ(data.front(), 0xFEEDFACEu);
  EXPECT_EQ(data.back(), 0xFEEDFACEu);
  EXPECT_THAT(data, Each(0xFEEDFACEu));
}

TEST_P(QueueTransferTest, CopySizeAndAlignmentClasses) {
  struct AlignmentCase {
    const char* name = nullptr;
    iree_device_size_t source_offset = 0;
    iree_device_size_t target_offset = 0;
  };
  const AlignmentCase alignment_cases[] = {
      {/*.name=*/"aligned16", /*.source_offset=*/0, /*.target_offset=*/0},
      {/*.name=*/"aligned8_not16", /*.source_offset=*/8, /*.target_offset=*/8},
      {/*.name=*/"aligned4_not8", /*.source_offset=*/4, /*.target_offset=*/4},
      {/*.name=*/"byte_misaligned", /*.source_offset=*/1, /*.target_offset=*/2},
  };
  const iree_device_size_t common_sizes[] = {
      4, 8, 16, 31, 32, 33, 64, 128, 256, 1024, 4 * 1024, 16 * 1024, 64 * 1024,
  };

  auto run_case = [&](const char* name, iree_device_size_t source_offset,
                      iree_device_size_t target_offset,
                      iree_device_size_t length) {
    SCOPED_TRACE(::testing::Message()
                 << name << " source_offset=" << source_offset
                 << " target_offset=" << target_offset << " length=" << length);
    iree_device_size_t buffer_size = source_offset + length;
    if (target_offset + length > buffer_size) {
      buffer_size = target_offset + length;
    }
    buffer_size += 16;

    std::vector<uint8_t> source_data = MakeDeterministicBytes(buffer_size);
    Ref<iree_hal_buffer_t> source;
    CreateDeviceBufferWithData(source_data.data(), buffer_size, source.out());
    Ref<iree_hal_buffer_t> target;
    CreateZeroedDeviceBuffer(buffer_size, target.out());

    QueueCopyAndWait(source, source_offset, target, target_offset, length);

    std::vector<uint8_t> expected(buffer_size, 0);
    memcpy(expected.data() + target_offset, source_data.data() + source_offset,
           length);
    auto data = ReadBufferBytes(target, 0, buffer_size);
    EXPECT_THAT(data, ContainerEq(expected));
  };

  for (iree_device_size_t length : common_sizes) {
    for (const AlignmentCase& alignment_case : alignment_cases) {
      run_case(alignment_case.name, alignment_case.source_offset,
               alignment_case.target_offset, length);
    }
  }
  run_case("aligned16_mib", 0, 0, 1024 * 1024);
}

//===----------------------------------------------------------------------===//
// Chained queue operations (semaphore ordering)
//===----------------------------------------------------------------------===//

// Chains fill → copy via semaphores: fill source, then copy to target.
// The copy waits on the fill's signal semaphore without host intervention.
TEST_P(QueueTransferTest, ChainedFillThenCopy) {
  const iree_device_size_t buffer_size = 512;
  Ref<iree_hal_buffer_t> source;
  CreateZeroedDeviceBuffer(buffer_size, source.out());
  Ref<iree_hal_buffer_t> target;
  CreateZeroedDeviceBuffer(buffer_size, target.out());

  // Fill signals semaphore at value 1.
  SemaphoreList fill_signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  uint32_t pattern = 0x11223344;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal, source, 0,
      buffer_size, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Copy waits on fill completion (value 1), signals at value 1 on its own
  // semaphore.
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_signal, copy_signal, source, 0,
      target, 0, buffer_size, IREE_HAL_COPY_FLAG_NONE));

  // Only wait on the copy's signal — if ordering is correct, the fill has
  // completed by the time the copy reads from source.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  auto data = ReadBufferData<uint32_t>(target);
  EXPECT_THAT(data, Each(0x11223344u));
}

// Chains update → copy via semaphores: update source from host, then copy
// to target.
TEST_P(QueueTransferTest, ChainedUpdateThenCopy) {
  const iree_device_size_t element_count = 32;
  const iree_device_size_t buffer_size = element_count * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> source;
  CreateZeroedDeviceBuffer(buffer_size, source.out());
  Ref<iree_hal_buffer_t> target;
  CreateZeroedDeviceBuffer(buffer_size, target.out());

  std::vector<uint32_t> host_data(element_count);
  std::iota(host_data.begin(), host_data.end(), 42u);

  // Update signals at value 1.
  SemaphoreList update_signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, update_signal,
      host_data.data(), 0, source, 0, buffer_size, IREE_HAL_UPDATE_FLAG_NONE));

  // Copy waits on update, signals its own semaphore.
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, update_signal, copy_signal, source,
      0, target, 0, buffer_size, IREE_HAL_COPY_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  auto readback = ReadBufferData<uint32_t>(target);
  EXPECT_THAT(readback, ContainerEq(host_data));
}

// Chains three operations: fill → copy → fill, all ordered by semaphores.
// Verifies multi-step pipeline ordering works correctly.
TEST_P(QueueTransferTest, ChainedFillCopyFill) {
  const iree_device_size_t buffer_size = 256;
  Ref<iree_hal_buffer_t> buffer_a;
  CreateZeroedDeviceBuffer(buffer_size, buffer_a.out());
  Ref<iree_hal_buffer_t> buffer_b;
  CreateZeroedDeviceBuffer(buffer_size, buffer_b.out());

  SemaphoreList empty_wait;

  // Step 1: fill buffer_a with 0xAA.
  SemaphoreList step1_signal(device_, {0}, {1});
  uint8_t pattern1 = 0xAA;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, step1_signal, buffer_a,
      0, buffer_size, &pattern1, sizeof(pattern1), IREE_HAL_FILL_FLAG_NONE));

  // Step 2: copy buffer_a → buffer_b (waits on step 1).
  SemaphoreList step2_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, step1_signal, step2_signal,
      buffer_a, 0, buffer_b, 0, buffer_size, IREE_HAL_COPY_FLAG_NONE));

  // Step 3: fill buffer_a with 0xBB (waits on step 2, so copy has read a).
  SemaphoreList step3_signal(device_, {0}, {1});
  uint8_t pattern3 = 0xBB;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, step2_signal, step3_signal,
      buffer_a, 0, buffer_size, &pattern3, sizeof(pattern3),
      IREE_HAL_FILL_FLAG_NONE));

  // Wait for the full chain.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      step3_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  // buffer_b should have 0xAA (from the copy in step 2).
  auto data_b = ReadBufferData<uint8_t>(buffer_b);
  EXPECT_THAT(data_b, Each(0xAA));

  // buffer_a should have 0xBB (from the fill in step 3).
  auto data_a = ReadBufferData<uint8_t>(buffer_a);
  EXPECT_THAT(data_a, Each(0xBB));
}

//===----------------------------------------------------------------------===//
// queue_barrier tests
//===----------------------------------------------------------------------===//

// Enqueues a pure barrier (no data movement) and verifies it signals.
TEST_P(QueueTransferTest, BarrierSignals) {
  SemaphoreList signal(device_, {0}, {1});
  SemaphoreList empty_wait;
  IREE_ASSERT_OK(iree_hal_device_queue_barrier(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, signal,
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));
}

// Chains fill → barrier → copy to verify barrier preserves ordering.
TEST_P(QueueTransferTest, BarrierPreservesOrdering) {
  const iree_device_size_t buffer_size = 256;
  Ref<iree_hal_buffer_t> source;
  CreateZeroedDeviceBuffer(buffer_size, source.out());
  Ref<iree_hal_buffer_t> target;
  CreateZeroedDeviceBuffer(buffer_size, target.out());

  SemaphoreList empty_wait;

  // Fill source.
  SemaphoreList fill_signal(device_, {0}, {1});
  uint32_t pattern = 0xBAADF00D;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_wait, fill_signal, source, 0,
      buffer_size, &pattern, sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Barrier between fill and copy.
  SemaphoreList barrier_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_barrier(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, fill_signal, barrier_signal,
      IREE_HAL_EXECUTE_FLAG_NONE));

  // Copy waits on barrier.
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, barrier_signal, copy_signal, source,
      0, target, 0, buffer_size, IREE_HAL_COPY_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_make_timeout_ms(5000), IREE_ASYNC_WAIT_FLAG_NONE));

  auto data = ReadBufferData<uint32_t>(target);
  EXPECT_THAT(data, Each(0xBAADF00Du));
}

CTS_REGISTER_TEST_SUITE(QueueTransferTest);

}  // namespace iree::hal::cts
