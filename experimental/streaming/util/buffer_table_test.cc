// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/util/buffer_table.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Dummy buffer structure for testing.
// We only need the device_ptr field for the buffer table.
struct iree_hal_streaming_buffer_t {
  iree_hal_streaming_deviceptr_t device_ptr;
  void* host_ptr;
  size_t size;
  // Padding to make it more realistic.
  uint64_t padding[5];
};

namespace iree::hal::stream {
namespace {

using ::iree::testing::status::StatusIs;

// Helper to create a dummy buffer with the given device pointer.
static iree_hal_streaming_buffer_t* CreateDummyBuffer(
    iree_hal_streaming_deviceptr_t device_ptr, size_t size,
    iree_allocator_t allocator, void* host_ptr = nullptr) {
  iree_hal_streaming_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(
      iree_allocator_malloc(allocator, sizeof(*buffer), (void**)&buffer));
  buffer->device_ptr = device_ptr;
  buffer->host_ptr = host_ptr;
  buffer->size = size;
  return buffer;
}

// Helper to free a dummy buffer.
static void FreeDummyBuffer(iree_hal_streaming_buffer_t* buffer,
                            iree_allocator_t allocator) {
  iree_allocator_free(allocator, buffer);
}

//===----------------------------------------------------------------------===//
// Basic operations
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, AllocateAndFree) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
  EXPECT_NE(table, nullptr);

  // Free should be safe even with an empty table.
  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, FreeNull) {
  // Should be safe to free NULL.
  iree_hal_streaming_buffer_table_free(nullptr);
}

TEST(BufferTableTest, InsertSingle) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupExact) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, RemoveSingle) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL));

  // Should not be found after removal.
  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

//===----------------------------------------------------------------------===//
// Error conditions
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, DoubleInsert) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  auto* buffer2 = CreateDummyBuffer(0x100000000ULL, 8192, allocator);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));

  // Second insert with same device_ptr should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, buffer2)),
              StatusIs(StatusCode::kAlreadyExists));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
}

TEST(BufferTableTest, LookupMissing) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));
  EXPECT_EQ(found, nullptr);

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, RemoveMissing) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL)),
      StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, LookupRangeInvalidSize) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100000000ULL, 0, &found)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, LookupRangeOverflow) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;
  // Request a range that would overflow.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, UINT64_MAX - 10, 20, &found)),
              StatusIs(StatusCode::kInvalidArgument));

  iree_hal_streaming_buffer_table_free(table);
}

//===----------------------------------------------------------------------===//
// Multiple operations
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, InsertMultiple) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 100;
  std::vector<iree_hal_streaming_buffer_t*> buffers;

  // Insert multiple buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify all can be looked up.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, addr, &found));
    EXPECT_EQ(found, buffers[i]);
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RemoveFIFO) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 50;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Remove in FIFO order.
  for (size_t i = 0; i < count; ++i) {
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, addresses[i]));

    // Verify it's gone.
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[i], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RemoveLIFO) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 50;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Remove in LIFO order.
  for (size_t i = count; i > 0; --i) {
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_remove(table, addresses[i - 1]));

    // Verify it's gone.
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[i - 1], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, RemoveRandom) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 50;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert buffers.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 4096, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Create random removal order.
  std::vector<size_t> removal_order(count);
  std::iota(removal_order.begin(), removal_order.end(), 0);
  std::mt19937 gen(0x12345678);
  std::shuffle(removal_order.begin(), removal_order.end(), gen);

  // Remove in random order.
  for (size_t index : removal_order) {
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_remove(table, addresses[index]));

    // Verify it's gone.
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[index], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

//===----------------------------------------------------------------------===//
// Range lookup tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, LookupRangeExact) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x8000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup exact range.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100000000ULL, 0x8000, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeWithin) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x8000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup range within buffer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100001000ULL, 0x2000, &found));
  EXPECT_EQ(found, buffer);

  // Lookup at the end of the buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100007000ULL, 0x1000, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeOutOfBounds) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x8000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  iree_hal_streaming_buffer_t* found = nullptr;

  // Range starts before buffer.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x0FFFFFFF00ULL, 0x1000, &found)),
              StatusIs(StatusCode::kNotFound));

  // Range extends past buffer end.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100007000ULL, 0x2000, &found)),
              StatusIs(StatusCode::kNotFound));

  // Range starts after buffer end.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100008000ULL, 0x1000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeMultipleBuffers) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  // Create adjacent buffers.
  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator);
  auto* buffer2 = CreateDummyBuffer(0x100004000ULL, 0x4000, allocator);
  auto* buffer3 = CreateDummyBuffer(0x100008000ULL, 0x4000, allocator);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer2));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer3));

  iree_hal_streaming_buffer_t* found = nullptr;

  // Range within first buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100001000ULL, 0x2000, &found));
  EXPECT_EQ(found, buffer1);

  // Range within second buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, 0x100005000ULL, 0x2000, &found));
  EXPECT_EQ(found, buffer2);

  // Range spanning two buffers should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100003000ULL, 0x2000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
  FreeDummyBuffer(buffer3, allocator);
}

//===----------------------------------------------------------------------===//
// Edge cases and stress tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, EmptyTableOperations) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  iree_hal_streaming_buffer_t* found = nullptr;

  // All operations should fail gracefully on empty table.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, 0x100000000ULL, 0x1000, &found)),
              StatusIs(StatusCode::kNotFound));

  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL)),
      StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
}

TEST(BufferTableTest, LargeScale) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 10000;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Generate random addresses.
  std::mt19937_64 gen(0xDEADBEEF);
  std::uniform_int_distribution<iree_hal_streaming_deviceptr_t> dist(
      0x100000000ULL, 0x700000000000ULL);

  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = dist(gen) & ~0xFULL;  // Align to 16.
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 0x1000, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify random lookups.
  std::uniform_int_distribution<size_t> index_dist(0, count - 1);
  for (size_t i = 0; i < 100; ++i) {
    size_t index = index_dist(gen);
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
        table, addresses[index], &found));
    EXPECT_EQ(found, buffers[index]);
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, SparseAddresses) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 100;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Create sparse addresses with exponentially growing gaps.
  iree_hal_streaming_deviceptr_t base = 0x100000000ULL;
  for (size_t i = 0; i < count; ++i) {
    base += 0x10000 * (i + 1);  // Gaps: 0x10000, 0x20000, 0x30000, etc.
    addresses.push_back(base);
    auto* buffer = CreateDummyBuffer(base, 0x1000, allocator);
    buffers.push_back(buffer);
    IREE_ASSERT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify all can be looked up.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
    EXPECT_EQ(found, buffers[i]);
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

TEST(BufferTableTest, InsertRemoveInsert) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 0x1000, allocator);
  auto* buffer2 = CreateDummyBuffer(0x100000000ULL, 0x2000, allocator);

  // Insert, remove, then insert again with same address.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, 0x100000000ULL));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer2));

  // Should find the second buffer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer2);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
}

TEST(BufferTableTest, MixedOperations) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  const size_t count = 100;
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  std::vector<iree_hal_streaming_deviceptr_t> addresses;

  // Insert initial set.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_deviceptr_t addr = 0x100000000ULL + i * 0x10000;
    addresses.push_back(addr);
    auto* buffer = CreateDummyBuffer(addr, 0x4000, allocator);
    buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Remove every other buffer.
  for (size_t i = 0; i < count; i += 2) {
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, addresses[i]));
  }

  // Verify remaining buffers can be found.
  for (size_t i = 1; i < count; i += 2) {
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
    EXPECT_EQ(found, buffers[i]);
  }

  // Verify removed buffers cannot be found.
  for (size_t i = 0; i < count; i += 2) {
    iree_hal_streaming_buffer_t* found = nullptr;
    EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                    table, addresses[i], &found)),
                StatusIs(StatusCode::kNotFound));
  }

  // Add new buffers at removed addresses.
  std::vector<iree_hal_streaming_buffer_t*> new_buffers;
  for (size_t i = 0; i < count; i += 2) {
    auto* buffer = CreateDummyBuffer(addresses[i], 0x8000, allocator);
    new_buffers.push_back(buffer);
    IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));
  }

  // Verify all buffers can now be found.
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_buffer_t* found = nullptr;
    IREE_EXPECT_OK(
        iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
    if (i % 2 == 0) {
      // Should find the new buffer.
      EXPECT_EQ(found, new_buffers[i / 2]);
    } else {
      // Should find the original buffer.
      EXPECT_EQ(found, buffers[i]);
    }
  }

  iree_hal_streaming_buffer_table_free(table);
  for (auto* buffer : buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
  for (auto* buffer : new_buffers) {
    FreeDummyBuffer(buffer, allocator);
  }
}

//===----------------------------------------------------------------------===//
// Host pointer tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, InsertWithHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  // Create a buffer with both device and host pointers.
  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Should be able to lookup by device pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer);

  // Should be able to lookup by host pointer.
  found = nullptr;
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, host_addr, &found));
  EXPECT_EQ(found, buffer);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, RemoveByHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 4096, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Remove using host pointer.
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, host_addr));

  // Should not be found by either pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));
  EXPECT_THAT(
      Status(iree_hal_streaming_buffer_table_lookup(table, host_addr, &found)),
      StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupRangeWithHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup range using host pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup_range(
      table, host_addr + 0x1000, 0x2000, &found));
  EXPECT_EQ(found, buffer);

  // Range extending beyond buffer should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup_range(
                  table, host_addr + 0x2000, 0x3000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, MixedHostPointers) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  // Insert buffers with and without host pointers.
  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 4096, allocator);
  void* host_ptr2 = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer2 = CreateDummyBuffer(0x110000000ULL, 4096, allocator, host_ptr2);
  auto* buffer3 = CreateDummyBuffer(0x120000000ULL, 4096, allocator);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer2));
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer3));

  // Verify all can be looked up by device pointer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL, &found));
  EXPECT_EQ(found, buffer1);
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x110000000ULL, &found));
  EXPECT_EQ(found, buffer2);
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, 0x120000000ULL, &found));
  EXPECT_EQ(found, buffer3);

  // Only buffer2 should be found by host pointer.
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr2;
  IREE_EXPECT_OK(
      iree_hal_streaming_buffer_table_lookup(table, host_addr, &found));
  EXPECT_EQ(found, buffer2);

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
  FreeDummyBuffer(buffer3, allocator);
}

//===----------------------------------------------------------------------===//
// Mid-buffer lookup tests
//===----------------------------------------------------------------------===//

TEST(BufferTableTest, LookupMidBufferDevice) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup at various offsets within the buffer.
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, 0x100000000ULL,
                                                        &found));  // Start
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, 0x100001000ULL,
                                                        &found));  // Middle
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, 0x100003FFFULL,
                                                        &found));  // Last byte
  EXPECT_EQ(found, buffer);

  // Just past the end should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100004000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, LookupMidBufferHost) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator, host_ptr);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Lookup at various offsets within the host buffer range.
  uint64_t host_addr = (uint64_t)(uintptr_t)host_ptr;
  iree_hal_streaming_buffer_t* found = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(table, host_addr,
                                                        &found));  // Start
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, host_addr + 0x1000, &found));  // Middle
  EXPECT_EQ(found, buffer);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_lookup(
      table, host_addr + 0x3FFF, &found));  // Last byte
  EXPECT_EQ(found, buffer);

  // Just past the end should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, host_addr + 0x4000, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, RemoveMidBuffer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  auto* buffer = CreateDummyBuffer(0x100000000ULL, 0x4000, allocator);
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer));

  // Remove using a pointer in the middle of the buffer.
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_remove(table, 0x100002000ULL));

  // Buffer should be gone.
  iree_hal_streaming_buffer_t* found = nullptr;
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_lookup(
                  table, 0x100000000ULL, &found)),
              StatusIs(StatusCode::kNotFound));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer, allocator);
}

TEST(BufferTableTest, DoubleInsertHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

  void* host_ptr = reinterpret_cast<void*>(0x200000000ULL);
  auto* buffer1 = CreateDummyBuffer(0x100000000ULL, 4096, allocator, host_ptr);
  // Different device pointer but same host pointer.
  auto* buffer2 = CreateDummyBuffer(0x110000000ULL, 4096, allocator, host_ptr);

  IREE_EXPECT_OK(iree_hal_streaming_buffer_table_insert(table, buffer1));

  // Second insert with same host_ptr should fail.
  EXPECT_THAT(Status(iree_hal_streaming_buffer_table_insert(table, buffer2)),
              StatusIs(StatusCode::kAlreadyExists));

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffer(buffer1, allocator);
  FreeDummyBuffer(buffer2, allocator);
}

}  // namespace
}  // namespace iree::hal::stream
