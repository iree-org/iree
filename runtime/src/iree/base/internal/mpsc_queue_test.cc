// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/mpsc_queue.h"

#include <atomic>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class MpscQueueTest : public ::testing::Test {
 protected:
  // Allocates aligned memory for a queue with the given capacity.
  void SetUp(uint32_t capacity) {
    iree_host_size_t size = iree_mpsc_queue_required_size(capacity);
    memory_.resize(size + 64);  // Extra for alignment.
    // Align to 64 bytes (cache line) for realistic behavior.
    uintptr_t base = (uintptr_t)memory_.data();
    uintptr_t aligned = (base + 63) & ~(uintptr_t)63;
    aligned_memory_ = (void*)aligned;
    memory_size_ = size;
  }

  void SetUp() override { SetUp(1024); }

  void* aligned_memory_ = nullptr;
  iree_host_size_t memory_size_ = 0;

 private:
  std::vector<uint8_t> memory_;
};

//===----------------------------------------------------------------------===//
// Required size computation
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, RequiredSizeComputation) {
  EXPECT_EQ(iree_mpsc_queue_required_size(64),
            IREE_MPSC_QUEUE_HEADER_SIZE + 64);
  EXPECT_EQ(iree_mpsc_queue_required_size(1024),
            IREE_MPSC_QUEUE_HEADER_SIZE + 1024);
  EXPECT_EQ(iree_mpsc_queue_required_size(65536),
            IREE_MPSC_QUEUE_HEADER_SIZE + 65536);
}

//===----------------------------------------------------------------------===//
// Initialization and validation
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, InitializeAndOpen) {
  iree_mpsc_queue_t producer;
  IREE_ASSERT_OK(iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024,
                                            &producer));

  // Verify header fields.
  EXPECT_EQ(producer.header->magic, IREE_MPSC_QUEUE_MAGIC);
  EXPECT_EQ(producer.header->version, IREE_MPSC_QUEUE_VERSION);
  EXPECT_EQ(producer.header->capacity, (uint32_t)1024);
  EXPECT_EQ(producer.header->entry_alignment,
            IREE_MPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT);
  EXPECT_EQ(producer.capacity, (uint32_t)1024);
  EXPECT_EQ(producer.mask, (uint32_t)1023);

  // Open the same memory as a consumer.
  iree_mpsc_queue_t consumer;
  IREE_ASSERT_OK(
      iree_mpsc_queue_open(aligned_memory_, memory_size_, &consumer));
  EXPECT_EQ(consumer.capacity, (uint32_t)1024);
  EXPECT_EQ(consumer.mask, (uint32_t)1023);
  EXPECT_EQ(consumer.entry_alignment, IREE_MPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT);
}

TEST_F(MpscQueueTest, InitializeNullMemory) {
  iree_mpsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_initialize(nullptr, memory_size_, 1024, &queue));
}

TEST_F(MpscQueueTest, InitializeCapacityTooSmall) {
  iree_mpsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 32, &queue));
}

TEST_F(MpscQueueTest, InitializeCapacityNotPowerOfTwo) {
  iree_mpsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 100, &queue));
}

TEST_F(MpscQueueTest, InitializeMemoryTooSmall) {
  iree_mpsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_initialize(aligned_memory_, 200, 1024, &queue));
}

TEST_F(MpscQueueTest, OpenBadMagic) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  queue.header->magic = 0xDEADBEEF;

  iree_mpsc_queue_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_open(aligned_memory_, memory_size_, &opener));
}

TEST_F(MpscQueueTest, OpenBadVersion) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  queue.header->version = 99;

  iree_mpsc_queue_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_open(aligned_memory_, memory_size_, &opener));
}

TEST_F(MpscQueueTest, OpenNonPowerOfTwoCapacity) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  queue.header->capacity = 100;

  iree_mpsc_queue_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_mpsc_queue_open(aligned_memory_, memory_size_, &opener));
}

//===----------------------------------------------------------------------===//
// Single-threaded producer/consumer
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, WriteReadSingle) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  const char* message = "hello";
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, message, strlen(message)));

  char buffer[64] = {0};
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, strlen(message));
  EXPECT_EQ(memcmp(buffer, message, length), 0);
}

TEST_F(MpscQueueTest, WriteReadMultiple) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_TRUE(iree_mpsc_queue_write(&queue, &i, sizeof(i))) << "write " << i;
  }

  for (uint32_t i = 0; i < 10; ++i) {
    uint32_t value = 0;
    iree_host_size_t length = 0;
    ASSERT_TRUE(iree_mpsc_queue_read(&queue, &value, sizeof(value), &length))
        << "read " << i;
    EXPECT_EQ(length, sizeof(uint32_t));
    EXPECT_EQ(value, i) << "at index " << i;
  }

  // Queue should now be empty.
  uint32_t dummy = 0;
  iree_host_size_t dummy_length = 0;
  EXPECT_FALSE(
      iree_mpsc_queue_read(&queue, &dummy, sizeof(dummy), &dummy_length));
}

TEST_F(MpscQueueTest, WriteReadVariableSizes) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  uint8_t small = 0xAB;
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, &small, 1));

  uint8_t medium[32];
  memset(medium, 0xCD, sizeof(medium));
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, medium, sizeof(medium)));

  uint8_t large[200];
  memset(large, 0xEF, sizeof(large));
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, large, sizeof(large)));

  uint8_t buffer[256];
  iree_host_size_t length = 0;

  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)1);
  EXPECT_EQ(buffer[0], 0xAB);

  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)32);
  EXPECT_EQ(memcmp(buffer, medium, 32), 0);

  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)200);
  EXPECT_EQ(memcmp(buffer, large, 200), 0);
}

TEST_F(MpscQueueTest, WriteFullReturnsFalse) {
  SetUp(64);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Each entry with 4 bytes of payload consumes 8 bytes (4 prefix + 4 payload,
  // already aligned to 8). So 64 / 8 = 8 entries should fill the ring.
  uint32_t value = 0;
  int written = 0;
  while (iree_mpsc_queue_write(&queue, &value, sizeof(value))) {
    ++written;
    ++value;
  }
  EXPECT_GT(written, 0);

  // Verify the ring is full.
  EXPECT_FALSE(iree_mpsc_queue_write(&queue, &value, sizeof(value)));
}

TEST_F(MpscQueueTest, ReadEmptyReturnsFalse) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  uint8_t buffer[64];
  iree_host_size_t length = 0;
  EXPECT_FALSE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)0);
}

TEST_F(MpscQueueTest, PeekAndConsume) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  uint32_t values[] = {100, 200, 300};
  for (auto v : values) {
    ASSERT_TRUE(iree_mpsc_queue_write(&queue, &v, sizeof(v)));
  }

  for (auto expected : values) {
    iree_host_size_t length = 0;
    const void* payload = iree_mpsc_queue_peek(&queue, &length);
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(length, sizeof(uint32_t));

    uint32_t value = 0;
    memcpy(&value, payload, sizeof(value));
    EXPECT_EQ(value, expected);

    iree_mpsc_queue_consume(&queue);
  }

  iree_host_size_t length = 0;
  EXPECT_EQ(iree_mpsc_queue_peek(&queue, &length), nullptr);
}

TEST_F(MpscQueueTest, PeekEmptyReturnsNull) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  iree_host_size_t length = 0;
  EXPECT_EQ(iree_mpsc_queue_peek(&queue, &length), nullptr);
  EXPECT_EQ(length, (iree_host_size_t)0);
}

//===----------------------------------------------------------------------===//
// Begin/commit/cancel write
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, BeginWriteCommit) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  iree_host_size_t payload_size = 16;
  iree_mpsc_queue_reservation_t reservation;
  void* payload =
      iree_mpsc_queue_begin_write(&queue, payload_size, &reservation);
  ASSERT_NE(payload, nullptr);

  memset(payload, 0x42, payload_size);
  iree_mpsc_queue_commit_write(&queue, reservation);

  uint8_t buffer[64];
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, payload_size);
  for (iree_host_size_t i = 0; i < length; ++i) {
    EXPECT_EQ(buffer[i], 0x42) << "at byte " << i;
  }
}

TEST_F(MpscQueueTest, CancelDoesNotDeliver) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  iree_host_size_t payload_size = 16;
  iree_mpsc_queue_reservation_t reservation;
  void* payload =
      iree_mpsc_queue_begin_write(&queue, payload_size, &reservation);
  ASSERT_NE(payload, nullptr);
  memset(payload, 0xAA, payload_size);
  iree_mpsc_queue_cancel_write(&queue, reservation);

  // The canceled entry occupies ring space but the consumer should skip it.
  // With MPSC, iree_mpsc_queue_can_read returns true (there IS reserved data),
  // but peek should skip the canceled entry and return NULL (nothing
  // committed).
  iree_host_size_t length = 0;
  EXPECT_EQ(iree_mpsc_queue_peek(&queue, &length), nullptr);
}

TEST_F(MpscQueueTest, CancelThenCommit) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Cancel a write.
  iree_mpsc_queue_reservation_t cancel_reservation;
  void* cancel_payload = iree_mpsc_queue_begin_write(&queue, sizeof(uint32_t),
                                                     &cancel_reservation);
  ASSERT_NE(cancel_payload, nullptr);
  uint32_t cancel_value = 0xDEAD;
  memcpy(cancel_payload, &cancel_value, sizeof(cancel_value));
  iree_mpsc_queue_cancel_write(&queue, cancel_reservation);

  // Commit a write after the cancel.
  iree_mpsc_queue_reservation_t commit_reservation;
  void* commit_payload = iree_mpsc_queue_begin_write(&queue, sizeof(uint32_t),
                                                     &commit_reservation);
  ASSERT_NE(commit_payload, nullptr);
  uint32_t commit_value = 42;
  memcpy(commit_payload, &commit_value, sizeof(commit_value));
  iree_mpsc_queue_commit_write(&queue, commit_reservation);

  // Should get only the committed value.
  uint32_t read_value = 0;
  iree_host_size_t length = 0;
  ASSERT_TRUE(
      iree_mpsc_queue_read(&queue, &read_value, sizeof(read_value), &length));
  EXPECT_EQ(length, sizeof(uint32_t));
  EXPECT_EQ(read_value, (uint32_t)42);

  // Queue should be empty now.
  EXPECT_FALSE(
      iree_mpsc_queue_read(&queue, &read_value, sizeof(read_value), &length));
}

TEST_F(MpscQueueTest, BeginWriteFullReturnsNull) {
  SetUp(64);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Fill the ring using begin/commit.
  int committed = 0;
  for (;;) {
    iree_mpsc_queue_reservation_t reservation;
    void* payload =
        iree_mpsc_queue_begin_write(&queue, sizeof(uint32_t), &reservation);
    if (!payload) break;
    uint32_t value = (uint32_t)committed;
    memcpy(payload, &value, sizeof(value));
    iree_mpsc_queue_commit_write(&queue, reservation);
    ++committed;
  }
  EXPECT_GT(committed, 0);

  // Should return NULL when full.
  iree_mpsc_queue_reservation_t reservation;
  EXPECT_EQ(iree_mpsc_queue_begin_write(&queue, sizeof(uint32_t), &reservation),
            nullptr);
}

//===----------------------------------------------------------------------===//
// Wrapping and skip markers
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, WrapAround) {
  SetUp(128);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 128, &queue));

  // Write and read enough entries to wrap around multiple times.
  const int total_entries = 100;
  for (int i = 0; i < total_entries; ++i) {
    uint32_t value = (uint32_t)i;
    while (!iree_mpsc_queue_write(&queue, &value, sizeof(value))) {
      uint32_t read_value = 0;
      iree_host_size_t length = 0;
      ASSERT_TRUE(iree_mpsc_queue_read(&queue, &read_value, sizeof(read_value),
                                       &length))
          << "failed to drain at entry " << i;
    }
  }

  // Drain remaining entries.
  uint32_t read_value = 0;
  iree_host_size_t length = 0;
  while (
      iree_mpsc_queue_read(&queue, &read_value, sizeof(read_value), &length)) {
  }
}

TEST_F(MpscQueueTest, SkipMarkerTransparency) {
  SetUp(64);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Write 7 entries of uint32_t to advance the write cursor to position 56.
  // Each entry = 8 bytes (4 prefix + 4 payload, aligned to 8).
  for (int i = 0; i < 7; ++i) {
    uint32_t value = (uint32_t)i;
    ASSERT_TRUE(iree_mpsc_queue_write(&queue, &value, sizeof(value)));
  }
  // Read them all to free space. reserve_pos = 56, read_pos = 56, free = 64.
  for (int i = 0; i < 7; ++i) {
    uint32_t value = 0;
    iree_host_size_t length = 0;
    ASSERT_TRUE(iree_mpsc_queue_read(&queue, &value, sizeof(value), &length));
    EXPECT_EQ(value, (uint32_t)i);
  }

  // Now write an 8-byte payload. Entry = align(4 + 8, 8) = 16 bytes.
  // Tail has only 8 bytes (64 - 56), so a skip marker is needed.
  uint8_t payload[8];
  memset(payload, 0xBB, sizeof(payload));
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, payload, sizeof(payload)));

  // Read it back — the skip marker should be transparent to the consumer.
  uint8_t buffer[64];
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)sizeof(payload));
  EXPECT_EQ(memcmp(buffer, payload, length), 0);
}

//===----------------------------------------------------------------------===//
// Query API
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, WriteAvailableAccuracy) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  EXPECT_EQ(iree_mpsc_queue_write_available(&queue), (iree_host_size_t)1024);
  EXPECT_EQ(iree_mpsc_queue_read_available(&queue), (iree_host_size_t)0);
  EXPECT_FALSE(iree_mpsc_queue_can_read(&queue));

  uint32_t value = 42;
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, &value, sizeof(value)));

  EXPECT_EQ(iree_mpsc_queue_write_available(&queue), (iree_host_size_t)1016);
  EXPECT_EQ(iree_mpsc_queue_read_available(&queue), (iree_host_size_t)8);
  EXPECT_TRUE(iree_mpsc_queue_can_read(&queue));

  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_mpsc_queue_read(&queue, &value, sizeof(value), &length));

  EXPECT_EQ(iree_mpsc_queue_write_available(&queue), (iree_host_size_t)1024);
  EXPECT_EQ(iree_mpsc_queue_read_available(&queue), (iree_host_size_t)0);
  EXPECT_FALSE(iree_mpsc_queue_can_read(&queue));
}

TEST_F(MpscQueueTest, EntryAlignmentPadding) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Write a 1-byte payload. Entry = 4 prefix + 1 payload = 5, padded to 8.
  uint8_t byte = 0xFF;
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, &byte, 1));

  // Write a 5-byte payload. Entry = 4 prefix + 5 payload = 9, padded to 16.
  uint8_t five_bytes[5] = {1, 2, 3, 4, 5};
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, five_bytes, sizeof(five_bytes)));

  // Total consumed: 8 + 16 = 24 bytes.
  EXPECT_EQ(iree_mpsc_queue_read_available(&queue), (iree_host_size_t)24);

  uint8_t buffer[16];
  iree_host_size_t length = 0;

  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)1);
  EXPECT_EQ(buffer[0], 0xFF);

  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)5);
  EXPECT_EQ(memcmp(buffer, five_bytes, 5), 0);
}

//===----------------------------------------------------------------------===//
// Backpressure: fill, drain, refill
//===----------------------------------------------------------------------===//

TEST_F(MpscQueueTest, BackpressureFull) {
  SetUp(256);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 256, &queue));

  // Fill the ring.
  int written = 0;
  uint32_t value = 0;
  while (iree_mpsc_queue_write(&queue, &value, sizeof(value))) {
    ++written;
    ++value;
  }
  ASSERT_GT(written, 0);

  // begin_write should also return NULL.
  iree_mpsc_queue_reservation_t reservation;
  EXPECT_EQ(iree_mpsc_queue_begin_write(&queue, sizeof(uint32_t), &reservation),
            nullptr);

  // Drain everything.
  int read_count = 0;
  uint32_t expected = 0;
  iree_host_size_t length = 0;
  while (iree_mpsc_queue_read(&queue, &value, sizeof(value), &length)) {
    EXPECT_EQ(value, expected) << "at read " << read_count;
    ++expected;
    ++read_count;
  }
  EXPECT_EQ(read_count, written);

  // Now writes should succeed again.
  value = 0xBEEF;
  ASSERT_TRUE(iree_mpsc_queue_write(&queue, &value, sizeof(value)));

  ASSERT_TRUE(iree_mpsc_queue_read(&queue, &value, sizeof(value), &length));
  EXPECT_EQ(value, (uint32_t)0xBEEF);
}

//===----------------------------------------------------------------------===//
// Multi-threaded stress tests
//===----------------------------------------------------------------------===//

// Single producer, single consumer — validates MPSC works as a strict superset
// of SPSC.
TEST_F(MpscQueueTest, SingleProducerConsumerStress) {
  SetUp(4096);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 4096, &queue));

  const int total_messages = 100000;
  std::atomic<int> consumer_count{0};

  std::thread producer([&]() {
    for (int i = 0; i < total_messages; ++i) {
      uint32_t value = (uint32_t)i;
      while (!iree_mpsc_queue_write(&queue, &value, sizeof(value))) {
        std::this_thread::yield();
      }
    }
  });

  std::thread consumer([&]() {
    uint32_t expected = 0;
    while (expected < (uint32_t)total_messages) {
      uint32_t value = 0;
      iree_host_size_t length = 0;
      if (iree_mpsc_queue_read(&queue, &value, sizeof(value), &length)) {
        ASSERT_EQ(length, sizeof(uint32_t))
            << "wrong length at message " << expected;
        ASSERT_EQ(value, expected) << "wrong value at message " << expected;
        ++expected;
        consumer_count.store((int)expected, std::memory_order_relaxed);
      }
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(consumer_count.load(), total_messages);
}

// Multiple producers, single consumer. Each producer writes its own sequence
// of tagged values. The consumer verifies:
//   - All values from all producers are received (no loss)
//   - Within each producer, values arrive in order (per-producer FIFO)
TEST_F(MpscQueueTest, ConcurrentProducers) {
  SetUp(8192);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 8192, &queue));

  const int producer_count = 4;
  const int messages_per_producer = 10000;
  const int total_messages = producer_count * messages_per_producer;

  // Each message is a (producer_id, sequence_number) pair.
  struct TaggedMessage {
    uint32_t producer_id;
    uint32_t sequence;
  };

  std::vector<std::thread> producers;
  for (int p = 0; p < producer_count; ++p) {
    producers.emplace_back([&queue, p, messages_per_producer]() {
      for (int i = 0; i < messages_per_producer; ++i) {
        TaggedMessage msg = {(uint32_t)p, (uint32_t)i};
        while (!iree_mpsc_queue_write(&queue, &msg, sizeof(msg))) {
          std::this_thread::yield();
        }
      }
    });
  }

  // Consumer: track per-producer sequence numbers to verify ordering.
  std::vector<uint32_t> next_expected(producer_count, 0);
  int received = 0;

  std::thread consumer([&]() {
    while (received < total_messages) {
      TaggedMessage msg = {};
      iree_host_size_t length = 0;
      if (iree_mpsc_queue_read(&queue, &msg, sizeof(msg), &length)) {
        ASSERT_EQ(length, sizeof(TaggedMessage));
        ASSERT_LT(msg.producer_id, (uint32_t)producer_count)
            << "invalid producer_id";
        ASSERT_EQ(msg.sequence, next_expected[msg.producer_id])
            << "out-of-order message from producer " << msg.producer_id;
        ++next_expected[msg.producer_id];
        ++received;
      }
    }
  });

  for (auto& t : producers) t.join();
  consumer.join();

  EXPECT_EQ(received, total_messages);
  for (int p = 0; p < producer_count; ++p) {
    EXPECT_EQ(next_expected[p], (uint32_t)messages_per_producer)
        << "producer " << p << " missing messages";
  }
}

// Concurrent producers using begin_write/commit_write (the zero-copy path).
TEST_F(MpscQueueTest, ConcurrentBeginWriteCommit) {
  SetUp(8192);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 8192, &queue));

  const int producer_count = 4;
  const int messages_per_producer = 10000;
  const int total_messages = producer_count * messages_per_producer;

  struct TaggedMessage {
    uint32_t producer_id;
    uint32_t sequence;
  };

  std::vector<std::thread> producers;
  for (int p = 0; p < producer_count; ++p) {
    producers.emplace_back([&queue, p, messages_per_producer]() {
      for (int i = 0; i < messages_per_producer; ++i) {
        iree_mpsc_queue_reservation_t reservation;
        void* payload;
        while (!(payload = iree_mpsc_queue_begin_write(
                     &queue, sizeof(TaggedMessage), &reservation))) {
          std::this_thread::yield();
        }
        TaggedMessage msg = {(uint32_t)p, (uint32_t)i};
        memcpy(payload, &msg, sizeof(msg));
        iree_mpsc_queue_commit_write(&queue, reservation);
      }
    });
  }

  std::vector<uint32_t> next_expected(producer_count, 0);
  int received = 0;

  std::thread consumer([&]() {
    while (received < total_messages) {
      TaggedMessage msg = {};
      iree_host_size_t length = 0;
      if (iree_mpsc_queue_read(&queue, &msg, sizeof(msg), &length)) {
        ASSERT_EQ(length, sizeof(TaggedMessage));
        ASSERT_LT(msg.producer_id, (uint32_t)producer_count);
        ASSERT_EQ(msg.sequence, next_expected[msg.producer_id])
            << "out-of-order from producer " << msg.producer_id;
        ++next_expected[msg.producer_id];
        ++received;
      }
    }
  });

  for (auto& t : producers) t.join();
  consumer.join();

  EXPECT_EQ(received, total_messages);
}

// Variable-length messages from multiple concurrent producers.
TEST_F(MpscQueueTest, ConcurrentVariableSizeStress) {
  SetUp(16384);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 16384, &queue));

  const int producer_count = 4;
  const int messages_per_producer = 5000;
  const int total_messages = producer_count * messages_per_producer;

  std::vector<std::thread> producers;
  for (int p = 0; p < producer_count; ++p) {
    producers.emplace_back([&queue, p, messages_per_producer]() {
      for (int i = 0; i < messages_per_producer; ++i) {
        // Variable payload: 8 bytes header + 0-120 bytes of fill.
        // Header is (producer_id, sequence).
        iree_host_size_t fill_size = (iree_host_size_t)(i % 121);
        iree_host_size_t payload_size = sizeof(uint32_t) * 2 + fill_size;

        uint8_t payload[256];
        uint32_t producer_id = (uint32_t)p;
        uint32_t sequence = (uint32_t)i;
        memcpy(payload, &producer_id, sizeof(producer_id));
        memcpy(payload + sizeof(producer_id), &sequence, sizeof(sequence));
        memset(payload + sizeof(uint32_t) * 2, (uint8_t)(i & 0xFF), fill_size);

        while (!iree_mpsc_queue_write(&queue, payload, payload_size)) {
          std::this_thread::yield();
        }
      }
    });
  }

  std::vector<uint32_t> next_expected(producer_count, 0);
  int received = 0;

  std::thread consumer([&]() {
    uint8_t buffer[512];
    while (received < total_messages) {
      iree_host_size_t length = 0;
      if (iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length)) {
        ASSERT_GE(length, sizeof(uint32_t) * 2);
        uint32_t producer_id = 0;
        uint32_t sequence = 0;
        memcpy(&producer_id, buffer, sizeof(producer_id));
        memcpy(&sequence, buffer + sizeof(producer_id), sizeof(sequence));
        ASSERT_LT(producer_id, (uint32_t)producer_count);
        ASSERT_EQ(sequence, next_expected[producer_id])
            << "out-of-order from producer " << producer_id;
        ++next_expected[producer_id];
        ++received;
      }
    }
  });

  for (auto& t : producers) t.join();
  consumer.join();

  EXPECT_EQ(received, total_messages);
}

// Test that canceled entries from concurrent producers are correctly skipped.
TEST_F(MpscQueueTest, ConcurrentCancelStress) {
  SetUp(8192);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 8192, &queue));

  const int producer_count = 4;
  const int messages_per_producer = 5000;
  const int total_committed = producer_count * messages_per_producer;

  struct TaggedMessage {
    uint32_t producer_id;
    uint32_t sequence;
  };

  std::vector<std::thread> producers;
  for (int p = 0; p < producer_count; ++p) {
    producers.emplace_back([&queue, p, messages_per_producer]() {
      uint32_t committed = 0;
      for (int i = 0; committed < (uint32_t)messages_per_producer; ++i) {
        iree_mpsc_queue_reservation_t reservation;
        void* payload;
        while (!(payload = iree_mpsc_queue_begin_write(
                     &queue, sizeof(TaggedMessage), &reservation))) {
          std::this_thread::yield();
        }

        // Cancel every 7th reservation.
        if (i % 7 == 3) {
          iree_mpsc_queue_cancel_write(&queue, reservation);
          continue;
        }

        TaggedMessage msg = {(uint32_t)p, committed};
        memcpy(payload, &msg, sizeof(msg));
        iree_mpsc_queue_commit_write(&queue, reservation);
        ++committed;
      }
    });
  }

  std::vector<uint32_t> next_expected(producer_count, 0);
  int received = 0;

  std::thread consumer([&]() {
    while (received < total_committed) {
      TaggedMessage msg = {};
      iree_host_size_t length = 0;
      if (iree_mpsc_queue_read(&queue, &msg, sizeof(msg), &length)) {
        ASSERT_EQ(length, sizeof(TaggedMessage));
        ASSERT_LT(msg.producer_id, (uint32_t)producer_count);
        ASSERT_EQ(msg.sequence, next_expected[msg.producer_id])
            << "out-of-order from producer " << msg.producer_id;
        ++next_expected[msg.producer_id];
        ++received;
      }
    }
  });

  for (auto& t : producers) t.join();
  consumer.join();

  EXPECT_EQ(received, total_committed);
}

// Producer faster than consumer with backpressure.
TEST_F(MpscQueueTest, ProducerFasterThanConsumer) {
  SetUp(1024);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  const int total_messages = 10000;
  std::atomic<int> consumer_count{0};

  std::thread producer([&]() {
    for (int i = 0; i < total_messages; ++i) {
      uint32_t value = (uint32_t)i;
      while (!iree_mpsc_queue_write(&queue, &value, sizeof(value))) {
        std::this_thread::yield();
      }
    }
  });

  std::thread consumer([&]() {
    uint32_t expected = 0;
    while (expected < (uint32_t)total_messages) {
      uint32_t value = 0;
      iree_host_size_t length = 0;
      if (iree_mpsc_queue_read(&queue, &value, sizeof(value), &length)) {
        ASSERT_EQ(value, expected);
        ++expected;
        consumer_count.store((int)expected, std::memory_order_relaxed);
        if (expected % 100 == 0) {
          std::this_thread::yield();
        }
      }
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(consumer_count.load(), total_messages);
}

// Large payloads near ring capacity.
TEST_F(MpscQueueTest, LargePayload) {
  SetUp(65536);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 65536, &queue));

  // Write a payload that's close to half the ring capacity.
  // align(4 + 32000, 8) = 32008 bytes per entry. Two of these fit in 65536.
  const iree_host_size_t payload_size = 32000;
  std::vector<uint8_t> expected(payload_size);
  for (iree_host_size_t i = 0; i < payload_size; ++i) {
    expected[i] = (uint8_t)(i & 0xFF);
  }

  ASSERT_TRUE(iree_mpsc_queue_write(&queue, expected.data(), payload_size));

  std::vector<uint8_t> buffer(payload_size);
  iree_host_size_t length = 0;
  ASSERT_TRUE(
      iree_mpsc_queue_read(&queue, buffer.data(), buffer.size(), &length));
  EXPECT_EQ(length, payload_size);
  EXPECT_EQ(memcmp(buffer.data(), expected.data(), payload_size), 0);
}

// Variable-sized entries across ring iterations. Exercises the scenario where
// a new entry's prefix falls at a physical offset that was previously inside
// a larger entry's payload. The consumer's full-entry memset on consume is
// what prevents stale payload data from being misinterpreted as entry states.
TEST_F(MpscQueueTest, VariableSizeReuseCorrectness) {
  SetUp(64);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Pass 1: write one large entry (24 bytes payload).
  // entry_size = align(4 + 24, 8) = 32. Occupies bytes 0-31.
  // Fill payload with 0xAA so any stale prefix byte is obviously non-zero.
  uint8_t large_payload[24];
  memset(large_payload, 0xAA, sizeof(large_payload));
  ASSERT_TRUE(
      iree_mpsc_queue_write(&queue, large_payload, sizeof(large_payload)));

  // Write another entry to fill the ring further.
  // entry_size = align(4 + 24, 8) = 32. Occupies bytes 32-63.
  ASSERT_TRUE(
      iree_mpsc_queue_write(&queue, large_payload, sizeof(large_payload)));

  // Consume both entries.
  uint8_t buffer[64];
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, sizeof(large_payload));
  ASSERT_TRUE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, sizeof(large_payload));

  // Pass 2: write small entries (4 bytes payload each).
  // entry_size = align(4 + 4, 8) = 8. These will tile the ring in 8-byte
  // chunks, and several entries' prefixes will land at offsets that were
  // inside the pass-1 large entries' payload regions (e.g., offset 8, 16, 24).
  for (int i = 0; i < 8; ++i) {
    uint32_t value = (uint32_t)(i + 100);
    ASSERT_TRUE(iree_mpsc_queue_write(&queue, &value, sizeof(value)))
        << "pass 2 write " << i;
  }

  // Read them all back — this is where the bug would manifest. Without the
  // consumer-side full-entry memset on consume, stale 0xAA bytes at prefix
  // positions would be misinterpreted as entry lengths, corrupting the queue.
  for (int i = 0; i < 8; ++i) {
    uint32_t value = 0;
    ASSERT_TRUE(iree_mpsc_queue_read(&queue, &value, sizeof(value), &length))
        << "pass 2 read " << i;
    EXPECT_EQ(length, sizeof(uint32_t));
    EXPECT_EQ(value, (uint32_t)(i + 100)) << "at index " << i;
  }

  // Queue should be empty.
  EXPECT_FALSE(iree_mpsc_queue_read(&queue, buffer, sizeof(buffer), &length));
}

// Validates that zero-length writes are rejected.
TEST_F(MpscQueueTest, ZeroLengthWriteFails) {
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  EXPECT_FALSE(iree_mpsc_queue_write(&queue, "x", 0));
}

// Validates that oversized writes are rejected immediately.
TEST_F(MpscQueueTest, OversizedWriteFails) {
  SetUp(64);
  iree_mpsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_mpsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // An entry with 64 bytes of payload needs align(4+64, 8) = 72 bytes, which
  // exceeds the 64-byte ring capacity.
  uint8_t payload[64];
  memset(payload, 0, sizeof(payload));
  EXPECT_FALSE(iree_mpsc_queue_write(&queue, payload, sizeof(payload)));
  EXPECT_EQ(iree_mpsc_queue_write_available(&queue), (iree_host_size_t)64);
}

}  // namespace
