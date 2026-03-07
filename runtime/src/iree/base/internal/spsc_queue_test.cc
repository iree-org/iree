// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/spsc_queue.h"

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class SpscQueueTest : public ::testing::Test {
 protected:
  // Allocates aligned memory for a queue with the given capacity.
  void SetUp(uint32_t capacity) {
    iree_host_size_t size = iree_spsc_queue_required_size(capacity);
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

TEST_F(SpscQueueTest, RequiredSizeComputation) {
  EXPECT_EQ(iree_spsc_queue_required_size(64),
            IREE_SPSC_QUEUE_HEADER_SIZE + 64);
  EXPECT_EQ(iree_spsc_queue_required_size(1024),
            IREE_SPSC_QUEUE_HEADER_SIZE + 1024);
  EXPECT_EQ(iree_spsc_queue_required_size(65536),
            IREE_SPSC_QUEUE_HEADER_SIZE + 65536);
}

//===----------------------------------------------------------------------===//
// Initialization and validation
//===----------------------------------------------------------------------===//

TEST_F(SpscQueueTest, InitializeAndOpen) {
  iree_spsc_queue_t producer;
  IREE_ASSERT_OK(iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024,
                                            &producer));

  // Verify header fields.
  EXPECT_EQ(producer.header->magic, IREE_SPSC_QUEUE_MAGIC);
  EXPECT_EQ(producer.header->version, IREE_SPSC_QUEUE_VERSION);
  EXPECT_EQ(producer.header->capacity, (uint32_t)1024);
  EXPECT_EQ(producer.header->entry_alignment,
            IREE_SPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT);
  EXPECT_EQ(producer.capacity, (uint32_t)1024);
  EXPECT_EQ(producer.mask, (uint32_t)1023);

  // Open the same memory as a consumer.
  iree_spsc_queue_t consumer;
  IREE_ASSERT_OK(
      iree_spsc_queue_open(aligned_memory_, memory_size_, &consumer));
  EXPECT_EQ(consumer.capacity, (uint32_t)1024);
  EXPECT_EQ(consumer.mask, (uint32_t)1023);
  EXPECT_EQ(consumer.entry_alignment, IREE_SPSC_QUEUE_DEFAULT_ENTRY_ALIGNMENT);
}

TEST_F(SpscQueueTest, InitializeNullMemory) {
  iree_spsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_initialize(nullptr, memory_size_, 1024, &queue));
}

TEST_F(SpscQueueTest, InitializeCapacityTooSmall) {
  iree_spsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 32, &queue));
}

TEST_F(SpscQueueTest, InitializeCapacityNotPowerOfTwo) {
  iree_spsc_queue_t queue;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 100, &queue));
}

TEST_F(SpscQueueTest, InitializeMemoryTooSmall) {
  iree_spsc_queue_t queue;
  // Provide less memory than required for capacity 1024.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_initialize(aligned_memory_, 200, 1024, &queue));
}

TEST_F(SpscQueueTest, OpenBadMagic) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  // Corrupt the magic.
  queue.header->magic = 0xDEADBEEF;

  iree_spsc_queue_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_open(aligned_memory_, memory_size_, &opener));
}

TEST_F(SpscQueueTest, OpenBadVersion) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  // Corrupt the version.
  queue.header->version = 99;

  iree_spsc_queue_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_open(aligned_memory_, memory_size_, &opener));
}

TEST_F(SpscQueueTest, OpenNonPowerOfTwoCapacity) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  // Corrupt capacity to a non-power-of-two value.
  queue.header->capacity = 100;

  iree_spsc_queue_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_spsc_queue_open(aligned_memory_, memory_size_, &opener));
}

//===----------------------------------------------------------------------===//
// Single-threaded producer/consumer
//===----------------------------------------------------------------------===//

TEST_F(SpscQueueTest, WriteReadSingle) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  const char* message = "hello";
  ASSERT_TRUE(iree_spsc_queue_write(&queue, message, strlen(message)));

  char buffer[64] = {0};
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, strlen(message));
  EXPECT_EQ(memcmp(buffer, message, length), 0);
}

TEST_F(SpscQueueTest, WriteReadMultiple) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Write multiple entries.
  for (uint32_t i = 0; i < 10; ++i) {
    ASSERT_TRUE(iree_spsc_queue_write(&queue, &i, sizeof(i))) << "write " << i;
  }

  // Read them all back and verify order.
  for (uint32_t i = 0; i < 10; ++i) {
    uint32_t value = 0;
    iree_host_size_t length = 0;
    ASSERT_TRUE(iree_spsc_queue_read(&queue, &value, sizeof(value), &length))
        << "read " << i;
    EXPECT_EQ(length, sizeof(uint32_t));
    EXPECT_EQ(value, i) << "at index " << i;
  }

  // Queue should now be empty.
  uint32_t dummy = 0;
  iree_host_size_t dummy_length = 0;
  EXPECT_FALSE(
      iree_spsc_queue_read(&queue, &dummy, sizeof(dummy), &dummy_length));
}

TEST_F(SpscQueueTest, WriteReadVariableSizes) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Write entries of varying sizes.
  uint8_t small = 0xAB;
  ASSERT_TRUE(iree_spsc_queue_write(&queue, &small, 1));

  uint8_t medium[32];
  memset(medium, 0xCD, sizeof(medium));
  ASSERT_TRUE(iree_spsc_queue_write(&queue, medium, sizeof(medium)));

  uint8_t large[200];
  memset(large, 0xEF, sizeof(large));
  ASSERT_TRUE(iree_spsc_queue_write(&queue, large, sizeof(large)));

  // Read back and verify.
  uint8_t buffer[256];
  iree_host_size_t length = 0;

  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)1);
  EXPECT_EQ(buffer[0], 0xAB);

  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)32);
  EXPECT_EQ(memcmp(buffer, medium, 32), 0);

  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)200);
  EXPECT_EQ(memcmp(buffer, large, 200), 0);
}

TEST_F(SpscQueueTest, WriteFullReturnsFalse) {
  // Use a small capacity to fill quickly.
  SetUp(64);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Each entry with 4 bytes of payload consumes 8 bytes (4 prefix + 4 payload,
  // already aligned to 8). So 64 / 8 = 8 entries should fill the ring.
  uint32_t value = 0;
  int written = 0;
  while (iree_spsc_queue_write(&queue, &value, sizeof(value))) {
    ++written;
    ++value;
  }
  EXPECT_GT(written, 0);

  // Verify the ring is full.
  EXPECT_FALSE(iree_spsc_queue_write(&queue, &value, sizeof(value)));
}

TEST_F(SpscQueueTest, ReadEmptyReturnsFalse) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  uint8_t buffer[64];
  iree_host_size_t length = 0;
  EXPECT_FALSE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)0);
}

TEST_F(SpscQueueTest, PeekAndConsume) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  uint32_t values[] = {100, 200, 300};
  for (auto v : values) {
    ASSERT_TRUE(iree_spsc_queue_write(&queue, &v, sizeof(v)));
  }

  for (auto expected : values) {
    iree_host_size_t length = 0;
    const void* payload = iree_spsc_queue_peek(&queue, &length);
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(length, sizeof(uint32_t));

    uint32_t value = 0;
    memcpy(&value, payload, sizeof(value));
    EXPECT_EQ(value, expected);

    iree_spsc_queue_consume(&queue);
  }

  // Queue should now be empty.
  iree_host_size_t length = 0;
  EXPECT_EQ(iree_spsc_queue_peek(&queue, &length), nullptr);
}

TEST_F(SpscQueueTest, PeekEmptyReturnsNull) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));
  iree_host_size_t length = 0;
  EXPECT_EQ(iree_spsc_queue_peek(&queue, &length), nullptr);
  EXPECT_EQ(length, (iree_host_size_t)0);
}

TEST_F(SpscQueueTest, WrapAround) {
  // Use a small ring to force wrapping.
  SetUp(128);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 128, &queue));

  // Write and read enough entries to wrap around multiple times.
  // With 8-byte entries (4 prefix + 4 payload), 128 bytes = 16 entries per lap.
  const int total_entries = 100;
  for (int i = 0; i < total_entries; ++i) {
    uint32_t value = (uint32_t)i;
    // Drain if full.
    while (!iree_spsc_queue_write(&queue, &value, sizeof(value))) {
      uint32_t read_value = 0;
      iree_host_size_t length = 0;
      ASSERT_TRUE(iree_spsc_queue_read(&queue, &read_value, sizeof(read_value),
                                       &length))
          << "failed to drain at entry " << i;
    }
  }

  // Drain remaining entries.
  uint32_t read_value = 0;
  iree_host_size_t length = 0;
  while (
      iree_spsc_queue_read(&queue, &read_value, sizeof(read_value), &length)) {
    // Just drain.
  }
}

TEST_F(SpscQueueTest, SkipMarkerTransparency) {
  // Use a small ring to force skip markers.
  SetUp(64);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Write 7 entries of uint32_t to advance the write cursor to position 56.
  // Each entry = 8 bytes (4 prefix + 4 payload, aligned to 8).
  for (int i = 0; i < 7; ++i) {
    uint32_t value = (uint32_t)i;
    ASSERT_TRUE(iree_spsc_queue_write(&queue, &value, sizeof(value)));
  }
  // Read them all to free space. write_pos = 56, read_pos = 56, free = 64.
  for (int i = 0; i < 7; ++i) {
    uint32_t value = 0;
    iree_host_size_t length = 0;
    ASSERT_TRUE(iree_spsc_queue_read(&queue, &value, sizeof(value), &length));
    EXPECT_EQ(value, (uint32_t)i);
  }

  // Now write an 8-byte payload. Entry = align(4 + 8, 8) = 16 bytes.
  // Tail has only 8 bytes (64 - 56), so a skip marker is needed.
  // After skip: write_pos = 64, free = 64 - (64 - 56) = 56 >= 16. Fits.
  uint8_t payload[8];
  memset(payload, 0xBB, sizeof(payload));
  ASSERT_TRUE(iree_spsc_queue_write(&queue, payload, sizeof(payload)));

  // Read it back — the skip marker should be transparent to the consumer.
  uint8_t buffer[64];
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)sizeof(payload));
  EXPECT_EQ(memcmp(buffer, payload, length), 0);
}

TEST_F(SpscQueueTest, BeginWriteCommit) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Use the two-phase API for zero-copy writes.
  iree_host_size_t payload_size = 16;
  void* payload = iree_spsc_queue_begin_write(&queue, payload_size);
  ASSERT_NE(payload, nullptr);

  // Write a pattern directly into the ring.
  memset(payload, 0x42, payload_size);
  iree_spsc_queue_commit_write(&queue, payload_size);

  // Read it back.
  uint8_t buffer[64];
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, payload_size);
  for (iree_host_size_t i = 0; i < length; ++i) {
    EXPECT_EQ(buffer[i], 0x42) << "at byte " << i;
  }
}

TEST_F(SpscQueueTest, BeginWriteFullReturnsNull) {
  SetUp(64);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 64, &queue));

  // Fill the ring.
  while (iree_spsc_queue_begin_write(&queue, sizeof(uint32_t)) != nullptr) {
    iree_spsc_queue_commit_write(&queue, sizeof(uint32_t));
  }

  // Should return NULL when full.
  EXPECT_EQ(iree_spsc_queue_begin_write(&queue, sizeof(uint32_t)), nullptr);
}

TEST_F(SpscQueueTest, WriteAvailableAccuracy) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Initially all space is available.
  EXPECT_EQ(iree_spsc_queue_write_available(&queue), (iree_host_size_t)1024);
  EXPECT_EQ(iree_spsc_queue_read_available(&queue), (iree_host_size_t)0);
  EXPECT_FALSE(iree_spsc_queue_can_read(&queue));

  // Write one entry (4 prefix + 4 payload = 8 bytes with default alignment).
  uint32_t value = 42;
  ASSERT_TRUE(iree_spsc_queue_write(&queue, &value, sizeof(value)));

  EXPECT_EQ(iree_spsc_queue_write_available(&queue), (iree_host_size_t)1016);
  EXPECT_EQ(iree_spsc_queue_read_available(&queue), (iree_host_size_t)8);
  EXPECT_TRUE(iree_spsc_queue_can_read(&queue));

  // Read it back.
  iree_host_size_t length = 0;
  ASSERT_TRUE(iree_spsc_queue_read(&queue, &value, sizeof(value), &length));

  EXPECT_EQ(iree_spsc_queue_write_available(&queue), (iree_host_size_t)1024);
  EXPECT_EQ(iree_spsc_queue_read_available(&queue), (iree_host_size_t)0);
  EXPECT_FALSE(iree_spsc_queue_can_read(&queue));
}

TEST_F(SpscQueueTest, EntryAlignmentPadding) {
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  // Write a 1-byte payload. Entry = 4 prefix + 1 payload = 5, padded to 8.
  uint8_t byte = 0xFF;
  ASSERT_TRUE(iree_spsc_queue_write(&queue, &byte, 1));

  // Write a 5-byte payload. Entry = 4 prefix + 5 payload = 9, padded to 16.
  uint8_t five_bytes[5] = {1, 2, 3, 4, 5};
  ASSERT_TRUE(iree_spsc_queue_write(&queue, five_bytes, sizeof(five_bytes)));

  // Total consumed: 8 + 16 = 24 bytes.
  EXPECT_EQ(iree_spsc_queue_read_available(&queue), (iree_host_size_t)24);

  // Read both back.
  uint8_t buffer[16];
  iree_host_size_t length = 0;

  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)1);
  EXPECT_EQ(buffer[0], 0xFF);

  ASSERT_TRUE(iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length));
  EXPECT_EQ(length, (iree_host_size_t)5);
  EXPECT_EQ(memcmp(buffer, five_bytes, 5), 0);
}

//===----------------------------------------------------------------------===//
// Multi-threaded stress tests
//===----------------------------------------------------------------------===//

TEST_F(SpscQueueTest, ProducerConsumerStress) {
  // Use a moderate ring to exercise wrapping under concurrent access.
  SetUp(4096);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 4096, &queue));

  const int total_messages = 100000;
  std::atomic<bool> producer_done{false};
  std::atomic<int> consumer_count{0};

  // Producer: writes sequential uint32_t values.
  std::thread producer([&]() {
    for (int i = 0; i < total_messages; ++i) {
      uint32_t value = (uint32_t)i;
      while (!iree_spsc_queue_write(&queue, &value, sizeof(value))) {
        std::this_thread::yield();
      }
    }
    producer_done.store(true, std::memory_order_release);
  });

  // Consumer: reads and verifies sequential values.
  std::thread consumer([&]() {
    uint32_t expected = 0;
    while (expected < (uint32_t)total_messages) {
      uint32_t value = 0;
      iree_host_size_t length = 0;
      if (iree_spsc_queue_read(&queue, &value, sizeof(value), &length)) {
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

TEST_F(SpscQueueTest, ProducerFasterThanConsumer) {
  SetUp(1024);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  const int total_messages = 10000;
  std::atomic<int> consumer_count{0};

  // Producer: writes as fast as possible, backs off when full.
  std::thread producer([&]() {
    for (int i = 0; i < total_messages; ++i) {
      uint32_t value = (uint32_t)i;
      while (!iree_spsc_queue_write(&queue, &value, sizeof(value))) {
        std::this_thread::yield();
      }
    }
  });

  // Consumer: reads with artificial delay.
  std::thread consumer([&]() {
    uint32_t expected = 0;
    while (expected < (uint32_t)total_messages) {
      uint32_t value = 0;
      iree_host_size_t length = 0;
      if (iree_spsc_queue_read(&queue, &value, sizeof(value), &length)) {
        ASSERT_EQ(value, expected);
        ++expected;
        consumer_count.store((int)expected, std::memory_order_relaxed);
        // Artificial delay every 100 messages.
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

TEST_F(SpscQueueTest, ConsumerFasterThanProducer) {
  SetUp(1024);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 1024, &queue));

  const int total_messages = 10000;
  std::atomic<bool> producer_done{false};
  std::atomic<int> consumer_count{0};

  // Producer: writes with artificial delay.
  std::thread producer([&]() {
    for (int i = 0; i < total_messages; ++i) {
      uint32_t value = (uint32_t)i;
      while (!iree_spsc_queue_write(&queue, &value, sizeof(value))) {
        std::this_thread::yield();
      }
      // Artificial delay every 100 messages.
      if (i % 100 == 0) {
        std::this_thread::yield();
      }
    }
    producer_done.store(true, std::memory_order_release);
  });

  // Consumer: reads as fast as possible.
  std::thread consumer([&]() {
    uint32_t expected = 0;
    while (expected < (uint32_t)total_messages) {
      uint32_t value = 0;
      iree_host_size_t length = 0;
      if (iree_spsc_queue_read(&queue, &value, sizeof(value), &length)) {
        ASSERT_EQ(value, expected);
        ++expected;
        consumer_count.store((int)expected, std::memory_order_relaxed);
      }
    }
  });

  producer.join();
  consumer.join();

  EXPECT_EQ(consumer_count.load(), total_messages);
}

TEST_F(SpscQueueTest, VariableSizeStress) {
  // Stress test with variable-length messages to exercise skip markers
  // and alignment padding under concurrent access.
  SetUp(4096);
  iree_spsc_queue_t queue;
  IREE_ASSERT_OK(
      iree_spsc_queue_initialize(aligned_memory_, memory_size_, 4096, &queue));

  const int total_messages = 50000;

  // Producer: writes messages with varying sizes.
  std::thread producer([&]() {
    for (int i = 0; i < total_messages; ++i) {
      // Size varies from 1 to 128 bytes. The first 4 bytes are the sequence
      // number for verification.
      iree_host_size_t payload_size = (iree_host_size_t)(1 + (i % 128));
      if (payload_size < sizeof(uint32_t)) payload_size = sizeof(uint32_t);

      uint8_t payload[128 + sizeof(uint32_t)];
      uint32_t sequence = (uint32_t)i;
      memcpy(payload, &sequence, sizeof(sequence));
      memset(payload + sizeof(sequence), (uint8_t)(i & 0xFF),
             payload_size - sizeof(sequence));

      while (!iree_spsc_queue_write(&queue, payload, payload_size)) {
        std::this_thread::yield();
      }
    }
  });

  // Consumer: reads and verifies sequence numbers.
  std::thread consumer([&]() {
    uint8_t buffer[256];
    uint32_t expected = 0;
    while (expected < (uint32_t)total_messages) {
      iree_host_size_t length = 0;
      if (iree_spsc_queue_read(&queue, buffer, sizeof(buffer), &length)) {
        ASSERT_GE(length, sizeof(uint32_t)) << "at message " << expected;
        uint32_t sequence = 0;
        memcpy(&sequence, buffer, sizeof(sequence));
        ASSERT_EQ(sequence, expected)
            << "sequence mismatch at message " << expected;
        ++expected;
      }
    }
  });

  producer.join();
  consumer.join();
}

}  // namespace
