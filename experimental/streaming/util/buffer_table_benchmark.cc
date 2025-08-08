// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include "experimental/streaming/util/buffer_table.h"
#include "iree/base/api.h"
#include "iree/testing/benchmark.h"

// Dummy buffer structure for benchmarking.
// We only need the device_ptr field for the buffer table.
struct iree_hal_streaming_buffer_t {
  iree_hal_streaming_deviceptr_t device_ptr;
  void* host_ptr;
  size_t size;
  // Padding to make it more realistic.
  uint64_t padding[5];
};

//===----------------------------------------------------------------------===//
// Helper functions for generating virtual addresses
//===----------------------------------------------------------------------===//

// Generates ascending addresses starting from a base.
static std::vector<iree_hal_streaming_deviceptr_t> GenerateAscendingAddresses(
    size_t count, iree_hal_streaming_deviceptr_t base = 0x100000000ULL,
    size_t stride = 0x10000) {
  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    addresses.push_back(base + i * stride);
  }
  return addresses;
}

// Generates descending addresses.
static std::vector<iree_hal_streaming_deviceptr_t> GenerateDescendingAddresses(
    size_t count, iree_hal_streaming_deviceptr_t base = 0x700000000000ULL,
    size_t stride = 0x10000) {
  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    addresses.push_back(base - i * stride);
  }
  return addresses;
}

// Generates interleaved ascending/descending addresses.
static std::vector<iree_hal_streaming_deviceptr_t>
GenerateInterleavedAscDescAddresses(
    size_t count, iree_hal_streaming_deviceptr_t low_base = 0x100000000ULL,
    iree_hal_streaming_deviceptr_t high_base = 0x700000000000ULL,
    size_t stride = 0x10000) {
  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (i % 2 == 0) {
      addresses.push_back(low_base + (i / 2) * stride);
    } else {
      addresses.push_back(high_base - (i / 2) * stride);
    }
  }
  return addresses;
}

// Generates random addresses with a fixed seed for reproducibility.
static std::vector<iree_hal_streaming_deviceptr_t> GenerateRandomAddresses(
    size_t count, uint32_t seed = 0x12345678) {
  std::mt19937_64 gen(seed);
  // Distribute addresses across the virtual address space.
  std::uniform_int_distribution<iree_hal_streaming_deviceptr_t> dist(
      0x100000000ULL, 0x700000000000ULL);

  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    addresses.push_back(dist(gen) & ~0xFULL);  // Align to 16 bytes.
  }
  return addresses;
}

// Generates sparse addresses with large gaps.
static std::vector<iree_hal_streaming_deviceptr_t> GenerateSparseAddresses(
    size_t count) {
  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);
  iree_hal_streaming_deviceptr_t base = 0x100000000ULL;
  for (size_t i = 0; i < count; ++i) {
    // Linearly increasing gaps that grow over time to avoid duplicates.
    // Use larger minimum gap and increasing multiplier.
    base += 0x10000 * (i + 1);  // Gaps: 0x10000, 0x20000, 0x30000, etc.
    addresses.push_back(base);
  }
  return addresses;
}

// Creates dummy buffers with the given addresses.
static std::vector<iree_hal_streaming_buffer_t*> CreateDummyBuffers(
    const std::vector<iree_hal_streaming_deviceptr_t>& addresses,
    iree_allocator_t allocator, bool with_host_ptr = false) {
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  buffers.reserve(addresses.size());
  for (size_t i = 0; i < addresses.size(); ++i) {
    iree_hal_streaming_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_allocator_malloc(
        allocator, sizeof(iree_hal_streaming_buffer_t), (void**)&buffer));
    buffer->device_ptr = addresses[i];
    // Set host_ptr if requested (use different address space).
    buffer->host_ptr =
        with_host_ptr ? reinterpret_cast<void*>(0x800000000ULL + i * 0x1000)
                      : nullptr;
    buffer->size = 0x1000;  // 4KB default size.
    buffers.push_back(buffer);
  }
  return buffers;
}

// Frees dummy buffers.
static void FreeDummyBuffers(
    const std::vector<iree_hal_streaming_buffer_t*>& buffers,
    iree_allocator_t allocator) {
  for (auto* buffer : buffers) {
    iree_allocator_free(allocator, buffer);
  }
}

//===----------------------------------------------------------------------===//
// Insertion benchmarks
//===----------------------------------------------------------------------===//

// Measures insertion performance with monotonically increasing addresses.
// Best case for append-based data structures.
IREE_BENCHMARK_FN(BM_InsertAscending) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateAscendingAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_InsertAscending);

// Measures insertion performance with monotonically decreasing addresses.
// Worst case for append-based structures, requires shifting/prepending.
IREE_BENCHMARK_FN(BM_InsertDescending) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateDescendingAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_InsertDescending);

// Measures insertion with alternating low and high addresses.
// Tests performance with insertions at both ends of the address space.
IREE_BENCHMARK_FN(BM_InsertInterleavedAscDesc) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateInterleavedAscDescAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_InsertInterleavedAscDesc);

// Measures insertion with random address ordering.
// Represents realistic allocation patterns with unpredictable addresses.
IREE_BENCHMARK_FN(BM_InsertRandom) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_InsertRandom);

// Measures insertion with exponentially growing gaps between addresses.
// Tests behavior with sparse virtual address usage.
IREE_BENCHMARK_FN(BM_InsertSparseAddresses) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateSparseAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_InsertSparseAddresses);

//===----------------------------------------------------------------------===//
// Lookup benchmarks
//===----------------------------------------------------------------------===//

// Measures lookup performance for existing buffers.
// Best case scenario with guaranteed hits.
IREE_BENCHMARK_FN(BM_LookupExactHit) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
  for (size_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
  }

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
      iree_optimization_barrier(found);
    }
  }

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LookupExactHit);

// Measures lookup performance for non-existent addresses.
// Tests worst-case search through entire table without finding target.
IREE_BENCHMARK_FN(BM_LookupExactMiss) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
  for (size_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
  }

  // Generate different addresses for misses.
  auto miss_addresses = GenerateRandomAddresses(count, 0x87654321);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      iree_status_t status = iree_hal_streaming_buffer_table_lookup(
          table, miss_addresses[i], &found);
      iree_status_ignore(status);
      iree_optimization_barrier(found);
    }
  }

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LookupExactMiss);

// Measures range lookup performance for addresses within buffer bounds.
// Tests additional overhead of range checking vs exact match.
IREE_BENCHMARK_FN(BM_LookupRangeHit) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateAscendingAddresses(count, 0x100000000ULL, 0x10000);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  // Set realistic sizes for buffers.
  for (size_t i = 0; i < count; ++i) {
    buffers[i]->size = 0x8000;  // 32KB buffers.
  }

  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
  for (size_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
  }

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      // Lookup in the middle of each buffer.
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_lookup_range(
          table, addresses[i] + 0x1000, 0x1000, &found));
      iree_optimization_barrier(found);
    }
  }

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LookupRangeHit);

// Measures lookup performance degradation with large table size.
// Tests scalability with 10,000 entries.
IREE_BENCHMARK_FN(BM_LookupAfterManyInserts) {
  iree_allocator_t allocator = iree_allocator_system();

  // Test with a single large table size to see lookup performance.
  const size_t count = 10000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  iree_hal_streaming_buffer_table_t* table = nullptr;
  IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
  for (size_t i = 0; i < count; ++i) {
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
  }

  // Measure lookup time with this large table.
  const size_t lookup_count = 100;
  while (iree_benchmark_keep_running(benchmark_state, lookup_count)) {
    for (size_t i = 0; i < lookup_count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_lookup(
          table, addresses[i % count], &found));
      iree_optimization_barrier(found);
    }
  }

  iree_hal_streaming_buffer_table_free(table);
  FreeDummyBuffers(buffers, allocator);

  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LookupAfterManyInserts);

//===----------------------------------------------------------------------===//
// Removal benchmarks
//===----------------------------------------------------------------------===//

// Measures removal in insertion order (first-in-first-out).
// Best case for structures that maintain insertion order.
IREE_BENCHMARK_FN(BM_RemoveFIFO) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Remove in same order (FIFO).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_remove(table, addresses[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_RemoveFIFO);

// Measures removal in reverse insertion order (last-in-first-out).
// Optimal for stack-like removal from end of array.
IREE_BENCHMARK_FN(BM_RemoveLIFO) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Remove in reverse order (LIFO).
    for (size_t i = count; i > 0; --i) {
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_remove(table, addresses[i - 1]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_RemoveLIFO);

// Measures removal in random order.
// Tests performance with unpredictable removal patterns and shifting.
IREE_BENCHMARK_FN(BM_RemoveRandom) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  // Create random removal order.
  std::vector<size_t> removal_order(count);
  std::iota(removal_order.begin(), removal_order.end(), 0);
  std::mt19937 gen(0xABCDEF);
  std::shuffle(removal_order.begin(), removal_order.end(), gen);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Remove in random order.
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_remove(
          table, addresses[removal_order[i]]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_RemoveRandom);

//===----------------------------------------------------------------------===//
// Mixed operation benchmarks
//===----------------------------------------------------------------------===//

// Measures interleaved insert and lookup operations.
// Simulates realistic usage with growing table and concurrent queries.
IREE_BENCHMARK_FN(BM_MixedInsertLookup) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count * 2)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    // Interleave inserts and lookups.
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));

      // Lookup a previously inserted buffer.
      if (i > 0) {
        iree_hal_streaming_buffer_t* found = nullptr;
        IREE_CHECK_OK(iree_hal_streaming_buffer_table_lookup(
            table, addresses[i / 2], &found));
        iree_optimization_barrier(found);
      }
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_MixedInsertLookup);

// Measures buffer replacement patterns (remove old, insert new).
// Simulates memory reallocation scenarios.
IREE_BENCHMARK_FN(BM_MixedInsertRemove) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses =
      GenerateRandomAddresses(count * 2);  // Need extra for replacements.
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count * 2)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert initial set (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Replace half with new buffers.
    for (size_t i = 0; i < count / 2; ++i) {
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_remove(table, addresses[i]));
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_insert(table, buffers[count + i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_MixedInsertRemove);

//===----------------------------------------------------------------------===//
// Stress test benchmarks
//===----------------------------------------------------------------------===//

// Stress test with 10,000 buffer insertions.
// Measures scalability and memory allocation overhead.
IREE_BENCHMARK_FN(BM_LargeScale) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 10000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LargeScale);

// Simulates fragmented virtual memory with clustered allocations.
// Tests performance with non-uniform address distribution.
IREE_BENCHMARK_FN(BM_FragmentedMemory) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;

  // Create a fragmented pattern: small buffers scattered across address space.
  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);
  std::mt19937_64 gen(0x11111111);

  // Create clusters of addresses.
  const size_t clusters = 10;
  const size_t per_cluster = count / clusters;
  for (size_t c = 0; c < clusters; ++c) {
    iree_hal_streaming_deviceptr_t cluster_base =
        0x100000000ULL + c * 0x100000000000ULL;
    for (size_t i = 0; i < per_cluster; ++i) {
      // Combine deterministic base (i * 0x10000) with random offset.
      // This ensures uniqueness while maintaining randomization.
      // The i * 0x10000 guarantees each address has a unique 64KB slot.
      // The random part adds variation within a 16KB range.
      uint64_t deterministic_offset = i * 0x10000;  // 64KB slots.
      uint64_t random_offset = (gen() % 0x4000);    // Random within 16KB.
      addresses.push_back(cluster_base + deterministic_offset + random_offset);
    }
  }

  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count * 2)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all buffers (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Perform lookups across clusters.
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_lookup(
          table, addresses[(i * 7) % count], &found));
      iree_optimization_barrier(found);
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_FragmentedMemory);

// Worst-case scenario for linear search implementation.
// Inserts in reverse order, lookups in forward order to maximize search
// distance.
IREE_BENCHMARK_FN(BM_PathologicalPattern) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;

  // Create a pathological pattern for linear search:
  // All addresses hash to similar values or are in worst-case order.
  std::vector<iree_hal_streaming_deviceptr_t> addresses;
  addresses.reserve(count);

  // Pattern 1: All addresses differ only in low bits (worst for some hashes).
  iree_hal_streaming_deviceptr_t base = 0x700000000000ULL;
  for (size_t i = 0; i < count; ++i) {
    addresses.push_back(base | (i << 4));
  }

  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert in reverse order to maximize search distance (not timed).
    for (size_t i = count; i > 0; --i) {
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_insert(table, buffers[i - 1]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Lookup in forward order (worst case for linear search).
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_lookup(table, addresses[i], &found));
      iree_optimization_barrier(found);
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_PathologicalPattern);

//===----------------------------------------------------------------------===//
// Host pointer benchmarks
//===----------------------------------------------------------------------===//

// Measures insertion performance with host pointers assigned.
IREE_BENCHMARK_FN(BM_InsertWithHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateAscendingAddresses(count);
  auto buffers =
      CreateDummyBuffers(addresses, allocator, true);  // with_host_ptr

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));
    iree_benchmark_resume_timing(benchmark_state);

    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_InsertWithHostPointer);

// Measures lookup performance using host pointers.
IREE_BENCHMARK_FN(BM_LookupHostPointer) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers =
      CreateDummyBuffers(addresses, allocator, true);  // with_host_ptr

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all buffers (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Lookup using host pointers.
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      uint64_t host_addr = (uint64_t)(uintptr_t)buffers[i]->host_ptr;
      IREE_CHECK_OK(
          iree_hal_streaming_buffer_table_lookup(table, host_addr, &found));
      iree_optimization_barrier(found);
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LookupHostPointer);

// Measures lookup performance with mid-buffer pointers.
IREE_BENCHMARK_FN(BM_LookupMidBuffer) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);
  auto buffers = CreateDummyBuffers(addresses, allocator);

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all buffers (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    // Generate mid-buffer offsets.
    std::vector<iree_hal_streaming_any_ptr_t> lookup_addrs;
    for (size_t i = 0; i < count; ++i) {
      // Use random offset within buffer (0 to size-1).
      uint32_t offset = (i * 31) % buffers[i]->size;  // Deterministic pattern.
      lookup_addrs.push_back(addresses[i] + offset);
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Lookup using mid-buffer pointers.
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_lookup(
          table, lookup_addrs[i], &found));
      iree_optimization_barrier(found);
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_LookupMidBuffer);

// Measures mixed workload: 98% device pointer lookups, 2% host pointer lookups.
IREE_BENCHMARK_FN(BM_MixedPointerTypes) {
  iree_allocator_t allocator = iree_allocator_system();
  const size_t count = 1000;
  auto addresses = GenerateRandomAddresses(count);

  // Create buffers with ~50% having host pointers to ensure coverage.
  std::vector<iree_hal_streaming_buffer_t*> buffers;
  buffers.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    iree_hal_streaming_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_allocator_malloc(
        allocator, sizeof(iree_hal_streaming_buffer_t), (void**)&buffer));
    buffer->device_ptr = addresses[i];
    // Half have host pointers.
    buffer->host_ptr =
        (i % 2 == 0) ? reinterpret_cast<void*>(0x800000000ULL + i * 0x1000)
                     : nullptr;
    buffer->size = 0x1000;
    buffers.push_back(buffer);
  }

  while (iree_benchmark_keep_running(benchmark_state, count)) {
    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_t* table = nullptr;
    IREE_CHECK_OK(iree_hal_streaming_buffer_table_allocate(allocator, &table));

    // Insert all buffers (not timed).
    for (size_t i = 0; i < count; ++i) {
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_insert(table, buffers[i]));
    }

    // Prepare lookup pattern: 98% device, 2% host.
    std::vector<iree_hal_streaming_any_ptr_t> lookup_addrs;
    for (size_t i = 0; i < count; ++i) {
      if (i % 50 == 0 && buffers[i]->host_ptr) {
        // 2% host pointer lookups.
        lookup_addrs.push_back((uint64_t)(uintptr_t)buffers[i]->host_ptr);
      } else {
        // 98% device pointer lookups.
        lookup_addrs.push_back(buffers[i]->device_ptr);
      }
    }

    iree_benchmark_resume_timing(benchmark_state);

    // Perform mixed lookups.
    for (size_t i = 0; i < count; ++i) {
      iree_hal_streaming_buffer_t* found = nullptr;
      IREE_CHECK_OK(iree_hal_streaming_buffer_table_lookup(
          table, lookup_addrs[i], &found));
      iree_optimization_barrier(found);
    }

    iree_benchmark_pause_timing(benchmark_state);
    iree_hal_streaming_buffer_table_free(table);
    iree_benchmark_resume_timing(benchmark_state);
  }

  FreeDummyBuffers(buffers, allocator);
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_MixedPointerTypes);
