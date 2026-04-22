// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/memory.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_profile_file_record_t MakeChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk;
  memset(&chunk, 0, sizeof(chunk));
  chunk.header.record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;
  chunk.content_type = IREE_SV("application/vnd.iree.test");
  chunk.payload = iree_make_const_byte_span(payload.data(), payload.size());
  return chunk;
}

static iree_hal_profile_file_record_t MakeMemoryChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS;
  return chunk;
}

static void AppendMemoryEvent(std::vector<uint8_t>* payload,
                              iree_hal_profile_memory_event_type_t event_type,
                              uint64_t event_id, uint64_t allocation_id,
                              uint64_t pool_id, uint64_t length) {
  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = event_type;
  event.event_id = event_id;
  event.allocation_id = allocation_id;
  event.pool_id = pool_id;
  event.physical_device_ordinal = 0;
  event.queue_ordinal = 0;
  event.memory_type = 1;
  event.buffer_usage = 1;
  event.length = length;
  event.alignment = 1;
  switch (event_type) {
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE:
      event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA:
      event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
      event.submission_id = event_id;
      break;
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT:
    case IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT:
      event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED;
      break;
    default:
      break;
  }

  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(event));
  memcpy(payload->data() + offset, &event, sizeof(event));
}

static void AddPoolStatsToLastMemoryEvent(
    std::vector<uint8_t>* payload, uint64_t bytes_reserved, uint64_t bytes_free,
    uint64_t bytes_committed, uint64_t budget_limit, uint32_t reservation_count,
    uint32_t slab_count) {
  const iree_host_size_t offset =
      payload->size() - sizeof(iree_hal_profile_memory_event_t);
  iree_hal_profile_memory_event_t event;
  memcpy(&event, payload->data() + offset, sizeof(event));
  event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS;
  event.pool_bytes_reserved = bytes_reserved;
  event.pool_bytes_free = bytes_free;
  event.pool_bytes_committed = bytes_committed;
  event.pool_budget_limit = budget_limit;
  event.pool_reservation_count = reservation_count;
  event.pool_slab_count = slab_count;
  memcpy(payload->data() + offset, &event, sizeof(event));
}

typedef struct memory_event_collector_t {
  uint64_t event_count;
  uint64_t first_event_id;
  bool saw_truncated_chunk;
} memory_event_collector_t;

static iree_status_t CollectMemoryEvent(
    void* user_data, const iree_profile_memory_event_row_t* row) {
  memory_event_collector_t* collector =
      static_cast<memory_event_collector_t*>(user_data);
  if (collector->event_count == 0) {
    collector->first_event_id = row->event->event_id;
  }
  ++collector->event_count;
  collector->saw_truncated_chunk |= row->is_truncated;
  return iree_ok_status();
}

TEST(ProfileMemoryTest, SeparatesReserveMaterializedAndInflightBytes) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
                    1, 99, 7, 64);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
                    2, 1, 7, 256);
  AppendMemoryEvent(&payload,
                    IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE, 3, 1,
                    7, 256);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
                    4, 1, 7, 40);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
                    5, 1, 7, 40);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
                    6, 1, 7, 256);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{}));

  ASSERT_EQ(1u, context.device_count);
  const iree_profile_memory_device_t* device = &context.devices[0];
  EXPECT_EQ(0u, device->pool_reservation_balance.current_bytes);
  EXPECT_EQ(256u, device->pool_reservation_balance.high_water_bytes);
  EXPECT_EQ(64u, device->pool_reservation_balance.partial_close_bytes);
  EXPECT_EQ(0u, device->pool_materialization_balance.current_bytes);
  EXPECT_EQ(256u, device->pool_materialization_balance.high_water_bytes);
  EXPECT_EQ(0u, device->pool_materialization_balance.partial_close_count);
  EXPECT_EQ(0u, device->queue_inflight_balance.current_bytes);
  EXPECT_EQ(40u, device->queue_inflight_balance.high_water_bytes);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, InvokesEventCallbackForMatchedEvents) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
                    1, 1, 7, 256);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
                    2, 1, 7, 40);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);
  chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;

  memory_event_collector_t collector = {0};
  const iree_profile_memory_event_callback_t event_callback = {
      CollectMemoryEvent,
      &collector,
  };
  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("pool_*"), -1, event_callback));

  EXPECT_EQ(2u, context.total_event_count);
  EXPECT_EQ(1u, context.matched_event_count);
  EXPECT_EQ(1u, context.truncated_event_count);
  EXPECT_EQ(1u, collector.event_count);
  EXPECT_EQ(1u, collector.first_event_id);
  EXPECT_TRUE(collector.saw_truncated_chunk);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, RecordsZeroPayloadTruncatedChunks) {
  std::vector<uint8_t> payload;
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);
  chunk.header.chunk_flags = IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
  chunk.header.dropped_record_count = 7;

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{0}));

  EXPECT_EQ(1u, context.truncated_chunk_count);
  EXPECT_EQ(7u, context.dropped_record_count);
  EXPECT_EQ(0u, context.total_event_count);
  EXPECT_EQ(0u, context.matched_event_count);
  EXPECT_EQ(0u, context.truncated_event_count);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, RecordsPoolStatSnapshots) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
                    1, 1, 7, 256);
  AddPoolStatsToLastMemoryEvent(&payload, 256, 768, 1024, 2048, 1, 1);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
                    2, 1, 7, 128);
  AddPoolStatsToLastMemoryEvent(&payload, 256, 768, 1024, 2048, 1, 1);
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
                    3, 1, 7, 256);
  AddPoolStatsToLastMemoryEvent(&payload, 0, 1024, 1024, 2048, 0, 1);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{}));

  const iree_profile_memory_pool_t* pool = nullptr;
  for (iree_host_size_t i = 0; i < context.pool_count; ++i) {
    if (context.pools[i].kind ==
            IREE_PROFILE_MEMORY_LIFECYCLE_KIND_POOL_RESERVATION &&
        context.pools[i].pool_id == 7) {
      pool = &context.pools[i];
      break;
    }
  }
  ASSERT_NE(nullptr, pool);
  EXPECT_EQ(3u, pool->pool_stats_sample_count);
  EXPECT_EQ(0u, pool->pool_bytes_reserved);
  EXPECT_EQ(256u, pool->pool_bytes_reserved_high_water);
  EXPECT_EQ(1024u, pool->pool_bytes_free);
  EXPECT_EQ(768u, pool->pool_bytes_free_low_water);
  EXPECT_EQ(1024u, pool->pool_bytes_committed);
  EXPECT_EQ(1024u, pool->pool_bytes_committed_high_water);
  EXPECT_EQ(2048u, pool->pool_budget_limit);
  EXPECT_EQ(0u, pool->pool_reservation_count);
  EXPECT_EQ(1u, pool->pool_reservation_high_water_count);
  EXPECT_EQ(1u, pool->pool_slab_count);
  EXPECT_EQ(1u, pool->pool_slab_high_water_count);

  iree_profile_memory_context_deinitialize(&context);
}

TEST(ProfileMemoryTest, SeparatesImportedBufferBytes) {
  std::vector<uint8_t> payload;
  AppendMemoryEvent(&payload, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_IMPORT,
                    1, 11, 0, 4096);
  AppendMemoryEvent(&payload,
                    IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_UNIMPORT, 2, 11,
                    0, 4096);
  iree_hal_profile_file_record_t chunk = MakeMemoryChunk(payload);

  iree_profile_memory_context_t context;
  iree_profile_memory_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(iree_profile_memory_context_accumulate_record(
      &context, &chunk, IREE_SV("*"), -1,
      iree_profile_memory_event_callback_t{}));

  ASSERT_EQ(1u, context.device_count);
  const iree_profile_memory_device_t* device = &context.devices[0];
  EXPECT_EQ(1u, device->buffer_import_count);
  EXPECT_EQ(1u, device->buffer_unimport_count);
  EXPECT_EQ(0u, device->buffer_allocation_balance.total_open_count);
  EXPECT_EQ(0u, device->buffer_import_balance.current_bytes);
  EXPECT_EQ(4096u, device->buffer_import_balance.high_water_bytes);

  ASSERT_EQ(1u, context.allocation_count);
  const iree_profile_memory_allocation_t* allocation = &context.allocations[0];
  EXPECT_EQ(IREE_PROFILE_MEMORY_LIFECYCLE_KIND_IMPORTED_BUFFER,
            allocation->kind);
  EXPECT_TRUE(iree_all_bits_set(
      allocation->flags, IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_EXTERNALLY_OWNED));

  iree_profile_memory_context_deinitialize(&context);
}

}  // namespace
