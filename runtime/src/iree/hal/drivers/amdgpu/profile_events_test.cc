// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_events.h"

#include <cstring>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct CapturedChunk {
  // Metadata copied from the sink write callback.
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();

  // Concatenated iovec payload copied from the sink write callback.
  std::vector<uint8_t> payload;
};

struct CapturingProfileSink {
  // HAL resource header for the sink.
  iree_hal_resource_t resource;

  // Chunks copied from sink write callbacks.
  std::vector<CapturedChunk> chunks;

  // Optional stream used to force a queue-event drop during a write callback.
  iree_hal_amdgpu_profile_event_streams_t* queue_streams_to_record = nullptr;

  // Queue event used when |queue_streams_to_record| is set.
  iree_hal_profile_queue_event_t queue_event_to_record = {};

  // Number of queue events to append during the next write callback.
  int queue_event_record_count_on_write = 0;
};

static CapturingProfileSink* CapturingProfileSinkCast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<CapturingProfileSink*>(sink);
}

static void CapturingProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t CapturingProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)sink;
  (void)metadata;
  return iree_ok_status();
}

static iree_status_t CapturingProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  auto* captured_sink = CapturingProfileSinkCast(sink);
  CapturedChunk chunk;
  chunk.metadata = *metadata;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    const uint8_t* source = iovecs[i].data;
    chunk.payload.insert(chunk.payload.end(), source,
                         source + iovecs[i].data_length);
  }
  captured_sink->chunks.push_back(std::move(chunk));
  while (captured_sink->queue_event_record_count_on_write > 0) {
    --captured_sink->queue_event_record_count_on_write;
    iree_hal_amdgpu_profile_event_streams_record_queue_event(
        captured_sink->queue_streams_to_record,
        &captured_sink->queue_event_to_record);
  }
  return iree_ok_status();
}

static iree_status_t CapturingProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kCapturingProfileSinkVTable = {
    .destroy = CapturingProfileSinkDestroy,
    .begin_session = CapturingProfileSinkBeginSession,
    .write = CapturingProfileSinkWrite,
    .end_session = CapturingProfileSinkEndSession,
};

static void CapturingProfileSinkInitialize(CapturingProfileSink* sink) {
  iree_hal_resource_initialize(&kCapturingProfileSinkVTable, &sink->resource);
}

static iree_hal_profile_sink_t* CapturingProfileSinkAsBase(
    CapturingProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

template <typename T>
static std::vector<T> DecodeRecords(const CapturedChunk& chunk) {
  EXPECT_EQ(0u, chunk.payload.size() % sizeof(T));
  std::vector<T> records(chunk.payload.size() / sizeof(T));
  memcpy(records.data(), chunk.payload.data(), chunk.payload.size());
  return records;
}

class ProfileEventsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amdgpu_profile_event_streams_initialize(&streams_);
    CapturingProfileSinkInitialize(&sink_);
  }

  void TearDown() override {
    iree_hal_amdgpu_profile_event_streams_deinitialize(&streams_,
                                                       iree_allocator_system());
  }

  iree_hal_amdgpu_profile_event_streams_t streams_ = {};
  CapturingProfileSink sink_ = {};
};

TEST_F(ProfileEventsTest, MemoryEventsPreserveRecordsAndReportDrops) {
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_ensure_memory_storage(
      &streams_, /*event_capacity=*/2, iree_allocator_system()));
  iree_hal_amdgpu_profile_event_streams_clear_memory(&streams_);

  uint64_t session_id = 0;
  const uint64_t allocation_id =
      iree_hal_amdgpu_profile_event_streams_allocate_memory_allocation_id(
          &streams_, /*active_session_id=*/7, &session_id);
  EXPECT_EQ(7u, session_id);
  EXPECT_EQ(1u, allocation_id);

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE;
  event.allocation_id = allocation_id;
  event.length = 4096;

  EXPECT_TRUE(iree_hal_amdgpu_profile_event_streams_record_memory_event(
      &streams_, /*active_session_id=*/7, session_id, &event));
  event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE;
  EXPECT_TRUE(iree_hal_amdgpu_profile_event_streams_record_memory_event(
      &streams_, /*active_session_id=*/7, session_id, &event));
  EXPECT_FALSE(iree_hal_amdgpu_profile_event_streams_record_memory_event(
      &streams_, /*active_session_id=*/7, session_id, &event));

  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_write_memory(
      &streams_, CapturingProfileSinkAsBase(&sink_), session_id,
      iree_allocator_system()));

  ASSERT_EQ(1u, sink_.chunks.size());
  const CapturedChunk& chunk = sink_.chunks.front();
  EXPECT_TRUE(
      iree_string_view_equal(IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS,
                             chunk.metadata.content_type));
  EXPECT_EQ(7u, chunk.metadata.session_id);
  EXPECT_TRUE(iree_all_bits_set(chunk.metadata.flags,
                                IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED));
  EXPECT_EQ(1u, chunk.metadata.dropped_record_count);

  std::vector<iree_hal_profile_memory_event_t> records =
      DecodeRecords<iree_hal_profile_memory_event_t>(chunk);
  ASSERT_EQ(2u, records.size());
  EXPECT_EQ(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE,
            records[0].type);
  EXPECT_EQ(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE, records[1].type);
  EXPECT_EQ(1u, records[0].event_id);
  EXPECT_EQ(2u, records[1].event_id);
  EXPECT_NE(0, records[0].host_time_ns);
}

TEST_F(ProfileEventsTest, MemoryEventsSkipMismatchedSession) {
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_ensure_memory_storage(
      &streams_, /*event_capacity=*/2, iree_allocator_system()));
  iree_hal_amdgpu_profile_event_streams_clear_memory(&streams_);

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE;
  EXPECT_FALSE(iree_hal_amdgpu_profile_event_streams_record_memory_event(
      &streams_, /*active_session_id=*/7, /*session_id=*/9, &event));

  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_write_memory(
      &streams_, CapturingProfileSinkAsBase(&sink_), /*session_id=*/7,
      iree_allocator_system()));
  EXPECT_TRUE(sink_.chunks.empty());
}

TEST_F(ProfileEventsTest, QueueEventsReportRetainedAndMetadataOnlyDrops) {
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_ensure_queue_storage(
      &streams_, /*event_capacity=*/1, iree_allocator_system()));
  iree_hal_amdgpu_profile_event_streams_clear_queue(&streams_);

  iree_hal_profile_queue_event_t event = iree_hal_profile_queue_event_default();
  event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH;
  event.stream_id = 42;
  iree_hal_amdgpu_profile_event_streams_record_queue_event(&streams_, &event);
  iree_hal_amdgpu_profile_event_streams_record_queue_event(&streams_, &event);

  sink_.queue_streams_to_record = &streams_;
  sink_.queue_event_to_record = event;
  sink_.queue_event_record_count_on_write = 1;
  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_write_queue(
      &streams_, CapturingProfileSinkAsBase(&sink_), /*session_id=*/11,
      iree_allocator_system()));

  ASSERT_EQ(1u, sink_.chunks.size());
  const CapturedChunk& chunk = sink_.chunks.front();
  EXPECT_TRUE(iree_string_view_equal(IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
                                     chunk.metadata.content_type));
  EXPECT_EQ(11u, chunk.metadata.session_id);
  EXPECT_TRUE(iree_all_bits_set(chunk.metadata.flags,
                                IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED));
  EXPECT_EQ(1u, chunk.metadata.dropped_record_count);

  std::vector<iree_hal_profile_queue_event_t> records =
      DecodeRecords<iree_hal_profile_queue_event_t>(chunk);
  ASSERT_EQ(1u, records.size());
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, records[0].type);
  EXPECT_EQ(1u, records[0].event_id);
  EXPECT_EQ(42u, records[0].stream_id);

  IREE_ASSERT_OK(iree_hal_amdgpu_profile_event_streams_write_queue(
      &streams_, CapturingProfileSinkAsBase(&sink_), /*session_id=*/11,
      iree_allocator_system()));

  ASSERT_EQ(2u, sink_.chunks.size());
  const CapturedChunk& metadata_only_chunk = sink_.chunks.back();
  EXPECT_TRUE(
      iree_string_view_equal(IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
                             metadata_only_chunk.metadata.content_type));
  EXPECT_TRUE(iree_all_bits_set(metadata_only_chunk.metadata.flags,
                                IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED));
  EXPECT_EQ(1u, metadata_only_chunk.metadata.dropped_record_count);
  EXPECT_TRUE(metadata_only_chunk.payload.empty());
}

}  // namespace
}  // namespace iree::hal::amdgpu
