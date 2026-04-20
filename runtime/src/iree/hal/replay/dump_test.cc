// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump.h"

#include <cstdint>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::testing::HasSubstr;

static iree_status_t AppendToString(void* user_data, iree_string_view_t text) {
  auto* output = static_cast<std::string*>(user_data);
  output->append(text.data, text.size);
  return iree_ok_status();
}

static std::vector<uint8_t> MakeReplayFileStorage() {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_file_record_metadata_t session_metadata = {};
  session_metadata.sequence_ordinal = 0;
  session_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &session_metadata, 0, nullptr, nullptr));

  iree_hal_replay_buffer_object_payload_t buffer_payload = {};
  buffer_payload.allocation_size = 256;
  buffer_payload.byte_length = 64;
  buffer_payload.allowed_usage = 0x11;
  iree_const_byte_span_t buffer_payload_span =
      iree_make_const_byte_span(&buffer_payload, sizeof(buffer_payload));
  iree_hal_replay_file_record_metadata_t object_metadata = {};
  object_metadata.sequence_ordinal = 1;
  object_metadata.object_id = 7;
  object_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  object_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT;
  object_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &object_metadata, 1, &buffer_payload_span, nullptr));

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static iree_const_byte_span_t MakeReplayFileContents(
    const std::vector<uint8_t>& storage) {
  auto* file_header =
      reinterpret_cast<const iree_hal_replay_file_header_t*>(storage.data());
  return iree_make_const_byte_span(
      storage.data(), static_cast<iree_host_size_t>(file_header->file_length));
}

TEST(ReplayDumpTest, EmitsTextSummary) {
  std::vector<uint8_t> storage = MakeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();

  std::string output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(MakeReplayFileContents(storage),
                                           &options, AppendToString, &output,
                                           iree_allocator_system()));

  EXPECT_THAT(output, HasSubstr("IREE HAL replay v1.0"));
  EXPECT_THAT(output, HasSubstr("summary:"));
  EXPECT_THAT(output, HasSubstr("hermetic: yes"));
  EXPECT_THAT(output, HasSubstr("strict_replay_supported: yes"));
  EXPECT_THAT(output, HasSubstr("files: total=0 external=0 inline=0"));
  EXPECT_THAT(output, HasSubstr("#0 session"));
  EXPECT_THAT(output, HasSubstr("#1 object"));
  EXPECT_THAT(output, HasSubstr("object=buffer"));
  EXPECT_THAT(output, HasSubstr("payload=buffer_object"));
  EXPECT_THAT(output, HasSubstr("allocation_size=256"));
}

TEST(ReplayDumpTest, EmitsJsonlWithPayloadRanges) {
  std::vector<uint8_t> storage = MakeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;

  std::string output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(MakeReplayFileContents(storage),
                                           &options, AppendToString, &output,
                                           iree_allocator_system()));

  EXPECT_THAT(output, HasSubstr("\"kind\":\"file\""));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"summary\""));
  EXPECT_THAT(output, HasSubstr("\"hermetic\":true"));
  EXPECT_THAT(output, HasSubstr("\"environment_referenced\":false"));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"session\""));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"object\""));
  EXPECT_THAT(output, HasSubstr("\"payload_type\":\"buffer_object\""));
  EXPECT_THAT(output, HasSubstr("\"payload_range\""));
  EXPECT_THAT(output, HasSubstr("\"allocation_size\":256"));
}

TEST(ReplayDumpTest, EmitsBufferRangeDataRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_buffer_range_data_payload_t payload = {};
  payload.byte_offset = 64;
  payload.byte_length = 4;
  payload.data_length = 4;
  payload.memory_access = IREE_HAL_MEMORY_ACCESS_WRITE;
  const uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(writer, &metadata, 2,
                                                           iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(MakeReplayFileContents(storage),
                                           &options, AppendToString, &output,
                                           iree_allocator_system()));

  EXPECT_THAT(output, HasSubstr("\"payload_type\":\"buffer_range_data\""));
  EXPECT_THAT(output, HasSubstr("\"data_range\""));
  EXPECT_THAT(output, HasSubstr("\"length\":4"));
}

TEST(ReplayDumpTest, EmitsQueueAllocaSemaphoreRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_device_queue_alloca_payload_t payload = {};
  payload.allocation.allocation_size = 4096;
  payload.allocation.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  payload.allocation.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  payload.allocation.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  payload.allocation.access = IREE_HAL_MEMORY_ACCESS_ALL;
  payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  payload.signal_semaphore_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 42;
  signal.value = 7;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(writer, &metadata, 2,
                                                           iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output, HasSubstr("payload=device_queue_alloca"));
  EXPECT_THAT(text_output, HasSubstr("submit_queue_affinity="));
  EXPECT_THAT(text_output, HasSubstr("wait_range="));
  EXPECT_THAT(text_output, HasSubstr("signal_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"device_queue_alloca\""));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"signal_semaphores_range\""));
}

TEST(ReplayDumpTest, EmitsQueueExecuteTables) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_device_queue_execute_payload_t payload = {};
  payload.command_buffer_id = 9;
  payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  payload.wait_semaphore_count = 1;
  payload.signal_semaphore_count = 1;
  payload.binding_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t wait = {};
  wait.semaphore_id = 42;
  wait.value = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 43;
  signal.value = 2;
  iree_hal_replay_buffer_ref_payload_t binding = {};
  binding.buffer_id = 7;
  binding.offset = 64;
  binding.length = 128;
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
      iree_make_const_byte_span(&binding, sizeof(binding)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output, HasSubstr("payload=device_queue_execute"));
  EXPECT_THAT(text_output, HasSubstr("wait_semaphores=[{semaphore_id=42"));
  EXPECT_THAT(text_output, HasSubstr("signal_semaphores=[{semaphore_id=43"));
  EXPECT_THAT(text_output, HasSubstr("bindings=[{buffer_id=7"));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"device_queue_execute\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"wait_semaphores\":[{\"semaphore_id\":42"));
  EXPECT_THAT(json_output,
              HasSubstr("\"signal_semaphores\":[{\"semaphore_id\":43"));
  EXPECT_THAT(json_output,
              HasSubstr("\"bindings\":[{\"buffer_id\":7,\"offset\":64"));
}

TEST(ReplayDumpTest, EmitsExecutionBarrierRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_command_buffer_execution_barrier_payload_t payload = {};
  payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER;
  payload.memory_barrier_count = 1;
  iree_hal_replay_memory_barrier_payload_t memory_barrier = {};
  memory_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  memory_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&memory_barrier, sizeof(memory_barrier)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(writer, &metadata, 2,
                                                           iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output,
              HasSubstr("payload=command_buffer_execution_barrier"));
  EXPECT_THAT(text_output, HasSubstr("memory_barriers_range="));
  EXPECT_THAT(text_output, HasSubstr("buffer_barriers_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(
      json_output,
      HasSubstr("\"payload_type\":\"command_buffer_execution_barrier\""));
  EXPECT_THAT(json_output, HasSubstr("\"memory_barriers_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"buffer_barriers_range\""));
}

TEST(ReplayDumpTest, EmitsEventPayloads) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_event_object_payload_t event_payload = {};
  event_payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  iree_const_byte_span_t event_iovec =
      iree_make_const_byte_span(&event_payload, sizeof(event_payload));
  iree_hal_replay_file_record_metadata_t event_metadata = {};
  event_metadata.sequence_ordinal = 0;
  event_metadata.object_id = 7;
  event_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  event_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_EVENT_OBJECT;
  event_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_EVENT;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &event_metadata, 1, &event_iovec, nullptr));

  iree_hal_replay_command_buffer_event_payload_t signal_payload = {};
  signal_payload.event_id = 7;
  signal_payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;
  iree_const_byte_span_t signal_iovec =
      iree_make_const_byte_span(&signal_payload, sizeof(signal_payload));
  iree_hal_replay_file_record_metadata_t signal_metadata = {};
  signal_metadata.sequence_ordinal = 1;
  signal_metadata.object_id = 8;
  signal_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  signal_metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EVENT;
  signal_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  signal_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_SIGNAL_EVENT;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &signal_metadata, 1, &signal_iovec, nullptr));

  iree_hal_replay_command_buffer_wait_events_payload_t wait_payload = {};
  wait_payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;
  wait_payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE;
  wait_payload.event_count = 1;
  wait_payload.memory_barrier_count = 1;
  wait_payload.buffer_barrier_count = 1;
  iree_hal_replay_object_id_t event_id = 7;
  iree_hal_replay_memory_barrier_payload_t memory_barrier = {};
  memory_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  memory_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
  iree_hal_replay_buffer_barrier_payload_t buffer_barrier = {};
  buffer_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
  buffer_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
  buffer_barrier.buffer_ref.buffer_id = 9;
  buffer_barrier.buffer_ref.length = 32;
  iree_const_byte_span_t wait_iovecs[4] = {
      iree_make_const_byte_span(&wait_payload, sizeof(wait_payload)),
      iree_make_const_byte_span(&event_id, sizeof(event_id)),
      iree_make_const_byte_span(&memory_barrier, sizeof(memory_barrier)),
      iree_make_const_byte_span(&buffer_barrier, sizeof(buffer_barrier)),
  };
  iree_hal_replay_file_record_metadata_t wait_metadata = {};
  wait_metadata.sequence_ordinal = 2;
  wait_metadata.object_id = 8;
  wait_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  wait_metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_WAIT_EVENTS;
  wait_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  wait_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_WAIT_EVENTS;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &wait_metadata, IREE_ARRAYSIZE(wait_iovecs), wait_iovecs,
      nullptr));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output, HasSubstr("payload=event_object"));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_event"));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_wait_events"));
  EXPECT_THAT(text_output, HasSubstr("events_range="));
  EXPECT_THAT(text_output, HasSubstr("memory_barriers_range="));
  EXPECT_THAT(text_output, HasSubstr("buffer_barriers_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"event_object\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_event\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_wait_events\""));
  EXPECT_THAT(json_output, HasSubstr("\"events_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"memory_barriers_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"buffer_barriers_range\""));
}

TEST(ReplayDumpTest, EmitsFilePayloads) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_file_object_payload_t file_payload = {};
  file_payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  file_payload.file_length = 4096;
  file_payload.file_device = 10;
  file_payload.file_inode = 20;
  file_payload.file_mtime_ns = 30;
  file_payload.access = IREE_HAL_MEMORY_ACCESS_READ;
  file_payload.handle_type = IREE_IO_FILE_HANDLE_TYPE_FD;
  file_payload.reference_type =
      IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH;
  file_payload.validation_type = IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_IDENTITY;
  file_payload.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  const char file_reference[] = "/tmp/model.irpa";
  file_payload.reference_length = sizeof(file_reference) - 1;
  iree_const_byte_span_t file_iovecs[2] = {
      iree_make_const_byte_span(&file_payload, sizeof(file_payload)),
      iree_make_const_byte_span(file_reference, sizeof(file_reference) - 1),
  };
  iree_hal_replay_file_record_metadata_t file_metadata = {};
  file_metadata.sequence_ordinal = 0;
  file_metadata.object_id = 7;
  file_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  file_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT;
  file_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_FILE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &file_metadata, IREE_ARRAYSIZE(file_iovecs), file_iovecs,
      nullptr));

  iree_hal_replay_device_queue_read_payload_t read_payload = {};
  read_payload.source_file_id = 7;
  read_payload.source_offset = 64;
  read_payload.target_ref.buffer_id = 8;
  read_payload.target_ref.length = 16;
  read_payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  read_payload.signal_semaphore_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 9;
  signal.value = 1;
  iree_const_byte_span_t read_iovecs[2] = {
      iree_make_const_byte_span(&read_payload, sizeof(read_payload)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
  };
  iree_hal_replay_file_record_metadata_t read_metadata = {};
  read_metadata.sequence_ordinal = 1;
  read_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  read_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ;
  read_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE;
  read_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &read_metadata, IREE_ARRAYSIZE(read_iovecs), read_iovecs,
      nullptr));

  iree_hal_replay_device_queue_write_payload_t write_payload = {};
  write_payload.source_ref.buffer_id = 8;
  write_payload.source_ref.length = 16;
  write_payload.target_file_id = 7;
  write_payload.target_offset = 128;
  write_payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  write_payload.wait_semaphore_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t wait = {};
  wait.semaphore_id = 9;
  wait.value = 1;
  iree_const_byte_span_t write_iovecs[2] = {
      iree_make_const_byte_span(&write_payload, sizeof(write_payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
  };
  iree_hal_replay_file_record_metadata_t write_metadata = {};
  write_metadata.sequence_ordinal = 2;
  write_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  write_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE;
  write_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE;
  write_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &write_metadata, IREE_ARRAYSIZE(write_iovecs), write_iovecs,
      nullptr));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output, HasSubstr("payload=file_object"));
  EXPECT_THAT(text_output, HasSubstr("hermetic: no"));
  EXPECT_THAT(text_output, HasSubstr("environment_referenced: yes"));
  EXPECT_THAT(text_output, HasSubstr("files: total=1 external=1 inline=0"));
  EXPECT_THAT(text_output, HasSubstr("file_validation: identity=1"));
  EXPECT_THAT(text_output, HasSubstr("reference_type=external_path(1)"));
  EXPECT_THAT(text_output, HasSubstr("validation_type=identity(1)"));
  EXPECT_THAT(text_output, HasSubstr("reference_range="));
  EXPECT_THAT(text_output, HasSubstr("payload=device_queue_read"));
  EXPECT_THAT(text_output, HasSubstr("source_file_id=7"));
  EXPECT_THAT(text_output, HasSubstr("payload=device_queue_write"));
  EXPECT_THAT(text_output, HasSubstr("target_file_id=7"));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"file_object\""));
  EXPECT_THAT(json_output, HasSubstr("\"kind\":\"summary\""));
  EXPECT_THAT(json_output, HasSubstr("\"hermetic\":false"));
  EXPECT_THAT(json_output, HasSubstr("\"environment_referenced\":true"));
  EXPECT_THAT(json_output, HasSubstr("\"external_file_count\":1"));
  EXPECT_THAT(json_output, HasSubstr("\"identity\":1"));
  EXPECT_THAT(json_output,
              HasSubstr("\"reference_type_name\":\"external_path\""));
  EXPECT_THAT(json_output, HasSubstr("\"validation_type_name\":\"identity\""));
  EXPECT_THAT(json_output, HasSubstr("\"reference_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"device_queue_read\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"device_queue_write\""));
  EXPECT_THAT(json_output, HasSubstr("\"source_file_id\":7"));
  EXPECT_THAT(json_output, HasSubstr("\"target_file_id\":7"));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"signal_semaphores_range\""));
}

TEST(ReplayDumpTest, EmitsQueueTransferRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_device_queue_update_payload_t payload = {};
  payload.target_ref.buffer_id = 7;
  payload.target_ref.length = 4;
  payload.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  payload.source_offset = 3;
  payload.data_length = 4;
  payload.wait_semaphore_count = 1;
  payload.signal_semaphore_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t wait = {};
  wait.semaphore_id = 42;
  wait.value = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 43;
  signal.value = 2;
  const uint8_t data[] = {0x10, 0x11, 0x12, 0x13};
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output, HasSubstr("payload=device_queue_update"));
  EXPECT_THAT(text_output, HasSubstr("wait_range="));
  EXPECT_THAT(text_output, HasSubstr("signal_range="));
  EXPECT_THAT(text_output, HasSubstr("data_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"device_queue_update\""));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"signal_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"data_range\""));
}

TEST(ReplayDumpTest, EmitsCommandBufferTransferRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_create(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_command_buffer_fill_buffer_payload_t fill_payload = {};
  fill_payload.target_ref.buffer_id = 7;
  fill_payload.target_ref.length = 4;
  fill_payload.pattern_length = 4;
  const uint32_t pattern = 0xA5A5A5A5u;
  iree_const_byte_span_t fill_iovecs[2] = {
      iree_make_const_byte_span(&fill_payload, sizeof(fill_payload)),
      iree_make_const_byte_span(&pattern, sizeof(pattern)),
  };
  iree_hal_replay_file_record_metadata_t fill_metadata = {};
  fill_metadata.sequence_ordinal = 0;
  fill_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  fill_metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER;
  fill_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  fill_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &fill_metadata, IREE_ARRAYSIZE(fill_iovecs), fill_iovecs,
      nullptr));

  iree_hal_replay_command_buffer_update_buffer_payload_t update_payload = {};
  update_payload.target_ref.buffer_id = 7;
  update_payload.target_ref.offset = 4;
  update_payload.target_ref.length = 4;
  update_payload.source_offset = 2;
  update_payload.data_length = 4;
  const uint8_t data[] = {0x20, 0x21, 0x22, 0x23};
  iree_const_byte_span_t update_iovecs[2] = {
      iree_make_const_byte_span(&update_payload, sizeof(update_payload)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  iree_hal_replay_file_record_metadata_t update_metadata = {};
  update_metadata.sequence_ordinal = 1;
  update_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  update_metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER;
  update_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  update_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &update_metadata, IREE_ARRAYSIZE(update_iovecs), update_iovecs,
      nullptr));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &text_options, AppendToString,
      &text_output, iree_allocator_system()));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_fill_buffer"));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_update_buffer"));
  EXPECT_THAT(text_output, HasSubstr("pattern_range="));
  EXPECT_THAT(text_output, HasSubstr("data_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(iree_hal_replay_dump_file(
      MakeReplayFileContents(storage), &json_options, AppendToString,
      &json_output, iree_allocator_system()));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_fill_buffer\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_update_buffer\""));
  EXPECT_THAT(json_output, HasSubstr("\"pattern_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"data_range\""));
}

}  // namespace
