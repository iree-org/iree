// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/bundle.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_hal_profile_file_record_t MakeChunk(
    iree_string_view_t content_type, const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk;
  memset(&chunk, 0, sizeof(chunk));
  chunk.header.record_type = IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK;
  chunk.content_type = content_type;
  chunk.payload = iree_make_const_byte_span(payload.data(), payload.size());
  return chunk;
}

static iree_status_t ParseChunk(iree_profile_att_profile_t* profile,
                                iree_string_view_t content_type,
                                const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(content_type, payload);
  return iree_profile_att_profile_parse_record(profile, &chunk);
}

template <typename T>
static void AppendRecord(std::vector<uint8_t>* payload, const T& record,
                         std::initializer_list<uint8_t> inline_payload = {}) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + record.record_length);
  memcpy(payload->data() + offset, &record, sizeof(record));
  if (inline_payload.size() > 0) {
    memcpy(payload->data() + offset + sizeof(record), inline_payload.begin(),
           inline_payload.size());
  }
}

static void AppendTraceRecord(std::vector<uint8_t>* payload,
                              iree_hal_profile_executable_trace_record_t record,
                              std::initializer_list<uint8_t> trace_payload) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record) + trace_payload.size());
  memcpy(payload->data() + offset, &record, sizeof(record));
  memcpy(payload->data() + offset + sizeof(record), trace_payload.begin(),
         trace_payload.size());
}

TEST(AttBundleTest, ParsesIndexesAndFindsRelationships) {
  iree_profile_att_profile_t profile;
  iree_profile_att_profile_initialize(iree_allocator_system(), &profile);

  std::vector<uint8_t> code_object_payload;
  iree_hal_profile_executable_code_object_record_t code_object =
      iree_hal_profile_executable_code_object_record_default();
  code_object.record_length = sizeof(code_object) + 3;
  code_object.executable_id = 10;
  code_object.code_object_id = 20;
  code_object.data_length = 3;
  AppendRecord(&code_object_payload, code_object, {0xAA, 0xBB, 0xCC});
  IREE_ASSERT_OK(ParseChunk(
      &profile, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS,
      code_object_payload));

  std::vector<uint8_t> load_payload;
  iree_hal_profile_executable_code_object_load_record_t load =
      iree_hal_profile_executable_code_object_load_record_default();
  load.physical_device_ordinal = 2;
  load.executable_id = 10;
  load.code_object_id = 20;
  load.load_delta = 4096;
  load.load_size = 65536;
  AppendRecord(&load_payload, load);
  IREE_ASSERT_OK(ParseChunk(
      &profile, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECT_LOADS,
      load_payload));

  std::vector<uint8_t> export_payload;
  iree_hal_profile_executable_export_record_t export_record =
      iree_hal_profile_executable_export_record_default();
  export_record.record_length = sizeof(export_record) + 11;
  export_record.executable_id = 10;
  export_record.export_ordinal = 3;
  export_record.name_length = 11;
  AppendRecord(&export_payload, export_record,
               {'d', 'i', 's', 'p', 'a', 't', 'c', 'h', '_', 'f', 'n'});
  IREE_ASSERT_OK(ParseChunk(&profile,
                            IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
                            export_payload));

  std::vector<uint8_t> dispatch_payload;
  iree_hal_profile_dispatch_event_t dispatch =
      iree_hal_profile_dispatch_event_default();
  dispatch.event_id = 7;
  dispatch.submission_id = 8;
  dispatch.command_buffer_id = 9;
  dispatch.command_index = 4;
  dispatch.executable_id = 10;
  dispatch.export_ordinal = 3;
  AppendRecord(&dispatch_payload, dispatch);
  iree_hal_profile_file_record_t dispatch_chunk = MakeChunk(
      IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS, dispatch_payload);
  dispatch_chunk.header.physical_device_ordinal = 2;
  dispatch_chunk.header.queue_ordinal = 5;
  IREE_ASSERT_OK(
      iree_profile_att_profile_parse_record(&profile, &dispatch_chunk));

  std::vector<uint8_t> trace_payload;
  iree_hal_profile_executable_trace_record_t trace =
      iree_hal_profile_executable_trace_record_default();
  trace.record_length = sizeof(trace);
  trace.format = IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT;
  trace.trace_id = 11;
  trace.dispatch_event_id = 7;
  trace.submission_id = 8;
  trace.command_buffer_id = 9;
  trace.command_index = 4;
  trace.executable_id = 10;
  trace.export_ordinal = 3;
  trace.physical_device_ordinal = 2;
  trace.queue_ordinal = 5;
  trace.data_length = 4;
  AppendTraceRecord(&trace_payload, trace, {0x01, 0x02, 0x03, 0x04});
  IREE_ASSERT_OK(ParseChunk(&profile,
                            IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES,
                            trace_payload));

  ASSERT_EQ(1u, profile.code_object_count);
  ASSERT_EQ(1u, profile.code_object_load_count);
  ASSERT_EQ(1u, profile.export_count);
  ASSERT_EQ(1u, profile.dispatch_count);
  ASSERT_EQ(1u, profile.trace_count);

  const iree_profile_att_code_object_t* found_code_object =
      iree_profile_att_profile_find_code_object(&profile, 10, 20);
  ASSERT_NE(nullptr, found_code_object);
  ASSERT_EQ(3u, found_code_object->data.data_length);
  EXPECT_EQ(0xBB, found_code_object->data.data[1]);

  const iree_profile_att_export_t* found_export =
      iree_profile_att_profile_find_export(&profile, 10, 3);
  ASSERT_NE(nullptr, found_export);
  EXPECT_TRUE(
      iree_string_view_equal(found_export->name, IREE_SV("dispatch_fn")));

  const iree_profile_att_dispatch_t* found_dispatch =
      iree_profile_att_profile_find_dispatch(&profile, &profile.traces[0]);
  ASSERT_NE(nullptr, found_dispatch);
  EXPECT_EQ(2u, found_dispatch->physical_device_ordinal);
  EXPECT_EQ(5u, found_dispatch->queue_ordinal);

  iree_profile_att_profile_deinitialize(&profile);
}

TEST(AttBundleTest, RejectsMalformedCodeObjectLength) {
  iree_profile_att_profile_t profile;
  iree_profile_att_profile_initialize(iree_allocator_system(), &profile);

  std::vector<uint8_t> payload;
  iree_hal_profile_executable_code_object_record_t code_object =
      iree_hal_profile_executable_code_object_record_default();
  code_object.record_length = sizeof(code_object) + 2;
  code_object.data_length = 3;
  AppendRecord(&payload, code_object, {0xAA, 0xBB});

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      ParseChunk(&profile,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_CODE_OBJECTS,
                 payload));

  iree_profile_att_profile_deinitialize(&profile);
}

}  // namespace
