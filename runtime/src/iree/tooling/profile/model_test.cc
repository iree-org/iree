// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/model.h"

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

static iree_hal_profile_file_record_t MakeCommandBuffersChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeExecutablesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeExecutableExportsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeCommandOperationsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS;
  return chunk;
}

static iree_hal_profile_file_record_t MakeMetricSourcesChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES;
  return chunk;
}

static iree_hal_profile_file_record_t MakeMetricDescriptorsChunk(
    const std::vector<uint8_t>& payload) {
  iree_hal_profile_file_record_t chunk = MakeChunk(payload);
  chunk.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS;
  return chunk;
}

template <typename T>
static void AppendPlainRecord(std::vector<uint8_t>* payload, const T& record) {
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + sizeof(record));
  memcpy(payload->data() + offset, &record, sizeof(record));
}

template <typename T>
static void AppendRecordWithPayload(std::vector<uint8_t>* payload,
                                    const T& record,
                                    iree_string_view_t trailing_payload) {
  AppendPlainRecord(payload, record);
  const iree_host_size_t offset = payload->size();
  payload->resize(offset + trailing_payload.size);
  memcpy(payload->data() + offset, trailing_payload.data,
         trailing_payload.size);
}

static void AppendCommandBuffer(std::vector<uint8_t>* payload,
                                uint64_t command_buffer_id) {
  iree_hal_profile_command_buffer_record_t record =
      iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id = command_buffer_id;
  record.physical_device_ordinal = 0;
  AppendPlainRecord(payload, record);
}

static void AppendCommandOperation(
    std::vector<uint8_t>* payload,
    const iree_hal_profile_command_operation_record_t& record) {
  AppendPlainRecord(payload, record);
}

static void AppendExecutable(std::vector<uint8_t>* payload,
                             uint64_t executable_id, uint32_t export_count) {
  iree_hal_profile_executable_record_t record =
      iree_hal_profile_executable_record_default();
  record.executable_id = executable_id;
  record.export_count = export_count;
  AppendPlainRecord(payload, record);
}

static void AppendExecutableExport(std::vector<uint8_t>* payload,
                                   uint64_t executable_id,
                                   uint32_t export_ordinal,
                                   iree_string_view_t name) {
  iree_hal_profile_executable_export_record_t record =
      iree_hal_profile_executable_export_record_default();
  record.executable_id = executable_id;
  record.export_ordinal = export_ordinal;
  record.name_length = name.size;
  record.record_length = sizeof(record) + name.size;
  AppendRecordWithPayload(payload, record, name);
}

static void AppendMetricSource(std::vector<uint8_t>* payload,
                               uint64_t source_id, iree_string_view_t name) {
  iree_hal_profile_device_metric_source_record_t record =
      iree_hal_profile_device_metric_source_record_default();
  record.source_id = source_id;
  record.physical_device_ordinal = 0;
  record.device_class = IREE_HAL_PROFILE_DEVICE_CLASS_GPU;
  record.metric_count = 1;
  record.name_length = name.size;
  record.record_length = sizeof(record) + record.name_length;
  AppendRecordWithPayload(payload, record, name);
}

static void AppendMetricDescriptor(std::vector<uint8_t>* payload,
                                   uint64_t source_id, uint64_t metric_id,
                                   iree_string_view_t name,
                                   iree_string_view_t description) {
  iree_hal_profile_device_metric_descriptor_record_t record =
      iree_hal_profile_device_metric_descriptor_record_default();
  record.source_id = source_id;
  record.metric_id = metric_id;
  record.unit = IREE_HAL_PROFILE_METRIC_UNIT_HERTZ;
  record.value_kind = IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64;
  record.semantic = IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT;
  record.plot_hint = IREE_HAL_PROFILE_METRIC_PLOT_HINT_FREQUENCY;
  record.name_length = name.size;
  record.description_length = description.size;
  record.record_length = sizeof(record) + name.size + description.size;
  AppendPlainRecord(payload, record);
  const iree_host_size_t name_offset = payload->size();
  payload->resize(name_offset + name.size + description.size);
  memcpy(payload->data() + name_offset, name.data, name.size);
  memcpy(payload->data() + name_offset + name.size, description.data,
         description.size);
}

static iree_hal_profile_clock_correlation_record_t MakeClockSample(
    uint64_t sample_id, uint64_t device_tick, uint64_t host_cpu_timestamp_ns) {
  iree_hal_profile_clock_correlation_record_t sample =
      iree_hal_profile_clock_correlation_record_default();
  sample.flags = IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
                 IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP;
  sample.sample_id = sample_id;
  sample.device_tick = device_tick;
  sample.host_cpu_timestamp_ns = host_cpu_timestamp_ns;
  return sample;
}

TEST(ProfileClockFitTest, MapsTicksWithIntegerRounding) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample =
      MakeClockSample(10, (1ull << 60) + 3, 900000000000000000ull);
  device.last_clock_sample =
      MakeClockSample(11, (1ull << 60) + 10, 900000000000000010ull);

  iree_profile_model_clock_fit_t fit;
  ASSERT_TRUE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      &fit));
  EXPECT_EQ(10u, fit.first_sample_id);
  EXPECT_EQ(11u, fit.last_sample_id);
  EXPECT_EQ(7u, fit.device_tick_span);
  EXPECT_EQ(10u, fit.time_span_ns);

  int64_t time_ns = 0;
  EXPECT_TRUE(
      iree_profile_model_clock_fit_map_tick(&fit, (1ull << 60) + 6, &time_ns));
  EXPECT_EQ(900000000000000004ll, time_ns);
  EXPECT_TRUE(
      iree_profile_model_clock_fit_map_tick(&fit, (1ull << 60), &time_ns));
  EXPECT_EQ(899999999999999996ll, time_ns);

  int64_t duration_ns = 0;
  EXPECT_TRUE(
      iree_profile_model_clock_fit_scale_ticks_to_ns(&fit, 7, &duration_ns));
  EXPECT_EQ(10, duration_ns);
  EXPECT_TRUE(
      iree_profile_model_clock_fit_scale_ticks_to_ns(&fit, 4, &duration_ns));
  EXPECT_EQ(6, duration_ns);
}

TEST(ProfileClockFitTest, FitsIreeHostTimeFromBracketMidpoints) {
  iree_profile_model_device_t device;
  memset(&device, 0, sizeof(device));
  device.clock_sample_count = 2;
  device.first_clock_sample = MakeClockSample(1, 1000, 5000);
  device.first_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  device.first_clock_sample.host_time_begin_ns = 100;
  device.first_clock_sample.host_time_end_ns = 103;
  device.last_clock_sample = MakeClockSample(2, 1010, 6000);
  device.last_clock_sample.flags |=
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  device.last_clock_sample.host_time_begin_ns = 170;
  device.last_clock_sample.host_time_end_ns = 173;

  iree_profile_model_clock_fit_t fit;
  ASSERT_TRUE(iree_profile_model_device_try_fit_clock_exact(
      &device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_IREE_HOST_TIME_NS, &fit));
  EXPECT_EQ(101, fit.first_time_ns);
  EXPECT_EQ(171, fit.last_time_ns);

  int64_t time_ns = 0;
  EXPECT_TRUE(iree_profile_model_clock_fit_map_tick(&fit, 1005, &time_ns));
  EXPECT_EQ(136, time_ns);
}

TEST(ProfileModelTest, AcceptsLinearCommandOperationsWithoutBlockStructure) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_operation_chunk));
  ASSERT_EQ(1u, model.command_operation_count);
  EXPECT_FALSE(iree_hal_profile_command_operation_has_block_structure(
      &model.command_operations[0].record));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, LinksExecutableExportsByOwner) {
  std::vector<uint8_t> executable_payload;
  AppendExecutable(&executable_payload, 5, 2);
  AppendExecutable(&executable_payload, 7, 1);
  iree_hal_profile_file_record_t executable_chunk =
      MakeExecutablesChunk(executable_payload);

  std::vector<uint8_t> export_payload;
  AppendExecutableExport(&export_payload, 5, 0, IREE_SV("main"));
  AppendExecutableExport(&export_payload, 7, 0, IREE_SV("side"));
  AppendExecutableExport(&export_payload, 5, 1, IREE_SV("tail"));
  iree_hal_profile_file_record_t export_chunk =
      MakeExecutableExportsChunk(export_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &executable_chunk));
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &export_chunk));

  const iree_profile_model_executable_t* executable =
      iree_profile_model_find_executable(&model, 5);
  ASSERT_NE(executable, nullptr);
  ASSERT_EQ(2u, executable->export_row_count);
  ASSERT_NE(IREE_HOST_SIZE_MAX, executable->first_export_index);
  const iree_profile_model_export_t* first_export =
      &model.exports[executable->first_export_index];
  EXPECT_EQ(5u, first_export->executable_id);
  EXPECT_EQ(0u, first_export->export_ordinal);
  ASSERT_NE(IREE_HOST_SIZE_MAX, first_export->next_export_index);
  const iree_profile_model_export_t* second_export =
      &model.exports[first_export->next_export_index];
  EXPECT_EQ(5u, second_export->executable_id);
  EXPECT_EQ(1u, second_export->export_ordinal);
  EXPECT_EQ(IREE_HOST_SIZE_MAX, second_export->next_export_index);

  const iree_profile_model_executable_t* other_executable =
      iree_profile_model_find_executable(&model, 7);
  ASSERT_NE(other_executable, nullptr);
  EXPECT_EQ(1u, other_executable->export_row_count);
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsExecutableExportWithoutOwner) {
  std::vector<uint8_t> export_payload;
  AppendExecutableExport(&export_payload, 5, 0, IREE_SV("main"));
  iree_hal_profile_file_record_t export_chunk =
      MakeExecutableExportsChunk(export_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_model_process_metadata_record(&model, &export_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsCommandOperationBlockFieldsWithoutFlag) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;
  operation.block_ordinal = 2;
  operation.block_command_ordinal = 3;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_process_metadata_record(
                            &model, &command_operation_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsBlockStructureWithoutBlockCoordinates) {
  std::vector<uint8_t> command_buffer_payload;
  AppendCommandBuffer(&command_buffer_payload, 1);
  iree_hal_profile_file_record_t command_buffer_chunk =
      MakeCommandBuffersChunk(command_buffer_payload);

  iree_hal_profile_command_operation_record_t operation =
      iree_hal_profile_command_operation_record_default();
  operation.type = IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH;
  operation.command_buffer_id = 1;
  operation.command_index = 0;
  operation.flags = IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_BLOCK_STRUCTURE;

  std::vector<uint8_t> command_operation_payload;
  AppendCommandOperation(&command_operation_payload, operation);
  iree_hal_profile_file_record_t command_operation_chunk =
      MakeCommandOperationsChunk(command_operation_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(iree_profile_model_process_metadata_record(
      &model, &command_buffer_chunk));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_profile_model_process_metadata_record(
                            &model, &command_operation_chunk));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, IndexesDeviceMetricDescriptors) {
  std::vector<uint8_t> source_payload;
  AppendMetricSource(&source_payload, 1, IREE_SV("test.metrics"));
  iree_hal_profile_file_record_t source_chunk =
      MakeMetricSourcesChunk(source_payload);

  std::vector<uint8_t> descriptor_payload;
  AppendMetricDescriptor(&descriptor_payload, 1, 2, IREE_SV("test.clock"),
                         IREE_SV("Test clock."));
  iree_hal_profile_file_record_t descriptor_chunk =
      MakeMetricDescriptorsChunk(descriptor_payload);

  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &source_chunk));
  IREE_ASSERT_OK(
      iree_profile_model_process_metadata_record(&model, &descriptor_chunk));

  const iree_profile_model_metric_source_t* source =
      iree_profile_model_find_metric_source(&model, 1);
  ASSERT_NE(source, nullptr);
  EXPECT_TRUE(iree_string_view_equal(source->name, IREE_SV("test.metrics")));

  const iree_profile_model_metric_descriptor_t* descriptor = NULL;
  IREE_ASSERT_OK(
      iree_profile_model_resolve_metric_descriptor(&model, 1, 2, &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor->name, IREE_SV("test.clock")));
  EXPECT_TRUE(
      iree_string_view_equal(descriptor->description, IREE_SV("Test clock.")));
  iree_profile_model_deinitialize(&model);
}

TEST(ProfileModelTest, RejectsMissingDeviceMetricDescriptorReference) {
  iree_profile_model_t model;
  iree_profile_model_initialize(iree_allocator_system(), &model);

  const iree_profile_model_metric_descriptor_t* descriptor = NULL;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_profile_model_resolve_metric_descriptor(&model, 1, 2, &descriptor));
  EXPECT_EQ(nullptr, descriptor);
  iree_profile_model_deinitialize(&model);
}

}  // namespace
