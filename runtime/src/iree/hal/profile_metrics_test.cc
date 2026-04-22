// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"

namespace iree {
namespace hal {
namespace {

TEST(ProfileMetricsTest, BuiltinMetricDescriptorsAreUnique) {
  const iree_host_size_t descriptor_count =
      iree_hal_profile_builtin_metric_descriptor_count();
  ASSERT_GT(descriptor_count, 0u);

  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const iree_hal_profile_metric_descriptor_t* descriptor =
        iree_hal_profile_builtin_metric_descriptor_at(i);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_TRUE(iree_hal_profile_metric_id_is_builtin(descriptor->metric_id));
    EXPECT_FALSE(iree_string_view_is_empty(descriptor->name));
    EXPECT_FALSE(iree_string_view_is_empty(descriptor->description));
    EXPECT_FALSE(iree_string_view_is_empty(
        iree_hal_profile_metric_unit_string(descriptor->unit)));
    EXPECT_FALSE(iree_string_view_is_empty(
        iree_hal_profile_metric_value_kind_string(descriptor->value_kind)));
    EXPECT_FALSE(iree_string_view_is_empty(
        iree_hal_profile_metric_semantic_string(descriptor->semantic)));
    EXPECT_FALSE(iree_string_view_is_empty(
        iree_hal_profile_metric_plot_hint_string(descriptor->plot_hint)));

    for (iree_host_size_t j = i + 1; j < descriptor_count; ++j) {
      const iree_hal_profile_metric_descriptor_t* other_descriptor =
          iree_hal_profile_builtin_metric_descriptor_at(j);
      ASSERT_NE(other_descriptor, nullptr);
      EXPECT_NE(descriptor->metric_id, other_descriptor->metric_id);
      EXPECT_FALSE(
          iree_string_view_equal(descriptor->name, other_descriptor->name));
    }
  }
}

TEST(ProfileMetricsTest, LookupRoundTripsThroughIdAndName) {
  const iree_hal_profile_metric_descriptor_t* descriptor =
      iree_hal_profile_builtin_metric_descriptor_lookup_name(
          IREE_SV("clock.compute.current"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_COMPUTE_CURRENT,
            descriptor->metric_id);
  EXPECT_EQ(IREE_HAL_PROFILE_METRIC_UNIT_HERTZ, descriptor->unit);
  EXPECT_EQ(IREE_HAL_PROFILE_METRIC_PLOT_HINT_FREQUENCY, descriptor->plot_hint);

  EXPECT_EQ(descriptor, iree_hal_profile_builtin_metric_descriptor_lookup(
                            descriptor->metric_id));
  EXPECT_EQ(nullptr, iree_hal_profile_builtin_metric_descriptor_lookup(0));
  EXPECT_EQ(nullptr, iree_hal_profile_builtin_metric_descriptor_lookup_name(
                         IREE_SV("does.not.exist")));
}

TEST(ProfileMetricsTest, ProducerMetricIdsAreSeparatedFromBuiltins) {
  EXPECT_TRUE(iree_hal_profile_metric_id_is_producer_specific(
      IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE));
  EXPECT_FALSE(iree_hal_profile_metric_id_is_builtin(
      IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE));
  EXPECT_FALSE(iree_hal_profile_metric_id_is_builtin(0));
}

TEST(ProfileMetricsTest, MetricRecordDefaultsUseAbsentSentinels) {
  iree_hal_profile_device_metric_source_record_t source_record =
      iree_hal_profile_device_metric_source_record_default();
  EXPECT_EQ(sizeof(source_record), source_record.record_length);
  EXPECT_EQ(UINT32_MAX, source_record.physical_device_ordinal);

  iree_hal_profile_device_metric_descriptor_record_t descriptor_record =
      iree_hal_profile_device_metric_descriptor_record_default();
  EXPECT_EQ(sizeof(descriptor_record), descriptor_record.record_length);
  EXPECT_EQ(0u, descriptor_record.source_id);
  EXPECT_EQ(0u, descriptor_record.metric_id);

  iree_hal_profile_device_metric_sample_record_t sample_record =
      iree_hal_profile_device_metric_sample_record_default();
  EXPECT_EQ(sizeof(sample_record), sample_record.record_length);
  EXPECT_EQ(UINT32_MAX, sample_record.physical_device_ordinal);
}

}  // namespace
}  // namespace hal
}  // namespace iree
