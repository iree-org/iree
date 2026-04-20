// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/profile_metrics.h"

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Built-in metric registry
//===----------------------------------------------------------------------===//

static const iree_hal_profile_metric_descriptor_t
    iree_hal_profile_builtin_metric_descriptors[] = {
#define IREE_HAL_PROFILE_BUILTIN_METRIC(enum_name, metric_id, name, unit, \
                                        value_kind, semantic, plot_hint,  \
                                        description)                      \
  {                                                                       \
      (metric_id), (unit),         (value_kind),          (semantic),     \
      (plot_hint), IREE_SVL(name), IREE_SVL(description),                 \
  },
#include "iree/hal/profile_metrics.inc"
#undef IREE_HAL_PROFILE_BUILTIN_METRIC
};

IREE_API_EXPORT iree_host_size_t
iree_hal_profile_builtin_metric_descriptor_count(void) {
  return IREE_ARRAYSIZE(iree_hal_profile_builtin_metric_descriptors);
}

IREE_API_EXPORT const iree_hal_profile_metric_descriptor_t*
iree_hal_profile_builtin_metric_descriptor_at(iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(iree_hal_profile_builtin_metric_descriptors)) {
    return NULL;
  }
  return &iree_hal_profile_builtin_metric_descriptors[index];
}

IREE_API_EXPORT const iree_hal_profile_metric_descriptor_t*
iree_hal_profile_builtin_metric_descriptor_lookup(uint64_t metric_id) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_profile_builtin_metric_descriptors); ++i) {
    const iree_hal_profile_metric_descriptor_t* descriptor =
        &iree_hal_profile_builtin_metric_descriptors[i];
    if (descriptor->metric_id == metric_id) return descriptor;
  }
  return NULL;
}

IREE_API_EXPORT const iree_hal_profile_metric_descriptor_t*
iree_hal_profile_builtin_metric_descriptor_lookup_name(
    iree_string_view_t name) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_profile_builtin_metric_descriptors); ++i) {
    const iree_hal_profile_metric_descriptor_t* descriptor =
        &iree_hal_profile_builtin_metric_descriptors[i];
    if (iree_string_view_equal(descriptor->name, name)) return descriptor;
  }
  return NULL;
}

IREE_API_EXPORT iree_string_view_t
iree_hal_profile_metric_unit_string(iree_hal_profile_metric_unit_t unit) {
  switch (unit) {
    case IREE_HAL_PROFILE_METRIC_UNIT_NONE:
      return IREE_SV("none");
    case IREE_HAL_PROFILE_METRIC_UNIT_COUNT:
      return IREE_SV("count");
    case IREE_HAL_PROFILE_METRIC_UNIT_HERTZ:
      return IREE_SV("hz");
    case IREE_HAL_PROFILE_METRIC_UNIT_BYTES:
      return IREE_SV("bytes");
    case IREE_HAL_PROFILE_METRIC_UNIT_BYTES_PER_SECOND:
      return IREE_SV("bytes_per_second");
    case IREE_HAL_PROFILE_METRIC_UNIT_MILLIPERCENT:
      return IREE_SV("millipercent");
    case IREE_HAL_PROFILE_METRIC_UNIT_MILLIDEGREES_CELSIUS:
      return IREE_SV("millidegrees_celsius");
    case IREE_HAL_PROFILE_METRIC_UNIT_MICROWATTS:
      return IREE_SV("microwatts");
    case IREE_HAL_PROFILE_METRIC_UNIT_MICROJOULES:
      return IREE_SV("microjoules");
    case IREE_HAL_PROFILE_METRIC_UNIT_BITFIELD:
      return IREE_SV("bitfield");
    case IREE_HAL_PROFILE_METRIC_UNIT_ENUM:
      return IREE_SV("enum");
    default:
      return iree_string_view_empty();
  }
}

IREE_API_EXPORT iree_string_view_t iree_hal_profile_metric_value_kind_string(
    iree_hal_profile_metric_value_kind_t value_kind) {
  switch (value_kind) {
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_NONE:
      return IREE_SV("none");
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_I64:
      return IREE_SV("i64");
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_U64:
      return IREE_SV("u64");
    case IREE_HAL_PROFILE_METRIC_VALUE_KIND_F64:
      return IREE_SV("f64");
    default:
      return iree_string_view_empty();
  }
}

IREE_API_EXPORT iree_string_view_t iree_hal_profile_metric_semantic_string(
    iree_hal_profile_metric_semantic_t semantic) {
  switch (semantic) {
    case IREE_HAL_PROFILE_METRIC_SEMANTIC_NONE:
      return IREE_SV("none");
    case IREE_HAL_PROFILE_METRIC_SEMANTIC_INSTANT:
      return IREE_SV("instant");
    case IREE_HAL_PROFILE_METRIC_SEMANTIC_AVERAGE:
      return IREE_SV("average");
    case IREE_HAL_PROFILE_METRIC_SEMANTIC_CUMULATIVE:
      return IREE_SV("cumulative");
    case IREE_HAL_PROFILE_METRIC_SEMANTIC_DELTA:
      return IREE_SV("delta");
    case IREE_HAL_PROFILE_METRIC_SEMANTIC_STATE:
      return IREE_SV("state");
    default:
      return iree_string_view_empty();
  }
}

IREE_API_EXPORT iree_string_view_t iree_hal_profile_metric_plot_hint_string(
    iree_hal_profile_metric_plot_hint_t plot_hint) {
  switch (plot_hint) {
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_NONE:
      return IREE_SV("none");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_NUMBER:
      return IREE_SV("number");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_MEMORY:
      return IREE_SV("memory");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_PERCENTAGE:
      return IREE_SV("percentage");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_FREQUENCY:
      return IREE_SV("frequency");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_TEMPERATURE:
      return IREE_SV("temperature");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_POWER:
      return IREE_SV("power");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_ENERGY:
      return IREE_SV("energy");
    case IREE_HAL_PROFILE_METRIC_PLOT_HINT_BANDWIDTH:
      return IREE_SV("bandwidth");
    default:
      return iree_string_view_empty();
  }
}
