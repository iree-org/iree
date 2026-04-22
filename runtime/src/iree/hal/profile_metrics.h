// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_PROFILE_METRICS_H_
#define IREE_HAL_PROFILE_METRICS_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/profile_schema.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Built-in metric ids occupy the low half of the metric id space. Producers may
// use ids at or above this value for source-specific metrics whose meaning is
// carried entirely by descriptor records in the profile bundle.
#define IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE (UINT64_C(1) << 63)

// Stable identifiers for HAL built-in device metrics.
typedef uint64_t iree_hal_profile_builtin_metric_id_t;
enum iree_hal_profile_builtin_metric_id_e {
  IREE_HAL_PROFILE_BUILTIN_METRIC_ID_NONE = 0u,

#define IREE_HAL_PROFILE_BUILTIN_METRIC(enum_name, metric_id, name, unit, \
                                        value_kind, semantic, plot_hint,  \
                                        description)                      \
  IREE_HAL_PROFILE_BUILTIN_METRIC_ID_##enum_name = (metric_id),
#include "iree/hal/profile_metrics.inc"
#undef IREE_HAL_PROFILE_BUILTIN_METRIC
};

// Process-local descriptor for one built-in device metric.
typedef struct iree_hal_profile_metric_descriptor_t {
  // Stable or source-specific metric identifier.
  uint64_t metric_id;
  // Display and scaling unit for this metric.
  iree_hal_profile_metric_unit_t unit;
  // Storage interpretation for sampled value_bits.
  iree_hal_profile_metric_value_kind_t value_kind;
  // Sampling semantic for this metric.
  iree_hal_profile_metric_semantic_t semantic;
  // Preferred visualization shape for this metric.
  iree_hal_profile_metric_plot_hint_t plot_hint;
  // Stable metric name.
  iree_string_view_t name;
  // Human-readable metric description.
  iree_string_view_t description;
} iree_hal_profile_metric_descriptor_t;

// Returns true when |metric_id| is in the built-in metric id range.
static inline bool iree_hal_profile_metric_id_is_builtin(uint64_t metric_id) {
  return metric_id != 0 && metric_id < IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE;
}

// Returns true when |metric_id| is in the producer-specific metric id range.
static inline bool iree_hal_profile_metric_id_is_producer_specific(
    uint64_t metric_id) {
  return metric_id >= IREE_HAL_PROFILE_METRIC_ID_PRODUCER_BASE;
}

// Returns the number of built-in metric descriptors.
IREE_API_EXPORT iree_host_size_t
iree_hal_profile_builtin_metric_descriptor_count(void);

// Returns the built-in metric descriptor at |index|, or NULL when out of range.
IREE_API_EXPORT const iree_hal_profile_metric_descriptor_t*
iree_hal_profile_builtin_metric_descriptor_at(iree_host_size_t index);

// Returns the built-in metric descriptor for |metric_id|, or NULL if unknown.
IREE_API_EXPORT const iree_hal_profile_metric_descriptor_t*
iree_hal_profile_builtin_metric_descriptor_lookup(uint64_t metric_id);

// Returns the built-in metric descriptor named |name|, or NULL if unknown.
IREE_API_EXPORT const iree_hal_profile_metric_descriptor_t*
iree_hal_profile_builtin_metric_descriptor_lookup_name(iree_string_view_t name);

// Returns the stable display name for |unit|.
IREE_API_EXPORT iree_string_view_t
iree_hal_profile_metric_unit_string(iree_hal_profile_metric_unit_t unit);

// Returns the stable display name for |value_kind|.
IREE_API_EXPORT iree_string_view_t iree_hal_profile_metric_value_kind_string(
    iree_hal_profile_metric_value_kind_t value_kind);

// Returns the stable display name for |semantic|.
IREE_API_EXPORT iree_string_view_t iree_hal_profile_metric_semantic_string(
    iree_hal_profile_metric_semantic_t semantic);

// Returns the stable display name for |plot_hint|.
IREE_API_EXPORT iree_string_view_t iree_hal_profile_metric_plot_hint_string(
    iree_hal_profile_metric_plot_hint_t plot_hint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_PROFILE_METRICS_H_
