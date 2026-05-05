// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;

#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_NAME_LENGTH 64
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_SAMPLE_VALUES 16

// Per-physical-device source sampled by a device-metrics session.
typedef struct iree_hal_amdgpu_profile_device_metric_source_t {
  // Profile metadata describing this source.
  struct {
    // Producer-defined source id unique within the profiling session.
    uint64_t id;

    // Session-local physical device ordinal sampled by this source.
    uint32_t physical_device_ordinal;

    // Producer-defined source kind written to source metadata.
    uint32_t kind;

    // Source revision derived from the backing implementation.
    uint32_t revision;

    // Human-readable source name stored in source metadata.
    char name[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_NAME_LENGTH];
  } metadata;

  // Built-in metric ids this source can emit.
  struct {
    // Number of entries in |ids|.
    iree_host_size_t count;

    // Pointer to static built-in metric ids emitted by the source.
    const uint64_t* ids;
  } metrics;

  // Mutable sampling state.
  struct {
    // Next nonzero sample id emitted for this source.
    uint64_t next_sample_id;
  } sampling;

  // Source implementation state.
  struct {
    // Host allocator used for |state| when owned by the source.
    iree_allocator_t host_allocator;

    // Opaque implementation-owned state.
    void* state;
  } platform;
} iree_hal_amdgpu_profile_device_metric_source_t;

// Builder for one packed device metric sample.
typedef struct iree_hal_amdgpu_profile_device_metric_sample_builder_t {
  // Sample record header being populated.
  iree_hal_profile_device_metric_sample_record_t record;

  // Fixed value storage written immediately after |record|.
  iree_hal_profile_device_metric_value_t
      values[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_SAMPLE_VALUES];
} iree_hal_amdgpu_profile_device_metric_sample_builder_t;

// Appends |value| to |builder| unless the metric is already present.
void iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint64_t value);

// Appends signed |value| to |builder| unless the metric is already present.
void iree_hal_amdgpu_profile_device_metric_sample_builder_append_i64(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, int64_t value);

// Initializes a per-device metric source for |physical_device|.
iree_status_t iree_hal_amdgpu_profile_device_metric_source_initialize(
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_device_metric_source_t* out_source);

// Deinitializes |source| and releases platform handles it owns.
void iree_hal_amdgpu_profile_device_metric_source_deinitialize(
    iree_hal_amdgpu_profile_device_metric_source_t* source);

// Samples |source| into |builder|.
iree_status_t iree_hal_amdgpu_profile_device_metric_source_sample(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_H_
