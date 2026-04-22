// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PROFILE_DEVICE_METRICS_H_
#define IREE_HAL_DRIVERS_AMDGPU_PROFILE_DEVICE_METRICS_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;
typedef struct iree_hal_amdgpu_profile_device_metrics_session_t
    iree_hal_amdgpu_profile_device_metrics_session_t;

// Allocates a flush-sampled AMDGPU device metrics session from |options|.
//
// The session owns only cold-path sampler state: PCI sysfs paths, open metric
// file descriptors, and per-source ids. Queue submission/completion paths never
// reference the session.
iree_status_t iree_hal_amdgpu_profile_device_metrics_session_allocate(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_device_metrics_session_t** out_session);

// Frees |session| and closes any sysfs file descriptors it owns.
void iree_hal_amdgpu_profile_device_metrics_session_free(
    iree_hal_amdgpu_profile_device_metrics_session_t* session);

// Writes source and descriptor metadata chunks for |session|.
iree_status_t iree_hal_amdgpu_profile_device_metrics_session_write_metadata(
    const iree_hal_amdgpu_profile_device_metrics_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name);

// Samples all active device metric sources and writes sample chunks.
iree_status_t iree_hal_amdgpu_profile_device_metrics_session_sample_and_write(
    iree_hal_amdgpu_profile_device_metrics_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PROFILE_DEVICE_METRICS_H_
