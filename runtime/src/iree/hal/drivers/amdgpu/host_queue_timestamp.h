// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_TIMESTAMP_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_TIMESTAMP_H_

#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Publishes a PM4 timestamp packet that writes a start tick.
void iree_hal_amdgpu_host_queue_commit_timestamp_start(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control, uint64_t* start_tick);

// Publishes a PM4 timestamp packet that writes an end tick.
void iree_hal_amdgpu_host_queue_commit_timestamp_end(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint64_t* end_tick);

// Publishes one PM4 timestamp packet that writes both start and end ticks.
void iree_hal_amdgpu_host_queue_commit_timestamp_range(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint64_t* start_tick,
    uint64_t* end_tick);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_TIMESTAMP_H_
