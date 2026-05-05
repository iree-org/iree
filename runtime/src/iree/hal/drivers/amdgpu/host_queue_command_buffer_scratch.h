// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_SCRATCH_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_SCRATCH_H_

#include "iree/base/api.h"

// Queue_execute binding table entries cached as raw device pointers under
// submission_mutex while replaying an AQL command buffer. Larger binding tables
// use temporary arena storage for the current submission.
#define IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BINDING_SCRATCH_CAPACITY 4096u

// Queue_execute packet metadata cached under submission_mutex while replaying
// an AQL command buffer. Larger packet-bearing blocks use a temporary arena
// block for the current submission.
#define IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_SCRATCH_CAPACITY 512u

// Lazily allocated host queue scratch used only by queue_execute.
typedef struct iree_hal_amdgpu_host_queue_command_buffer_scratch_t {
  // Resolved queue_execute binding-table device pointers.
  struct {
    // Raw device pointers indexed by queue_execute binding table slot.
    uint64_t ptrs
        [IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BINDING_SCRATCH_CAPACITY];
  } bindings;
  // Packet sidebands populated by block processors before AQL publication.
  struct {
    // AQL packet header words indexed by emitted packet ordinal.
    uint16_t headers
        [IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_SCRATCH_CAPACITY];
    // AQL packet setup words indexed by emitted packet ordinal.
    uint16_t setups
        [IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_SCRATCH_CAPACITY];
  } packets;
} iree_hal_amdgpu_host_queue_command_buffer_scratch_t;

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_SCRATCH_H_
