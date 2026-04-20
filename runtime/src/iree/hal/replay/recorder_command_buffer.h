// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_COMMAND_BUFFER_H_
#define IREE_HAL_REPLAY_RECORDER_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void iree_hal_replay_recorder_command_buffer_make_object_payload(
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_replay_command_buffer_object_payload_t* out_payload);

iree_status_t iree_hal_replay_recorder_command_buffer_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t command_buffer_id,
    iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_t* base_command_buffer,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

iree_hal_command_buffer_t* iree_hal_replay_recorder_command_buffer_base_or_self(
    iree_hal_command_buffer_t* command_buffer);

iree_hal_replay_object_id_t iree_hal_replay_recorder_command_buffer_id_or_none(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_COMMAND_BUFFER_H_
