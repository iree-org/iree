// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_BUFFER_H_
#define IREE_HAL_REPLAY_RECORDER_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void iree_hal_replay_recorder_buffer_make_object_payload(
    iree_hal_buffer_t* base_buffer,
    iree_hal_replay_buffer_object_payload_t* out_payload);

iree_status_t iree_hal_replay_recorder_buffer_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t buffer_id, iree_hal_device_t* placement_device,
    iree_hal_buffer_t* base_buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer);

iree_hal_buffer_t* iree_hal_replay_recorder_buffer_base_or_self(
    iree_hal_buffer_t* buffer);

iree_status_t iree_hal_replay_recorder_buffer_unwrap_for_call(
    iree_hal_buffer_t* buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_base_buffer,
    iree_hal_buffer_t** out_temporary_buffer);

void iree_hal_replay_recorder_buffer_release_temporary(
    iree_hal_buffer_t* temporary_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_BUFFER_H_
