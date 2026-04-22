// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_ALLOCATOR_H_
#define IREE_HAL_REPLAY_RECORDER_ALLOCATOR_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void iree_hal_replay_recorder_allocator_make_allocate_buffer_payload(
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_replay_allocator_allocate_buffer_payload_t* out_payload);

iree_status_t iree_hal_replay_recorder_wrap_allocator(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_device_t* placement_device, iree_hal_allocator_t* base_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_ALLOCATOR_H_
