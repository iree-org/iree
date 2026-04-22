// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_EVENT_H_
#define IREE_HAL_REPLAY_RECORDER_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void iree_hal_replay_recorder_event_make_object_payload(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_hal_replay_event_object_payload_t* out_payload);

iree_status_t iree_hal_replay_recorder_event_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t event_id, iree_hal_event_t* base_event,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event);

iree_hal_event_t* iree_hal_replay_recorder_event_base_or_self(
    iree_hal_event_t* event);

iree_hal_replay_object_id_t iree_hal_replay_recorder_event_id_or_none(
    iree_hal_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_EVENT_H_
