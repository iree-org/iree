// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_EXECUTABLE_H_
#define IREE_HAL_REPLAY_RECORDER_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_replay_recorder_executable_cache_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t executable_cache_id,
    iree_hal_executable_cache_t* base_executable_cache,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

iree_hal_executable_t* iree_hal_replay_recorder_executable_base_or_self(
    iree_hal_executable_t* executable);

iree_hal_replay_object_id_t iree_hal_replay_recorder_executable_id_or_none(
    iree_hal_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_EXECUTABLE_H_
