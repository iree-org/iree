// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_RECORD_H_
#define IREE_HAL_REPLAY_RECORDER_RECORD_H_

#include "iree/base/api.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Pending operation metadata owned by a begin/end recording pair.
typedef struct iree_hal_replay_pending_record_t {
  // Recorder whose mutex is held until the pending record is completed.
  iree_hal_replay_recorder_t* recorder;
  // Metadata to write once the intercepted operation has completed.
  iree_hal_replay_file_record_metadata_t metadata;
} iree_hal_replay_pending_record_t;

void iree_hal_replay_recorder_fail(iree_hal_replay_recorder_t* recorder,
                                   iree_status_t status);

iree_status_t iree_hal_replay_recorder_reserve_object_id(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_object_id_t* out_object_id);

iree_status_t iree_hal_replay_recorder_record_object(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs,
    iree_hal_replay_object_id_t* out_object_id);

iree_status_t iree_hal_replay_recorder_begin_operation(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record);

iree_status_t iree_hal_replay_recorder_end_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status);

iree_status_t iree_hal_replay_recorder_end_operation_with_payload(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs);

iree_status_t iree_hal_replay_recorder_end_creation_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t operation_iovec_count,
    const iree_const_byte_span_t* operation_iovecs,
    iree_hal_replay_object_type_t created_object_type,
    iree_hal_replay_object_id_t created_object_id,
    iree_hal_replay_payload_type_t object_payload_type,
    iree_host_size_t object_iovec_count,
    const iree_const_byte_span_t* object_iovecs);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_RECORD_H_
