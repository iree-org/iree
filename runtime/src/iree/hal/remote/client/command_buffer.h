// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client-side command buffer that serializes HAL recording calls into the wire
// format defined in protocol/commands.h. The recorded bytes ARE the network
// payload -- no translation between recording and sending.
//
// Two execution modes:
//   - One-shot (ONE_SHOT flag set): stream data is sent inline in the
//     COMMAND_BUFFER_EXECUTE queue op at queue_execute time.
//   - Reusable (ONE_SHOT flag clear): end() uploads the stream via
//     COMMAND_BUFFER_UPLOAD control RPC (async, provisional ID). Subsequent
//     queue_execute calls reference the server-cached resource by ID.

#ifndef IREE_HAL_REMOTE_CLIENT_COMMAND_BUFFER_H_
#define IREE_HAL_REMOTE_CLIENT_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_client_device_t iree_hal_remote_client_device_t;

// Creates a remote client command buffer that records into wire format.
iree_status_t iree_hal_remote_client_command_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a remote client command buffer.
bool iree_hal_remote_client_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns the serialized command stream data. The span is valid until the
// command buffer is destroyed or begin() is called (re-recording).
iree_const_byte_span_t iree_hal_remote_client_command_buffer_stream(
    iree_hal_command_buffer_t* command_buffer);

// Returns the server resource ID for a reusable command buffer. Returns 0 for
// one-shot command buffers or if upload hasn't completed yet.
iree_hal_remote_resource_id_t iree_hal_remote_client_command_buffer_resource_id(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_COMMAND_BUFFER_H_
