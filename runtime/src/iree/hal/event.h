// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_EVENT_H_
#define IREE_HAL_EVENT_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitmask of flags controlling the behavior of an event.
enum iree_hal_event_flag_bits_t {
  IREE_HAL_EVENT_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_event_flags_t;

//===----------------------------------------------------------------------===//
// iree_hal_event_t
//===----------------------------------------------------------------------===//

// Events are used for defining synchronization scopes within command buffers.
// An event only exists within a single CommandBuffer and must not be used
// across command buffers from the same device or others.
//
// See iree_hal_command_buffer_signal_event and
// iree_hal_command_buffer_wait_events for more info.
//
// Maps to VkEvent:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkEvent.html
typedef struct iree_hal_event_t iree_hal_event_t;

// Creates an event for recording into command buffers.
// The returned event object is only usable with the device it is created on and
// only on the queues specified. Events must only be used to synchronize within
// the same queue.
IREE_API_EXPORT iree_status_t iree_hal_event_create(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event);

// Retains the given |event| for the caller.
IREE_API_EXPORT void iree_hal_event_retain(iree_hal_event_t* event);

// Releases the given |event| from the caller.
IREE_API_EXPORT void iree_hal_event_release(iree_hal_event_t* event);

//===----------------------------------------------------------------------===//
// iree_hal_event_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_event_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_event_t* event);
} iree_hal_event_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_event_vtable_t);

IREE_API_EXPORT void iree_hal_event_destroy(iree_hal_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EVENT_H_
