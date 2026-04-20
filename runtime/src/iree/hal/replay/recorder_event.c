// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_event.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_event_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_event_t {
  // HAL event resource header for the recording wrapper event.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying event receiving forwarded HAL calls.
  iree_hal_event_t* base_event;
  // Session-local device object id associated with this event.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this event.
  iree_hal_replay_object_id_t event_id;
} iree_hal_replay_recorder_event_t;

static const iree_hal_event_vtable_t iree_hal_replay_recorder_event_vtable;

static bool iree_hal_replay_recorder_event_isa(iree_hal_event_t* base_event) {
  return iree_hal_resource_is(base_event,
                              &iree_hal_replay_recorder_event_vtable);
}

static iree_hal_replay_recorder_event_t* iree_hal_replay_recorder_event_cast(
    iree_hal_event_t* base_event) {
  IREE_HAL_ASSERT_TYPE(base_event, &iree_hal_replay_recorder_event_vtable);
  return (iree_hal_replay_recorder_event_t*)base_event;
}

void iree_hal_replay_recorder_event_make_object_payload(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_hal_replay_event_object_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->queue_affinity = queue_affinity;
  out_payload->flags = flags;
}

iree_status_t iree_hal_replay_recorder_event_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t event_id, iree_hal_event_t* base_event,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_event);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;

  if (iree_hal_replay_recorder_event_isa(base_event)) {
    iree_hal_event_retain(base_event);
    *out_event = base_event;
    return iree_ok_status();
  }

  iree_hal_replay_recorder_event_t* event = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));
  memset(event, 0, sizeof(*event));

  iree_hal_resource_initialize(&iree_hal_replay_recorder_event_vtable,
                               &event->resource);
  event->host_allocator = host_allocator;
  event->recorder = recorder;
  iree_hal_replay_recorder_retain(event->recorder);
  event->base_event = base_event;
  iree_hal_event_retain(event->base_event);
  event->device_id = device_id;
  event->event_id = event_id;

  *out_event = (iree_hal_event_t*)event;
  return iree_ok_status();
}

iree_hal_event_t* iree_hal_replay_recorder_event_base_or_self(
    iree_hal_event_t* event) {
  return iree_hal_replay_recorder_event_isa(event)
             ? iree_hal_replay_recorder_event_cast(event)->base_event
             : event;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_event_id_or_none(
    iree_hal_event_t* event) {
  return event && iree_hal_replay_recorder_event_isa(event)
             ? iree_hal_replay_recorder_event_cast(event)->event_id
             : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

static void iree_hal_replay_recorder_event_destroy(
    iree_hal_event_t* base_event) {
  iree_hal_replay_recorder_event_t* event =
      iree_hal_replay_recorder_event_cast(base_event);
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_event_release(event->base_event);
  iree_hal_replay_recorder_release(event->recorder);
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_event_vtable_t iree_hal_replay_recorder_event_vtable = {
    .destroy = iree_hal_replay_recorder_event_destroy,
};
