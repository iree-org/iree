// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/wait_handle.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_wait_handle_t
//===----------------------------------------------------------------------===//

void iree_wait_handle_wrap_primitive(
    iree_wait_primitive_type_t primitive_type,
    iree_wait_primitive_value_t primitive_value,
    iree_wait_handle_t* out_handle) {
  memset(out_handle, 0, sizeof(*out_handle));
  out_handle->type = primitive_type;
  out_handle->value = primitive_value;
}

void iree_wait_handle_deinitialize(iree_wait_handle_t* handle) {
  memset(handle, 0, sizeof(*handle));
}

iree_status_t iree_wait_handle_resolve(
    iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_wait_source_resolve_callback_t callback, void* user_data) {
  // The wait handle is stored inline in the wait source's storage[] union.
  iree_wait_handle_t* wait_handle = (iree_wait_handle_t*)wait_source.storage;

  if (iree_wait_handle_is_immediate(*wait_handle)) {
    if (callback) callback(user_data, iree_ok_status());
    return iree_ok_status();
  }

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_status_t status = iree_wait_one(wait_handle, deadline_ns);
  if (callback) {
    callback(user_data, status);
    return iree_ok_status();
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_wait_source_import(
    iree_wait_primitive_t wait_primitive, iree_wait_source_t* out_wait_source) {
  if (iree_wait_primitive_is_immediate(wait_primitive)) {
    *out_wait_source = iree_wait_source_immediate();
  } else {
    iree_wait_handle_t* wait_handle =
        (iree_wait_handle_t*)out_wait_source->storage;
    iree_wait_handle_wrap_primitive(wait_primitive.type, wait_primitive.value,
                                    wait_handle);
    out_wait_source->resolve = iree_wait_handle_resolve;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_wait_source_t iree_event_await(iree_event_t* event) {
  iree_wait_source_t wait_source;
  memcpy(wait_source.storage, event, sizeof(*event));
  wait_source.resolve = iree_wait_handle_resolve;
  return wait_source;
}
