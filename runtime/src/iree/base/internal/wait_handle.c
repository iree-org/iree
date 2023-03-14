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

iree_status_t iree_wait_handle_ctl(iree_wait_source_t wait_source,
                                   iree_wait_source_command_t command,
                                   const void* params, void** inout_ptr) {
  iree_wait_handle_t* wait_handle = iree_wait_handle_from_source(&wait_source);
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t* out_wait_status_code = (iree_status_code_t*)inout_ptr;
      if (iree_wait_handle_is_immediate(*wait_handle)) {
        // Immediately resolved.
        *out_wait_status_code = IREE_STATUS_OK;
        return iree_ok_status();
      } else {
        // Poll the handle: a deadline exceeded indicates unresolved.
        iree_status_t status =
            iree_wait_one(wait_handle, IREE_TIME_INFINITE_PAST);
        if (iree_status_is_deadline_exceeded(status)) {
          *out_wait_status_code = IREE_STATUS_DEFERRED;
          return iree_status_ignore(status);
        }
        return status;
      }
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      // Wait for the handle.
      return iree_wait_one(
          wait_handle,
          iree_timeout_as_deadline_ns(
              ((const iree_wait_source_wait_params_t*)params)->timeout));
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT: {
      const iree_wait_primitive_type_t target_type =
          ((const iree_wait_source_export_params_t*)params)->target_type;
      if (target_type != IREE_WAIT_PRIMITIVE_TYPE_ANY &&
          target_type != wait_handle->type) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "requested wait primitive type %d is unavailable; have %d",
            (int)target_type, (int)wait_handle->type);
      }
      iree_wait_primitive_t* out_wait_primitive =
          (iree_wait_primitive_t*)inout_ptr;
      out_wait_primitive->type = wait_handle->type;
      out_wait_primitive->value = wait_handle->value;
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented wait_source command");
  }
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
    out_wait_source->ctl = iree_wait_handle_ctl;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_wait_source_t iree_event_await(iree_event_t* event) {
  iree_wait_source_t wait_source;
  memcpy(wait_source.storage, event, sizeof(*event));
  wait_source.ctl = iree_wait_handle_ctl;
  return wait_source;
}
