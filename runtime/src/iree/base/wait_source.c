// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/wait_source.h"

#include "iree/base/assert.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_wait_source_t
//===----------------------------------------------------------------------===//

// NOTE: iree_wait_source_import lives in iree/base/internal/wait_handle.c
// for now as that lets us compile out native wait handle support at a coarse
// level.

IREE_API_EXPORT iree_status_t iree_wait_source_export(
    iree_wait_source_t wait_source, iree_wait_primitive_type_t target_type,
    iree_timeout_t timeout, iree_wait_primitive_t* out_wait_primitive) {
  IREE_ASSERT_ARGUMENT(out_wait_primitive);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (IREE_LIKELY(wait_source.ctl)) {
    const iree_wait_source_export_params_t params = {
        .target_type = target_type,
        .timeout = timeout,
    };
    status = wait_source.ctl(wait_source, IREE_WAIT_SOURCE_COMMAND_EXPORT,
                             &params, (void**)out_wait_primitive);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_wait_source_query(
    iree_wait_source_t wait_source, iree_status_code_t* out_wait_status_code) {
  IREE_ASSERT_ARGUMENT(out_wait_status_code);
  *out_wait_status_code = IREE_STATUS_OK;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (IREE_LIKELY(wait_source.ctl)) {
    status = wait_source.ctl(wait_source, IREE_WAIT_SOURCE_COMMAND_QUERY, NULL,
                             (void**)out_wait_status_code);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_wait_source_wait_one(
    iree_wait_source_t wait_source, iree_timeout_t timeout) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture time as an absolute value as we don't know when it's going to run.
  iree_convert_timeout_to_absolute(&timeout);

  iree_status_t status = iree_ok_status();
  if (IREE_LIKELY(wait_source.ctl)) {
    const iree_wait_source_wait_params_t params = {
        .timeout = timeout,
    };
    status = wait_source.ctl(wait_source, IREE_WAIT_SOURCE_COMMAND_WAIT_ONE,
                             &params, NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_wait_source_delay
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_wait_source_delay_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void* params, void** inout_ptr) {
  iree_time_t delay_deadline_ns = (iree_time_t)wait_source.data;
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t* out_wait_status_code = (iree_status_code_t*)inout_ptr;
      *out_wait_status_code = iree_time_now() >= delay_deadline_ns
                                  ? IREE_STATUS_OK
                                  : IREE_STATUS_DEFERRED;
      return iree_ok_status();
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      iree_time_t timeout_deadline_ns = iree_timeout_as_deadline_ns(
          ((const iree_wait_source_wait_params_t*)params)->timeout);
      if (timeout_deadline_ns > delay_deadline_ns) {
        // Delay is before timeout and we can perform a simple sleep.
        return iree_wait_until(delay_deadline_ns)
                   ? iree_ok_status()
                   : iree_status_from_code(IREE_STATUS_DEFERRED);
      } else {
        // Timeout is before deadline, just wait for the deadline. We _may_
        // wake after the delay deadline but can't be sure.
        iree_wait_until(timeout_deadline_ns);
        return iree_time_now() >= delay_deadline_ns
                   ? iree_ok_status()
                   : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
      return iree_status_from_code(IREE_STATUS_DEFERRED);
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "delay wait sources cannot be exported");
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unhandled wait source command");
  }
}
