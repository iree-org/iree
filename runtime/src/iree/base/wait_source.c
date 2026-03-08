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

IREE_API_EXPORT iree_status_t iree_wait_source_query(
    iree_wait_source_t wait_source, iree_status_code_t* out_wait_status_code) {
  IREE_ASSERT_ARGUMENT(out_wait_status_code);
  *out_wait_status_code = IREE_STATUS_OK;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_LIKELY(wait_source.resolve)) {
    // Resolve with immediate timeout and no callback (synchronous query).
    iree_status_t status =
        wait_source.resolve(wait_source, iree_make_timeout_ms(0),
                            /*callback=*/NULL, /*user_data=*/NULL);
    if (iree_status_is_ok(status)) {
      *out_wait_status_code = IREE_STATUS_OK;
    } else if (iree_status_code(status) == IREE_STATUS_DEADLINE_EXCEEDED) {
      *out_wait_status_code = IREE_STATUS_DEFERRED;
      iree_status_ignore(status);
    } else {
      *out_wait_status_code = iree_status_code(status);
      iree_status_ignore(status);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_wait_source_wait_one(
    iree_wait_source_t wait_source, iree_timeout_t timeout) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture time as an absolute value as we don't know when it's going to run.
  iree_convert_timeout_to_absolute(&timeout);

  iree_status_t status = iree_ok_status();
  if (IREE_LIKELY(wait_source.resolve)) {
    status = wait_source.resolve(wait_source, timeout,
                                 /*callback=*/NULL, /*user_data=*/NULL);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_wait_source_delay
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_wait_source_delay_resolve(
    iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_wait_source_resolve_callback_t callback, void* user_data) {
  iree_time_t delay_deadline_ns = (iree_time_t)wait_source.data;

  // Check if delay has already passed.
  if (iree_time_now() >= delay_deadline_ns) {
    if (callback) callback(user_data, iree_ok_status());
    return iree_ok_status();
  }

  // Not yet reached. Determine the effective wait deadline.
  iree_time_t timeout_deadline_ns = iree_timeout_as_deadline_ns(timeout);
  iree_time_t wait_deadline_ns = timeout_deadline_ns < delay_deadline_ns
                                     ? timeout_deadline_ns
                                     : delay_deadline_ns;

  // Sleep until the earlier of delay deadline and timeout deadline.
  iree_wait_until(wait_deadline_ns);

  // Check if the delay has passed after sleeping.
  bool reached = iree_time_now() >= delay_deadline_ns;
  iree_status_t status =
      reached ? iree_ok_status()
              : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  if (callback) {
    callback(user_data, status);
    return iree_ok_status();
  }
  return status;
}
