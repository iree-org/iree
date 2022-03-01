// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/loop.h"

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_loop_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_loop_call(iree_loop_t loop,
                                             iree_loop_priority_t priority,
                                             iree_loop_callback_fn_t callback,
                                             void* user_data) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_loop_call_params_t params = {
      .callback =
          {
              .fn = callback,
              .user_data = user_data,
          },
      .priority = priority,
  };
  iree_status_t status =
      loop.ctl(loop.self, IREE_LOOP_COMMAND_CALL, &params, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_loop_dispatch(
    iree_loop_t loop, const uint32_t workgroup_count_xyz[3],
    iree_loop_workgroup_fn_t workgroup_callback,
    iree_loop_callback_fn_t completion_callback, void* user_data) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)workgroup_count_xyz[0]);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)workgroup_count_xyz[1]);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)workgroup_count_xyz[2]);

  const iree_loop_dispatch_params_t params = {
      .callback =
          {
              .fn = completion_callback,
              .user_data = user_data,
          },
      .workgroup_fn = workgroup_callback,
      .workgroup_count_xyz =
          {
              workgroup_count_xyz[0],
              workgroup_count_xyz[1],
              workgroup_count_xyz[2],
          },
  };
  iree_status_t status =
      loop.ctl(loop.self, IREE_LOOP_COMMAND_DISPATCH, &params, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_loop_wait_until(iree_loop_t loop, iree_timeout_t timeout,
                     iree_loop_callback_fn_t callback, void* user_data) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture time as an absolute value as we don't know when it's going to run.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  const iree_loop_wait_until_params_t params = {
      .callback =
          {
              .fn = callback,
              .user_data = user_data,
          },
      .deadline_ns = deadline_ns,
  };
  iree_status_t status =
      loop.ctl(loop.self, IREE_LOOP_COMMAND_WAIT_UNTIL, &params, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_loop_wait_one(
    iree_loop_t loop, iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_loop_callback_fn_t callback, void* user_data) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture time as an absolute value as we don't know when it's going to run.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  const iree_loop_wait_one_params_t params = {
      .callback =
          {
              .fn = callback,
              .user_data = user_data,
          },
      .deadline_ns = deadline_ns,
      .wait_source = wait_source,
  };
  iree_status_t status =
      loop.ctl(loop.self, IREE_LOOP_COMMAND_WAIT_ONE, &params, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_loop_wait_multi(
    iree_loop_command_t command, iree_loop_t loop, iree_host_size_t count,
    iree_wait_source_t* wait_sources, iree_timeout_t timeout,
    iree_loop_callback_fn_t callback, void* user_data) {
  if (count == 0) {
    // No wait handles; issue the callback as if it had completed async.
    return iree_loop_call(loop, IREE_LOOP_PRIORITY_DEFAULT, callback,
                          user_data);
  } else if (count == 1) {
    // One wait handle can go down the fast path.
    return iree_loop_wait_one(loop, wait_sources[0], timeout, callback,
                              user_data);
  }

  // Capture time as an absolute value as we don't know when it's going to run.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  const iree_loop_wait_multi_params_t params = {
      .callback =
          {
              .fn = callback,
              .user_data = user_data,
          },
      .deadline_ns = deadline_ns,
      .count = count,
      .wait_sources = wait_sources,
  };
  return loop.ctl(loop.self, command, &params, NULL);
}

IREE_API_EXPORT iree_status_t iree_loop_wait_any(
    iree_loop_t loop, iree_host_size_t count, iree_wait_source_t* wait_sources,
    iree_timeout_t timeout, iree_loop_callback_fn_t callback, void* user_data) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)count);
  iree_status_t status =
      iree_loop_wait_multi(IREE_LOOP_COMMAND_WAIT_ANY, loop, count,
                           wait_sources, timeout, callback, user_data);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_loop_wait_all(
    iree_loop_t loop, iree_host_size_t count, iree_wait_source_t* wait_sources,
    iree_timeout_t timeout, iree_loop_callback_fn_t callback, void* user_data) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (uint64_t)count);
  iree_status_t status =
      iree_loop_wait_multi(IREE_LOOP_COMMAND_WAIT_ALL, loop, count,
                           wait_sources, timeout, callback, user_data);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_loop_drain(iree_loop_t loop,
                                              iree_timeout_t timeout) {
  if (IREE_UNLIKELY(!loop.ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "null loop");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Capture time as an absolute value as we don't know when it's going to run.
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  const iree_loop_drain_params_t params = {
      .deadline_ns = deadline_ns,
  };
  iree_status_t status =
      loop.ctl(loop.self, IREE_LOOP_COMMAND_DRAIN, &params, NULL);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
