// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_TIME_H_
#define IREE_BASE_TIME_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/attributes.h"
#include "iree/base/config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A point in time represented as nanoseconds since unix epoch.
// TODO(benvanik): pick something easy to get into/out-of time_t/etc.
typedef int64_t iree_time_t;

// A time in the infinite past used to indicate "already happened".
// This forces APIs that wait for a point in time to act as a poll and always
// return IREE_STATUS_DEADLINE_EXCEEDED instead of blocking the caller.
#define IREE_TIME_INFINITE_PAST INT64_MIN

// A time in the infinite future used to indicate "never".
// This causes APIs that wait for a point in time to wait however long is needed
// to satisfy the wait condition.
#define IREE_TIME_INFINITE_FUTURE INT64_MAX

// A duration represented as relative nanoseconds.
typedef int64_t iree_duration_t;

// A zero-length duration.
// Like IREE_TIME_INFINITE_PAST this forces APIs that would wait to instead
// return IREE_STATUS_DEADLINE_EXCEEDED immediately.
#define IREE_DURATION_ZERO 0

// An infinite-length duration.
// Like IREE_TIME_INFINITE_FUTURE this causes APIs that wait to do so until
// their wait condition is satisfied without returning early.
#define IREE_DURATION_INFINITE INT64_MAX

// Returns the current system time in unix nanoseconds.
// Depending on the system architecture and power mode this time may have a
// very coarse granularity (on the order of microseconds to milliseconds).
//
// The system timer may not be monotonic; users should ensure when comparing
// times they check for negative values in case the time moves backwards.
IREE_API_EXPORT iree_time_t iree_time_now(void);

// Converts a relative timeout duration to an absolute deadline time.
// This handles the special cases of IREE_DURATION_ZERO and
// IREE_DURATION_INFINITE to avoid extraneous time queries.
IREE_API_EXPORT iree_time_t
iree_relative_timeout_to_deadline_ns(iree_duration_t timeout_ns);

// Converts an absolute deadline time to a relative timeout duration in nanos.
// This handles the special cases of IREE_TIME_INFINITE_PAST and
// IREE_TIME_INFINITE_FUTURE to avoid extraneous time queries.
IREE_API_EXPORT iree_duration_t
iree_absolute_deadline_to_timeout_ns(iree_time_t deadline_ns);

// Converts an absolute deadline time to a relative timeout duration in millis.
// This handles the special cases of IREE_TIME_INFINITE_PAST and
// IREE_TIME_INFINITE_FUTURE to avoid extraneous time queries.
IREE_API_EXPORT uint32_t
iree_absolute_deadline_to_timeout_ms(iree_time_t deadline_ns);

typedef enum iree_timeout_type_e {
  // Timeout is defined by an absolute value `deadline_ns`.
  IREE_TIMEOUT_ABSOLUTE = 0,
  // Timeout is defined by a relative value `timeout_ns`.
  IREE_TIMEOUT_RELATIVE = 1,
} iree_timeout_type_t;

// A timeout defined either by an absolute or relative value.
typedef struct iree_timeout_t {
  iree_timeout_type_t type;
  iree_time_t nanos;
} iree_timeout_t;

// Returns a timeout that will be exceeded immediately.
// This can be used with APIs that would otherwise wait to cause them to poll.
//
// Example:
//   status = iree_wait_for_signal_or_timeout(&obj, iree_immediate_timeout());
//   if (iree_status_is_deadline_exceeded(status)) {
//     // Would have waited indicating the signal has not occurred. If the
//     // timeout was not immediate the call would have blocked the caller.
//   }
static inline iree_timeout_t iree_immediate_timeout(void) {
  iree_timeout_t timeout = {IREE_TIMEOUT_ABSOLUTE, IREE_TIME_INFINITE_PAST};
  return timeout;
}

// Returns true if the |timeout| indicates an immediate/polling/nonblocking
// timeout.
static inline bool iree_timeout_is_immediate(iree_timeout_t timeout) {
  return timeout.type == IREE_TIMEOUT_ABSOLUTE
             ? timeout.nanos == IREE_TIME_INFINITE_PAST
             : timeout.nanos == IREE_DURATION_ZERO;
}

// Returns a timeout that will never be reached.
// This can be used with APIs that can wait to disable the early
// deadline-exceeded returns when a condition is not met. It should be used with
// care as it can complicate program state and make termination more prone to
// hangs. On the other hand, it's really useful to not bother with actual
// deadlines. YMMV.
static inline iree_timeout_t iree_infinite_timeout(void) {
  iree_timeout_t timeout = {IREE_TIMEOUT_ABSOLUTE, IREE_TIME_INFINITE_FUTURE};
  return timeout;
}

// Returns true if the |timeout| indicates an infinite/forever blocking timeout.
static inline bool iree_timeout_is_infinite(iree_timeout_t timeout) {
  return timeout.type == IREE_TIMEOUT_ABSOLUTE
             ? timeout.nanos == IREE_TIME_INFINITE_FUTURE
             : timeout.nanos == IREE_DURATION_INFINITE;
}

// Defines an absolute timeout with the given time in nanoseconds.
static inline iree_timeout_t iree_make_deadline(iree_time_t deadline_ns) {
  iree_timeout_t timeout = {IREE_TIMEOUT_ABSOLUTE, deadline_ns};
  return timeout;
}

// Defines a relative timeout with the given time in nanoseconds.
static inline iree_timeout_t iree_make_timeout_ns(iree_duration_t timeout_ns) {
  iree_timeout_t timeout = {IREE_TIMEOUT_RELATIVE, timeout_ns};
  return timeout;
}

// Defines a relative timeout with the given time in milliseconds.
static inline iree_timeout_t iree_make_timeout_ms(iree_duration_t timeout_ms) {
  iree_timeout_t timeout = {
      IREE_TIMEOUT_RELATIVE,
      timeout_ms == IREE_DURATION_INFINITE ? IREE_DURATION_INFINITE
                                           : timeout_ms * 1000000,
  };
  return timeout;
}

// Converts a timeout from relative to absolute (if it is).
//
// Absolute timeouts (deadlines) are better for long-running tasks or when
// making calls that may complete in stages as relative ones will tend to skew;
// if a wait is performed with a relative timeout of 10ms but it takes 5ms to
// get from the origin of the call to the actual wait using the timeout then
// the total latency of the call may be 15ms (5ms to prepare + 10ms on the
// wait). Instead if an absolute deadline is used the caller can ensure that
// the total time spent in the operation happens regardless of the intervening
// work that happens.
//
// For this reason IREE internal APIs try to convert to absolute times and users
// may be able to reduce overhead by populating the times as absolute to start
// with via iree_make_deadline.
static inline void iree_convert_timeout_to_absolute(iree_timeout_t* timeout) {
  if (timeout->type == IREE_TIMEOUT_RELATIVE) {
    timeout->type = IREE_TIMEOUT_ABSOLUTE;
    timeout->nanos = iree_relative_timeout_to_deadline_ns(timeout->nanos);
  }
}

// Returns an absolute deadline in nanoseconds from the given timeout.
static inline iree_time_t iree_timeout_as_deadline_ns(iree_timeout_t timeout) {
  return timeout.type == IREE_TIMEOUT_ABSOLUTE
             ? timeout.nanos
             : iree_relative_timeout_to_deadline_ns(timeout.nanos);
}

// Returns the earliest timeout between |lhs| and |rhs|.
static inline iree_timeout_t iree_timeout_min(iree_timeout_t lhs,
                                              iree_timeout_t rhs) {
  iree_convert_timeout_to_absolute(&lhs);
  iree_convert_timeout_to_absolute(&rhs);
  return iree_make_deadline(lhs.nanos < rhs.nanos ? lhs.nanos : rhs.nanos);
}

// Waits until |deadline_ns| (or longer), putting the calling thread to sleep.
// The precision of this varies across platforms and may have a minimum
// granularity anywhere between microsecond to milliseconds.
// Returns true if the sleep completed successfully and false if it was aborted.
bool iree_wait_until(iree_time_t deadline_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_TIME_H_
