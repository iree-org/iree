// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/threading/wait_address.h"

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !defined(IREE_RUNTIME_USE_FUTEX)

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

#include "iree/base/threading/call_once.h"

//===----------------------------------------------------------------------===//
// Pthread wait-address fallback
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_APPLE)
// Apple supports relative pthread condition waits but not setting the condition
// variable clock to CLOCK_MONOTONIC.
#define IREE_WAIT_ADDRESS_USE_RELATIVE_TIMEDWAIT 1
#elif defined(_POSIX_CLOCK_SELECTION) && (_POSIX_CLOCK_SELECTION >= 0)
// POSIX platforms with selectable condition variable clocks can use IREE's
// monotonic host time directly.
#define IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK 1
#else
// Fallback platforms convert from monotonic absolute time to realtime absolute
// time at the point of wait.
#define IREE_WAIT_ADDRESS_USE_RELATIVE_FALLBACK 1
#endif  // IREE_PLATFORM_APPLE

#define IREE_WAIT_ADDRESS_BUCKET_COUNT 256

typedef struct iree_wait_address_bucket_t {
  // Guards condition wait state for all addresses hashed to this bucket.
  pthread_mutex_t mutex;

  // Broadcast when any address in this bucket may have changed.
  pthread_cond_t condition;
} iree_wait_address_bucket_t;

static iree_once_flag iree_wait_address_initialize_once = IREE_ONCE_FLAG_INIT;
static iree_wait_address_bucket_t
    iree_wait_address_buckets[IREE_WAIT_ADDRESS_BUCKET_COUNT];

static void iree_wait_address_initialize_buckets(void) {
#if defined(IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK)
  pthread_condattr_t condition_attribute;
  pthread_condattr_init(&condition_attribute);
  pthread_condattr_setclock(&condition_attribute, CLOCK_MONOTONIC);
#endif  // IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK

  for (iree_host_size_t i = 0; i < IREE_WAIT_ADDRESS_BUCKET_COUNT; ++i) {
    iree_wait_address_bucket_t* bucket = &iree_wait_address_buckets[i];
    pthread_mutex_init(&bucket->mutex, NULL);
#if defined(IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK)
    pthread_cond_init(&bucket->condition, &condition_attribute);
#else
    pthread_cond_init(&bucket->condition, NULL);
#endif  // IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK
  }

#if defined(IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK)
  pthread_condattr_destroy(&condition_attribute);
#endif  // IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK
}

static iree_wait_address_bucket_t* iree_wait_address_bucket_for(
    const iree_atomic_int32_t* address) {
  uintptr_t address_bits = (uintptr_t)address >> 2;
  address_bits ^= address_bits >> 16;
  address_bits *= (uintptr_t)0x7feb352d;
  address_bits ^= address_bits >> 15;
  return &iree_wait_address_buckets[address_bits &
                                    (IREE_WAIT_ADDRESS_BUCKET_COUNT - 1)];
}

static int iree_wait_address_condition_wait(iree_wait_address_bucket_t* bucket,
                                            iree_time_t deadline_ns) {
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    return pthread_cond_wait(&bucket->condition, &bucket->mutex);
  }
  if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    return ETIMEDOUT;
  }

#if defined(IREE_WAIT_ADDRESS_USE_CONDATTR_CLOCK)
  struct timespec absolute_time = {
      .tv_sec = (time_t)(deadline_ns / 1000000000ull),
      .tv_nsec = (long)(deadline_ns % 1000000000ull),
  };
  return pthread_cond_timedwait(&bucket->condition, &bucket->mutex,
                                &absolute_time);
#elif defined(IREE_WAIT_ADDRESS_USE_RELATIVE_TIMEDWAIT)
  iree_duration_t timeout_ns =
      iree_absolute_deadline_to_timeout_ns(deadline_ns);
  if (timeout_ns <= 0) return ETIMEDOUT;
  struct timespec relative_time = {
      .tv_sec = (time_t)(timeout_ns / 1000000000ull),
      .tv_nsec = (long)(timeout_ns % 1000000000ull),
  };
  return pthread_cond_timedwait_relative_np(&bucket->condition, &bucket->mutex,
                                            &relative_time);
#else
  iree_duration_t timeout_ns =
      iree_absolute_deadline_to_timeout_ns(deadline_ns);
  if (timeout_ns <= 0) return ETIMEDOUT;
  struct timespec now_realtime;
  clock_gettime(CLOCK_REALTIME, &now_realtime);
  int64_t deadline_realtime_ns = (int64_t)now_realtime.tv_sec * 1000000000ll +
                                 now_realtime.tv_nsec + timeout_ns;
  struct timespec absolute_time = {
      .tv_sec = (time_t)(deadline_realtime_ns / 1000000000ll),
      .tv_nsec = (long)(deadline_realtime_ns % 1000000000ll),
  };
  return pthread_cond_timedwait(&bucket->condition, &bucket->mutex,
                                &absolute_time);
#endif  // IREE_WAIT_ADDRESS_USE_*
}

iree_status_code_t iree_wait_address_wait_int32(iree_atomic_int32_t* address,
                                                int32_t expected_value,
                                                iree_time_t deadline_ns) {
  if (iree_atomic_load(address, iree_memory_order_acquire) != expected_value) {
    return IREE_STATUS_OK;
  }

  iree_call_once(&iree_wait_address_initialize_once,
                 iree_wait_address_initialize_buckets);
  iree_wait_address_bucket_t* bucket = iree_wait_address_bucket_for(address);

  iree_status_code_t status_code = IREE_STATUS_OK;
  pthread_mutex_lock(&bucket->mutex);
  while (iree_atomic_load(address, iree_memory_order_acquire) ==
         expected_value) {
    int wait_result = iree_wait_address_condition_wait(bucket, deadline_ns);
    if (wait_result == ETIMEDOUT) {
      status_code = IREE_STATUS_DEADLINE_EXCEEDED;
      break;
    }
    if (wait_result != 0) {
      status_code = IREE_STATUS_UNAVAILABLE;
      break;
    }
  }
  pthread_mutex_unlock(&bucket->mutex);
  return status_code;
}

void iree_wait_address_wake_all(iree_atomic_int32_t* address) {
  iree_call_once(&iree_wait_address_initialize_once,
                 iree_wait_address_initialize_buckets);
  iree_wait_address_bucket_t* bucket = iree_wait_address_bucket_for(address);
  pthread_mutex_lock(&bucket->mutex);
  pthread_cond_broadcast(&bucket->condition);
  pthread_mutex_unlock(&bucket->mutex);
}

#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE && !IREE_RUNTIME_USE_FUTEX
