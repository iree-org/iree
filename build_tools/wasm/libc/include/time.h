// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// <time.h> for wasm32.
// Time is provided by the JS host via performance.now().

#ifndef IREE_WASM_LIBC_TIME_H_
#define IREE_WASM_LIBC_TIME_H_

#include <stddef.h>
#include <stdint.h>

typedef int64_t time_t;
typedef int32_t clock_t;

struct timespec {
  time_t tv_sec;
  long tv_nsec;
};

#define CLOCKS_PER_SEC ((clock_t)1000000)

// Clock IDs.
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1

clock_t clock(void);
time_t time(time_t* timer);
int clock_gettime(int clock_id, struct timespec* tp);

#endif  // IREE_WASM_LIBC_TIME_H_
