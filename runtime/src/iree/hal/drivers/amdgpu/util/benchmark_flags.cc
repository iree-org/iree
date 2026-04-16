// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/benchmark_flags.h"

#include <cstdio>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

static iree_status_t parse_completion_wait_flags(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value) {
  (void)flag_name;
  iree_async_wait_flags_t* wait_flags = (iree_async_wait_flags_t*)storage;
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *wait_flags = IREE_ASYNC_WAIT_FLAG_NONE;
    return iree_ok_status();
  } else if (iree_string_view_equal(value, IREE_SV("yield"))) {
    *wait_flags = IREE_ASYNC_WAIT_FLAG_YIELD;
    return iree_ok_status();
  } else if (iree_string_view_equal(value, IREE_SV("active"))) {
    *wait_flags = IREE_ASYNC_WAIT_FLAG_ACTIVE;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported completion wait flags");
}

static void print_completion_wait_flags(iree_string_view_t flag_name,
                                        void* storage, FILE* file) {
  const iree_async_wait_flags_t wait_flags =
      *(const iree_async_wait_flags_t*)storage;
  const char* wait_flags_string = "none";
  if (iree_any_bit_set(wait_flags, IREE_ASYNC_WAIT_FLAG_ACTIVE)) {
    wait_flags_string = "active";
  } else if (iree_any_bit_set(wait_flags, IREE_ASYNC_WAIT_FLAG_YIELD)) {
    wait_flags_string = "yield";
  }
  fprintf(file, "--%.*s=\"%s\"\n", (int)flag_name.size, flag_name.data,
          wait_flags_string);
}

static iree_async_wait_flags_t FLAG_completion_wait_flags =
    IREE_ASYNC_WAIT_FLAG_NONE;
IREE_FLAG_CALLBACK(
    parse_completion_wait_flags, print_completion_wait_flags,
    &FLAG_completion_wait_flags, completion_wait_flags,
    "Wait strategy used by benchmark completion waits. One of 'none', "
    "'yield', or 'active'. Active wait spins on the calling thread and should "
    "only be used for latency-sensitive short waits.");

iree_async_wait_flags_t iree_hal_amdgpu_benchmark_completion_wait_flags(void) {
  return FLAG_completion_wait_flags;
}

void iree_hal_amdgpu_benchmark_set_completion_wait_counters(
    benchmark::State& state) {
  state.counters["completion_wait_active"] =
      iree_any_bit_set(FLAG_completion_wait_flags, IREE_ASYNC_WAIT_FLAG_ACTIVE)
          ? 1.0
          : 0.0;
  state.counters["completion_wait_yield"] =
      iree_any_bit_set(FLAG_completion_wait_flags, IREE_ASYNC_WAIT_FLAG_YIELD)
          ? 1.0
          : 0.0;
}
