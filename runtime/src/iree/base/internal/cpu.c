// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: must be first before _any_ system includes.
#define _GNU_SOURCE

#include "iree/base/internal/cpu.h"

#include "iree/base/target_platform.h"

//===----------------------------------------------------------------------===//
// iree_cpu_*
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

#include <sched.h>

iree_cpu_processor_id_t iree_cpu_query_processor_id(void) {
  // This path is relatively portable and should work on linux/bsd/etc-likes.
  // We may want to use getcpu when available so that we can get the group ID.
  // https://man7.org/linux/man-pages/man3/sched_getcpu.3.html
  //
  // libc implementations can use vDSO and other fun stuff to make this really
  // cheap: http://git.musl-libc.org/cgit/musl/tree/src/sched/sched_getcpu.c
  int id = sched_getcpu();
  return id != -1 ? id : 0;
}

#elif defined(IREE_PLATFORM_WINDOWS)

iree_cpu_processor_id_t iree_cpu_query_processor_id(void) {
  PROCESSOR_NUMBER pn;
  GetCurrentProcessorNumberEx(&pn);
  return 64 * pn.Group + pn.Number;
}

#else

// No implementation.
// We could allow an iree/base/config.h override to externalize this.
iree_cpu_processor_id_t iree_cpu_query_processor_id(void) { return 0; }

#endif  // IREE_PLATFORM_*

void iree_cpu_requery_processor_id(iree_cpu_processor_tag_t* IREE_RESTRICT tag,
                                   iree_cpu_processor_id_t* IREE_RESTRICT
                                       processor_id) {
  IREE_ASSERT_ARGUMENT(tag);
  IREE_ASSERT_ARGUMENT(processor_id);

  // TODO(benvanik): set a frequency for this and use a coarse timer
  // (CLOCK_MONOTONIC_COARSE) to do a ~4-10Hz refresh. We can store the last
  // query time and the last processor ID in the tag and only perform the query
  // if it has changed.

  *processor_id = iree_cpu_query_processor_id();
}
