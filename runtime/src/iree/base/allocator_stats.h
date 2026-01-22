// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_ALLOCATOR_STATS_H_
#define IREE_BASE_ALLOCATOR_STATS_H_

#include "iree/base/allocator.h"

typedef struct iree_slim_mutex_t iree_slim_mutex_t;

// Allocation statistics of the allocator
typedef struct iree_allocator_statistics_t {
  // Peak number of bytes allocated at any one time
  iree_host_size_t bytes_peak;
  // Total number of bytes allocated
  iree_host_size_t bytes_allocated;
  // Total number of bytes freed
  iree_host_size_t bytes_freed;
  // Mutex protecting the statistics
  iree_slim_mutex_t* mutex;
} iree_allocator_statistics_t;

typedef struct iree_allocator_with_stats_t {
  iree_allocator_t base_allocator;
  iree_allocator_statistics_t statistics;
} iree_allocator_with_stats_t;

// Allocator control function for the system allocator with statistics enabled
IREE_API_EXPORT iree_status_t
iree_allocator_stats_ctl(void* self, iree_allocator_command_t command,
                         const void* params, void** inout_ptr);

IREE_API_EXPORT iree_allocator_t
iree_allocator_stats_init(iree_allocator_with_stats_t* stats_allocator,
                          iree_allocator_t base_allocator);

IREE_API_EXPORT void iree_allocator_stats_deinit(
    iree_allocator_with_stats_t* stats_allocator);

// Prints allocator statistics to the given file, if statistics are enabled.
IREE_API_EXPORT iree_status_t iree_allocator_statistics_fprint(
    FILE* file, iree_allocator_with_stats_t* stats_allocator);

#endif  // IREE_BASE_ALLOCATOR_STATS_H_
