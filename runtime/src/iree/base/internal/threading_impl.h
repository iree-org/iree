// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_THREADING_IMPL_H_
#define IREE_BASE_INTERNAL_THREADING_IMPL_H_

// Ensure that any posix header we include exposes GNU stuff. Ignored on
// platforms where we either don't have the GNU stuff or don't have posix
// headers at all.
//
// Note that this does not need to be the same for all compilation units, only
// those we want to access the non-portable features in. It *must* be defined
// prior to including any of the files, though, as otherwise header-guards will
// cause the setting at the time of first inclusion to win.
//
// https://stackoverflow.com/a/5583764
#define _GNU_SOURCE 1

#include <assert.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"

#ifdef __cplusplus
extern "C" {
#endif

// strncpy_s shall copy the first N characters of src to dst, where N is the
// lesser of MaxCount and the length of src.
//
// We have this here patching over GNU being stubborn about supporting this.
// If we start using it other places we can move it into a helper file.
int iree_strncpy_s(char* dest, size_t destsz, const char* src, size_t count);

typedef void (*iree_thread_set_priority_fn_t)(
    iree_thread_t* thread, iree_thread_priority_class_t priority_class);

typedef struct iree_thread_override_list_t {
  iree_thread_set_priority_fn_t set_priority_fn;
  iree_thread_priority_class_t base_priority_class;
  iree_allocator_t allocator;
  iree_slim_mutex_t mutex;
  iree_thread_priority_class_t current_priority_class;
  iree_thread_override_t* head;
} iree_thread_override_list_t;

// Initializes the override list for a thread with |base_priority_class|.
// |set_priority_fn| will be used to update the thread priority when needed.
void iree_thread_override_list_initialize(
    iree_thread_set_priority_fn_t set_priority_fn,
    iree_thread_priority_class_t base_priority_class,
    iree_allocator_t allocator, iree_thread_override_list_t* out_list);

// Deinitializes an override list; expects that all overrides have been removed.
void iree_thread_override_list_deinitialize(iree_thread_override_list_t* list);

// Adds a new override to the list and returns an allocated handle.
iree_thread_override_t* iree_thread_override_list_add(
    iree_thread_override_list_t* list, iree_thread_t* thread,
    iree_thread_priority_class_t priority_class);

// Removes an override from its parent list and deallocates it.
void iree_thread_override_remove_self(iree_thread_override_t* override);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_THREADING_IMPL_H_
