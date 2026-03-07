// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/proactor.h"

//===----------------------------------------------------------------------===//
// iree_async_proactor_initialize
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_async_proactor_initialize(
    const iree_async_proactor_vtable_t* vtable, iree_string_view_t debug_name,
    iree_allocator_t allocator, iree_async_proactor_t* out_proactor) {
  IREE_ASSERT_ARGUMENT(vtable);
  (void)debug_name;
  iree_atomic_ref_count_init(&out_proactor->ref_count);
  out_proactor->vtable = vtable;
  out_proactor->allocator = allocator;
  out_proactor->progress_list = NULL;
  IREE_TRACE({
    iree_host_size_t copy_length =
        iree_min(debug_name.size, sizeof(out_proactor->debug_name) - 1);
    memcpy(out_proactor->debug_name, debug_name.data, copy_length);
    out_proactor->debug_name[copy_length] = '\0';
  });
}

//===----------------------------------------------------------------------===//
// Progress callback management
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_async_proactor_register_progress(
    iree_async_proactor_t* proactor, iree_async_progress_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(entry);
  IREE_ASSERT_ARGUMENT(entry->fn);
  entry->remove_requested = false;
  entry->next = proactor->progress_list;
  proactor->progress_list = entry;
}

IREE_API_EXPORT void iree_async_proactor_unregister_progress(
    iree_async_proactor_t* proactor, iree_async_progress_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(entry);
  iree_async_progress_entry_t** prev = &proactor->progress_list;
  while (*prev) {
    if (*prev == entry) {
      *prev = entry->next;
      entry->next = NULL;
      return;
    }
    prev = &(*prev)->next;
  }
}

IREE_API_EXPORT iree_host_size_t
iree_async_proactor_run_progress(iree_async_proactor_t* proactor) {
  iree_host_size_t total = 0;
  iree_async_progress_entry_t** prev = &proactor->progress_list;
  while (*prev) {
    iree_async_progress_entry_t* entry = *prev;
    iree_async_progress_entry_t* next = entry->next;
    total += entry->fn(entry->user_data);
    if (entry->remove_requested) {
      // Unlink from list before calling on_remove. After on_remove the entry
      // may be freed (e.g., deactivation completion frees the owning object).
      *prev = next;
      void (*on_remove)(void*) = entry->on_remove;
      void* on_remove_user_data = entry->user_data;
      entry->next = NULL;
      entry->remove_requested = false;
      if (on_remove) {
        on_remove(on_remove_user_data);
      }
      // Do not access entry after on_remove — it may be freed.
    } else {
      prev = &entry->next;
    }
  }
  return total;
}
