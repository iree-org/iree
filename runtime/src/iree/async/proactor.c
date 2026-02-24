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
  IREE_TRACE({
    iree_host_size_t copy_length =
        iree_min(debug_name.size, sizeof(out_proactor->debug_name) - 1);
    memcpy(out_proactor->debug_name, debug_name.data, copy_length);
    out_proactor->debug_name[copy_length] = '\0';
  });
}
