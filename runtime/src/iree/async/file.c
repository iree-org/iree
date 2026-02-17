// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/file.h"

#include "iree/async/proactor.h"

IREE_API_EXPORT iree_status_t iree_async_file_import(
    iree_async_proactor_t* proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      proactor->vtable->import_file(proactor, primitive, out_file);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Destroys the file, releasing backend resources and closing the underlying
// platform handle. Routed through the proactor that created this file.
// Must not be called while operations referencing this file are in flight.
static void iree_async_file_destroy(iree_async_file_t* file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  file->proactor->vtable->destroy_file(file->proactor, file);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_async_file_retain(iree_async_file_t* file) {
  if (file) {
    iree_atomic_ref_count_inc(&file->ref_count);
  }
}

IREE_API_EXPORT void iree_async_file_release(iree_async_file_t* file) {
  if (file && iree_atomic_ref_count_dec(&file->ref_count) == 1) {
    iree_async_file_destroy(file);
  }
}
