// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/executable.h"

typedef struct iree_hal_null_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_null_executable_t;

static const iree_hal_executable_vtable_t iree_hal_null_executable_vtable;

static iree_hal_null_executable_t* iree_hal_null_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_executable_vtable);
  return (iree_hal_null_executable_t*)base_value;
}

iree_status_t iree_hal_null_executable_create(
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate storage for the executable and its associated data structures.
  iree_hal_null_executable_t* executable = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable),
                                (void**)&executable));
  iree_hal_resource_initialize(&iree_hal_null_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;

  // TODO(null): load executable module(s). Note that the input data should be
  // treated as untrusted and should be verified to the best ability the format
  // provides. A target that cannot provide verification will be treated as
  // unsafe. For JIT-style implementations as much work as possible should be
  // done here so that errors can be propagated back to users - do not defer
  // preparation.
  //
  // In general the executable should only retain information required to
  // service the command buffer implementation that will be dispatching entry
  // points within it. Optionally information can be retained for tracing and
  // debugging.
  //
  // Implementations with flexible formats (ELF, etc) can directly use those for
  // metadata as well with custom sections. If an implementation does not have a
  // flexible format or support linking and requires several modules a wrapper
  // can be used instead. In upstream IREE HALs Flatbuffers is used and is the
  // preferred format (zero-copy, mmappable, verifiable, near header-only dep
  // with no binary size or runtime overheads, etc) and is the easiest to use,
  // but you do you.
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "executable not implemented");

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_null_executable_t* executable =
      iree_hal_null_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): release any implementation resources.

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_vtable_t iree_hal_null_executable_vtable = {
    .destroy = iree_hal_null_executable_destroy,
};
