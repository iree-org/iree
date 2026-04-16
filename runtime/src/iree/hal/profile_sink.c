// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/profile_sink.h"

#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_profile_sink_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(sink, method_name) \
  IREE_HAL_VTABLE_DISPATCH(sink, iree_hal_profile_sink, method_name)

IREE_HAL_API_RETAIN_RELEASE(profile_sink);

IREE_API_EXPORT iree_status_t iree_hal_profile_sink_begin_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(sink, begin_session)(sink, metadata);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_profile_sink_write(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(sink, write)(sink, metadata, iovec_count, iovecs);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_profile_sink_end_session(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  IREE_ASSERT_ARGUMENT(sink);
  IREE_ASSERT_ARGUMENT(metadata);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(sink, end_session)(sink, metadata, session_status_code);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
