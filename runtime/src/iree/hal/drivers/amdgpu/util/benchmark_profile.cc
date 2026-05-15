// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/benchmark_profile.h"

typedef struct iree_hal_amdgpu_benchmark_discard_profile_sink_t {
  // Resource header for iree_hal_profile_sink_t lifetime management.
  iree_hal_resource_t resource;
  // Host allocator used for sink lifetime.
  iree_allocator_t host_allocator;
  // Number of profile chunks observed by the sink.
  uint64_t write_count;
  // Total bytes observed across all profile chunk iovecs.
  uint64_t payload_byte_count;
} iree_hal_amdgpu_benchmark_discard_profile_sink_t;

static void iree_hal_amdgpu_benchmark_discard_profile_sink_destroy(
    iree_hal_profile_sink_t* base_sink) {
  iree_hal_amdgpu_benchmark_discard_profile_sink_t* sink =
      (iree_hal_amdgpu_benchmark_discard_profile_sink_t*)base_sink;
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t iree_hal_amdgpu_benchmark_discard_profile_sink_begin(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  (void)base_sink;
  (void)metadata;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_benchmark_discard_profile_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  (void)metadata;
  iree_hal_amdgpu_benchmark_discard_profile_sink_t* sink =
      (iree_hal_amdgpu_benchmark_discard_profile_sink_t*)base_sink;
  ++sink->write_count;
  for (iree_host_size_t i = 0; i < iovec_count; ++i) {
    sink->payload_byte_count += iovecs[i].data_length;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_benchmark_discard_profile_sink_end(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  (void)base_sink;
  (void)metadata;
  (void)session_status_code;
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t
    iree_hal_amdgpu_benchmark_discard_profile_sink_vtable = {
        .destroy = iree_hal_amdgpu_benchmark_discard_profile_sink_destroy,
        .begin_session = iree_hal_amdgpu_benchmark_discard_profile_sink_begin,
        .write = iree_hal_amdgpu_benchmark_discard_profile_sink_write,
        .end_session = iree_hal_amdgpu_benchmark_discard_profile_sink_end,
};

iree_status_t iree_hal_amdgpu_benchmark_discard_profile_sink_create(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  IREE_ASSERT_ARGUMENT(out_sink);
  *out_sink = NULL;
  iree_hal_amdgpu_benchmark_discard_profile_sink_t* sink = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink));
  iree_hal_resource_initialize(
      &iree_hal_amdgpu_benchmark_discard_profile_sink_vtable, &sink->resource);
  sink->host_allocator = host_allocator;
  sink->write_count = 0;
  sink->payload_byte_count = 0;
  *out_sink = (iree_hal_profile_sink_t*)sink;
  return iree_ok_status();
}
