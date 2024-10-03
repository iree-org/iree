// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/channel.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_channel_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_null_channel_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Parent channel this was split from, if any.
  // This is only used to keep the parent channel live for as long as there are
  // any split channels live (including transitive splits).
  iree_hal_channel_t* parent_channel;
} iree_hal_null_channel_t;

static const iree_hal_channel_vtable_t iree_hal_null_channel_vtable;

static iree_hal_null_channel_t* iree_hal_null_channel_cast(
    iree_hal_channel_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_channel_vtable);
  return (iree_hal_null_channel_t*)base_value;
}

static const iree_hal_null_channel_t* iree_hal_null_channel_const_cast(
    const iree_hal_channel_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_channel_vtable);
  return (const iree_hal_null_channel_t*)base_value;
}

iree_status_t iree_hal_null_channel_create(iree_hal_channel_params_t params,
                                           iree_allocator_t host_allocator,
                                           iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(out_channel);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_channel = NULL;

  iree_hal_null_channel_t* channel = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*channel),
                                (void**)&channel));
  iree_hal_resource_initialize(&iree_hal_null_channel_vtable,
                               &channel->resource);
  channel->host_allocator = host_allocator;

  // TODO(null): implement channel setup using params. Note that the id is not
  // retained and must be copied local if needed beyond this function call.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "collective channels not implemented");

  if (iree_status_is_ok(status)) {
    *out_channel = (iree_hal_channel_t*)channel;
  } else {
    iree_hal_channel_release((iree_hal_channel_t*)channel);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_channel_destroy(iree_hal_channel_t* base_channel) {
  iree_hal_null_channel_t* channel = iree_hal_null_channel_cast(base_channel);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = channel->host_allocator;

  // TODO(null): destroy any implementation resources.

  iree_hal_channel_release(channel->parent_channel);
  iree_allocator_free(host_allocator, channel);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_null_channel_split(
    iree_hal_channel_t* base_channel, int32_t color, int32_t key,
    iree_hal_channel_flags_t flags, iree_hal_channel_t** out_split_channel) {
  iree_hal_null_channel_t* channel = iree_hal_null_channel_cast(base_channel);

  // TODO(null): split the channel and get any native resources required.
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "channel splitting not implemented");

  // Wrap the split channel resources in a new HAL channel.
  iree_hal_null_channel_t* split_channel = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(channel->host_allocator, sizeof(*split_channel),
                              (void**)&split_channel);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_null_channel_vtable,
                                 &split_channel->resource);
    split_channel->host_allocator = channel->host_allocator;
    split_channel->parent_channel = base_channel;
    iree_hal_channel_retain(base_channel);

    // TODO(null): transfer ownership of the implementation resources.
  }

  if (iree_status_is_ok(status)) {
    *out_split_channel = (iree_hal_channel_t*)split_channel;
  } else {
    iree_hal_channel_release((iree_hal_channel_t*)split_channel);
  }
  return status;
}

static void iree_hal_null_channel_query_rank_and_count(
    const iree_hal_channel_t* base_channel, int32_t* out_rank,
    int32_t* out_count) {
  const iree_hal_null_channel_t* channel =
      iree_hal_null_channel_const_cast(base_channel);

  // TODO(null): query the rank and count from the implementation or cache them
  // locally to avoid overheads (this may be called frequently).
  (void)channel;
  *out_rank = 0;
  *out_count = 0;
}

static const iree_hal_channel_vtable_t iree_hal_null_channel_vtable = {
    .destroy = iree_hal_null_channel_destroy,
    .split = iree_hal_null_channel_split,
    .query_rank_and_count = iree_hal_null_channel_query_rank_and_count,
};
