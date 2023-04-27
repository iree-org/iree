// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/channel.h"

#include <stddef.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_channel_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(channel, method_name) \
  IREE_HAL_VTABLE_DISPATCH(channel, iree_hal_channel, method_name)

IREE_HAL_API_RETAIN_RELEASE(channel);

IREE_API_EXPORT iree_status_t iree_hal_channel_create(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_channel);
  *out_channel = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, create_channel)(
          device, queue_affinity, params, out_channel);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_channel_query_rank_and_count(
    const iree_hal_channel_t* channel, int32_t* out_rank, int32_t* out_count) {
  IREE_ASSERT_ARGUMENT(channel);
  int32_t rank = 0;
  int32_t count = 0;
  _VTABLE_DISPATCH(channel, query_rank_and_count)(channel, &rank, &count);
  if (out_rank) *out_rank = rank;
  if (out_count) *out_count = count;
}

IREE_API_EXPORT int32_t
iree_hal_channel_rank(const iree_hal_channel_t* channel) {
  int32_t rank = 0;
  iree_hal_channel_query_rank_and_count(channel, &rank, NULL);
  return rank;
}

IREE_API_EXPORT int32_t
iree_hal_channel_count(const iree_hal_channel_t* channel) {
  int32_t count = 0;
  iree_hal_channel_query_rank_and_count(channel, NULL, &count);
  return count;
}
