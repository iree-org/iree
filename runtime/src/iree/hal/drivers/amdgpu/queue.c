// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_initialize(
    hsa_agent_t agent, iree_allocator_t host_allocator,
    iree_hal_amdgpu_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->agent = agent;

  // out_queue->control_queue;
  // out_queue->execution_queue;

  // out_queue->trace_buffer;
  // out_queue->scheduler;

  // DO NOT SUBMIT
  iree_status_t status = iree_ok_status();

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_queue_deinitialize(iree_hal_amdgpu_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT

  // queue->scheduler;
  // queue->trace_buffer;

  // queue->execution_queue;
  // queue->control_queue;

  IREE_TRACE_ZONE_END(z0);
}
