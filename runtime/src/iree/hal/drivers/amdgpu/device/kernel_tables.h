// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Blits (blit.h)
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X 32
#define IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y 1
#define IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z 1

IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x1,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x2,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x4,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_x8,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_fill_block_x16,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_x1,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_buffer_copy_block_x16,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_BLIT_WORKGROUP_SIZE_Z)

//===----------------------------------------------------------------------===//
// Command buffers (command_buffer.h)
//===----------------------------------------------------------------------===//

// TODO(benvanik): evaluate the optimal size for issue workgroup size.
// Lower sizes (ideally 1) are the most reliable on current hardware that does
// not allow for divergent threads _and_ the assumption that we have a mix of
// commands that causes each thread to diverge, but that's a guess. We may find
// that since 90+% of packets are dispatches we're mostly running the same code
// paths per command and can benefit from thread-level parallelism.
#define IREE_HAL_AMDGPU_CMD_ISSUE_WORKGROUP_SIZE_X 32
#define IREE_HAL_AMDGPU_CMD_ISSUE_WORKGROUP_SIZE_Y 1
#define IREE_HAL_AMDGPU_CMD_ISSUE_WORKGROUP_SIZE_Z 1

#define IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_X 1
#define IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Y 1
#define IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Z 1

// NOTE: these workgroup sizes are guesses and need to be changed.
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_block_issue,
                              IREE_HAL_AMDGPU_CMD_ISSUE_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_CMD_ISSUE_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_CMD_ISSUE_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_dispatch_update,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_branch,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_cond_branch,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Z)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_cmd_return,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_X,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Y,
                              IREE_HAL_AMDGPU_CMD_CONTROL_WORKGROUP_SIZE_Z)

//===----------------------------------------------------------------------===//
// Scheduling (scheduler.h)
//===----------------------------------------------------------------------===//

// NOTE: these workgroup sizes are guesses and need to be changed.
#if 0
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_queue_scheduler_initialize,
                              1, 1, 1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_queue_scheduler_tick, 1, 1,
                              1)
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_queue_retire_entry, 1, 1,
                              1)
#endif

//===----------------------------------------------------------------------===//
// Tracing (tracing.h)
//===----------------------------------------------------------------------===//

#if 0
// NOTE: these workgroup sizes are guesses and need to be changed.
IREE_HAL_AMDGPU_DEVICE_KERNEL(iree_hal_amdgpu_device_trace_buffer_initialize,
                              32, 1, 1)
#endif
