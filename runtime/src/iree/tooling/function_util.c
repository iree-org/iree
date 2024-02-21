// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/function_util.h"

#include "iree/modules/hal/module.h"

iree_status_t iree_tooling_append_async_fences(
    iree_vm_list_t* list, iree_vm_function_t function,
    iree_hal_device_t* device, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t** out_signal_fence) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_view_t model = iree_vm_function_lookup_attr_by_name(
      &function, IREE_SV("iree.abi.model"));
  if (!iree_string_view_equal(model, IREE_SV("coarse-fences"))) {
    // Ignore unknown models - the user may have provided their own fences.
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Create the signal fence as a 0->1 transition. The caller will wait on that.
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_create(device, 0ull, &semaphore));
  iree_hal_fence_t* signal_fence = NULL;
  iree_status_t status = iree_hal_fence_create_at(
      semaphore, 1ull, iree_hal_device_host_allocator(device), &signal_fence);
  iree_hal_semaphore_release(semaphore);

  // Append (wait, signal) fences.
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t wait_fence_ref = iree_hal_fence_retain_ref(wait_fence);
    status = iree_vm_list_push_ref_move(list, &wait_fence_ref);
    iree_vm_ref_release(&wait_fence_ref);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_ref_t signal_fence_ref = iree_hal_fence_retain_ref(signal_fence);
    status = iree_vm_list_push_ref_move(list, &signal_fence_ref);
    iree_vm_ref_release(&signal_fence_ref);
  }

  if (iree_status_is_ok(status)) {
    *out_signal_fence = signal_fence;
  } else {
    iree_hal_fence_release(signal_fence);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_tooling_requires_buffer_transfer(
    iree_hal_buffer_t* source_buffer, iree_hal_device_t* target_device,
    iree_hal_buffer_params_t target_params) {
  // TODO(benvanik): if source/target devices don't match or can't be imported
  // then we need a transfer.
  return !iree_all_bits_set(iree_hal_buffer_memory_type(source_buffer),
                            target_params.type) ||
         !iree_all_bits_set(iree_hal_buffer_allowed_usage(source_buffer),
                            target_params.usage);
}

static iree_status_t iree_tooling_setup_buffer_transfer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_hal_allocator_t* target_allocator,
    iree_hal_buffer_params_t target_params,
    iree_hal_buffer_t** out_target_buffer) {
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_allocator);
  IREE_ASSERT_ARGUMENT(out_target_buffer);
  *out_target_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_t* target_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(
              target_allocator, target_params,
              iree_hal_buffer_allocation_size(source_buffer), &target_buffer));

  iree_status_t status = iree_hal_command_buffer_copy_buffer(
      command_buffer, source_buffer, 0, target_buffer, 0,
      iree_hal_buffer_byte_length(source_buffer));

  if (iree_status_is_ok(status)) {
    *out_target_buffer = target_buffer;
  } else {
    iree_hal_buffer_release(target_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_submit_transfer(
    iree_hal_device_t* device, iree_hal_fence_t* wait_fence,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t* command_buffer, iree_hal_fence_t* signal_fence) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  bool needs_wait = signal_fence == NULL;
  if (needs_wait) {
    iree_hal_semaphore_t* semaphore = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_create(device, 0ull, &semaphore));
    status = iree_hal_fence_create_at(
        semaphore, 1ull, iree_hal_device_host_allocator(device), &signal_fence);
    iree_hal_semaphore_release(semaphore);
  } else {
    iree_hal_fence_retain(signal_fence);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, queue_affinity, iree_hal_fence_semaphore_list(wait_fence),
        iree_hal_fence_semaphore_list(signal_fence), 1, &command_buffer);
  }

  if (iree_status_is_ok(status) && needs_wait) {
    status = iree_hal_fence_wait(signal_fence, iree_infinite_timeout());
  }

  iree_hal_fence_release(signal_fence);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tooling_transfer_variants(
    iree_vm_list_t* list, iree_hal_device_t* target_device,
    iree_hal_allocator_t* target_allocator,
    iree_hal_buffer_params_t target_params, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t* signal_fence) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(target_device);
  IREE_ASSERT_ARGUMENT(target_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  // If all buffers are already host-accessible we can skip the transfer.
  bool requires_transfer = false;
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_ref_t value = iree_vm_ref_null();
    IREE_IGNORE_ERROR(iree_vm_list_get_ref_assign(list, i, &value));
    if (iree_hal_buffer_isa(value)) {
      iree_hal_buffer_t* source_buffer = iree_hal_buffer_deref(value);
      if (iree_tooling_requires_buffer_transfer(source_buffer, target_device,
                                                target_params)) {
        requires_transfer = true;
        break;
      }
    } else if (iree_hal_buffer_view_isa(value)) {
      iree_hal_buffer_view_t* source_view = iree_hal_buffer_view_deref(value);
      iree_hal_buffer_t* source_buffer =
          iree_hal_buffer_view_buffer(source_view);
      if (iree_tooling_requires_buffer_transfer(source_buffer, target_device,
                                                target_params)) {
        requires_transfer = true;
        break;
      }
    }
  }
  if (!requires_transfer) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_create(
              target_device,
              IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
                  IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
              IREE_HAL_COMMAND_CATEGORY_TRANSFER, target_params.queue_affinity,
              /*binding_capacity=*/0, &command_buffer));

  iree_status_t status = iree_hal_command_buffer_begin(command_buffer);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
      iree_vm_ref_t value = iree_vm_ref_null();
      IREE_IGNORE_ERROR(iree_vm_list_get_ref_assign(list, i, &value));
      if (iree_hal_buffer_isa(value)) {
        iree_hal_buffer_t* source_buffer = iree_hal_buffer_deref(value);
        if (!iree_tooling_requires_buffer_transfer(source_buffer, target_device,
                                                   target_params)) {
          // Already ok.
          continue;
        }
        iree_hal_buffer_t* target_buffer = NULL;
        status = iree_tooling_setup_buffer_transfer(
            command_buffer, source_buffer, target_allocator, target_params,
            &target_buffer);
        if (!iree_status_is_ok(status)) break;
        status = iree_vm_list_set_buffer_retain(list, i, target_buffer);
        iree_hal_buffer_release(target_buffer);
        if (!iree_status_is_ok(status)) break;
      } else if (iree_hal_buffer_view_isa(value)) {
        iree_hal_buffer_view_t* source_view = iree_hal_buffer_view_deref(value);
        iree_hal_buffer_t* source_buffer =
            iree_hal_buffer_view_buffer(source_view);
        if (!iree_tooling_requires_buffer_transfer(source_buffer, target_device,
                                                   target_params)) {
          // Already ok.
          continue;
        }
        iree_hal_buffer_t* target_buffer = NULL;
        status = iree_tooling_setup_buffer_transfer(
            command_buffer, source_buffer, target_allocator, target_params,
            &target_buffer);
        if (!iree_status_is_ok(status)) break;
        iree_hal_buffer_view_t* target_view = NULL;
        status = iree_hal_buffer_view_create_like(
            target_buffer, source_view,
            iree_hal_allocator_host_allocator(target_allocator), &target_view);
        iree_hal_buffer_release(target_buffer);
        if (!iree_status_is_ok(status)) break;
        status = iree_vm_list_set_buffer_view_retain(list, i, target_view);
        iree_hal_buffer_view_release(target_view);
        if (!iree_status_is_ok(status)) break;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }

  if (iree_status_is_ok(status)) {
    status = iree_tooling_submit_transfer(target_device, wait_fence,
                                          target_params.queue_affinity,
                                          command_buffer, signal_fence);
  }

  iree_hal_command_buffer_release(command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
