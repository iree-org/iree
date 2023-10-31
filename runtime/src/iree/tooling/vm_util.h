// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_VM_UTIL_H_
#define IREE_TOOLING_VM_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: this file is not best-practice and needs to be rewritten; consider this
// appropriate only for test code.

// Parses |input_strings| into a variant list of VM scalars and buffers.
// Scalars should be in the format:
//   type=value
// Buffers should be in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in iree/hal/api.h
// Uses |device_allocator| to allocate the buffers.
// The returned variant list must be freed by the caller.
iree_status_t iree_tooling_parse_to_variant_list(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    const iree_string_view_t* input_strings,
    iree_host_size_t input_strings_count, iree_allocator_t host_allocator,
    iree_vm_list_t** out_list);

// Parses |input_strings| into a variant list of VM scalars and buffers.
// Scalars should be in the format:
//   type=value
// Buffers should be in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in iree/hal/api.h
// Uses |device_allocator| to allocate the buffers.
iree_status_t iree_tooling_parse_into_variant_list(
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    const iree_string_view_t* input_strings,
    iree_host_size_t input_strings_count, iree_allocator_t host_allocator,
    iree_vm_list_t* list);

// Appends fences to |list| if the invocation model of |function| requires them.
// If no |wait_fence| is provided then the invocation will begin immediately.
// The caller must wait on the returned |out_signal_fence| before accessing the
// contents of any buffers returned from the invocation.
iree_status_t iree_tooling_append_async_fence_inputs(
    iree_vm_list_t* list, const iree_vm_function_t* function,
    iree_hal_device_t* device, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t** out_signal_fence);

// Transfers all buffers in |list| to ones using |target_params|.
// If no |wait_fence| is provided then the transfer will begin immediately.
// If no |signal_fence| is provided then the call will block until the transfer
// completes.
iree_status_t iree_tooling_transfer_variant_list(
    iree_hal_device_t* device, iree_vm_list_t* list,
    iree_hal_allocator_t* target_allocator,
    iree_hal_buffer_params_t target_params, iree_hal_fence_t* wait_fence,
    iree_hal_fence_t* signal_fence);

// Appends a variant list of VM scalars and buffers to |builder|.
// |list_name| will be printed alongside each element ordinal.
//
// Prints scalars in the format:
//   value
// Prints buffers in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in
// https://github.com/openxla/iree/tree/main/runtime/src/iree/hal/api.h
iree_status_t iree_tooling_append_variant_list_lines(
    iree_string_view_t list_name, iree_vm_list_t* list,
    iree_host_size_t max_element_count, iree_string_builder_t* builder);

// Prints a variant list to a |file|.
// |list_name| will be printed alongside each element ordinal.
iree_status_t iree_tooling_variant_list_fprint(
    iree_string_view_t list_name, iree_vm_list_t* list,
    iree_host_size_t max_element_count, FILE* file);

// Prints a variant |list| to targets based on the provided |output_strings|.
//
// |output_strings| format:
//   (empty): ignore output
//   `-`: print textual form to |file|
//   `@file.npy`: create/overwrite a numpy .npy file.
//   `+file.npy': create/append a numpy .npy file.
iree_status_t iree_tooling_output_variant_list(
    iree_vm_list_t* list, const iree_string_view_t* output_strings,
    iree_host_size_t output_strings_count, iree_host_size_t max_element_count,
    FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOOLING_VM_UTIL_H_
