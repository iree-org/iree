// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROCESS_RESULTS_H_
#define IREE_TOOLING_PROCESS_RESULTS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/stream.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Handles either printing/writing the |results| of an invocation or checking
// them against expected values (basic pass/fail testing) as specified by the
// --output=/--expected_output= flags. Textual output is written to |stream|.
// Returns the process result code in |out_exit_code| (0 for success).
iree_status_t iree_tooling_process_results(iree_hal_device_t* device,
                                           iree_string_view_t results_cconv,
                                           iree_vm_list_t* results,
                                           iree_io_stream_t* stream,
                                           iree_allocator_t host_allocator,
                                           int* out_exit_code);

// Processes the |results| of an invocation with stdout as the output stream.
// Refer to iree_tooling_process_results for details.
iree_status_t iree_tooling_process_results_and_print(
    iree_hal_device_t* device, iree_string_view_t results_cconv,
    iree_vm_list_t* results, iree_allocator_t host_allocator,
    int* out_exit_code);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROCESS_RESULTS_H_
