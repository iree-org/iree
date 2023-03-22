// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_RUN_MODULE_H_
#define IREE_TOOLING_RUN_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Runs the module/function specified on the command line with inputs/outputs.
// Returns the process result code in |out_exit_code| (0 for success).
//
// One or more --module= flags can be used to specify all required modules.
// --function= is used to specify which function in the last module registered
// is to be executed. One --input= flag per function input can be used to
// provide function inputs from textual or file sources. One --output= flag per
// function output can be used to write outputs to a file. Optionally
// --expected_output= flags can be used to perform basic comparisons against
// the actual function outputs. See --help for more information.
iree_status_t iree_tooling_run_module_from_flags(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    int* out_exit_code);

// Runs the module/function specified on the command line with the given
// in-memory main module. Equivalent to iree_tooling_run_module_from_flags but
// the provided |module_contents| are registered with the context prior to
// execution.
//
// Optionally |default_device_uri| can be used to specify which device should
// be used if no --device= flag is provided by the user.
iree_status_t iree_tooling_run_module_with_data(
    iree_vm_instance_t* instance, iree_string_view_t default_device_uri,
    iree_const_byte_span_t module_contents, iree_allocator_t host_allocator,
    int* out_exit_code);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_RUN_MODULE_H_
