// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULES_IO_STREAM_MODULE_H_
#define IREE_MODULES_IO_STREAM_MODULE_H_

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/io/stream.h"
#include "iree/vm/api.h"

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_io_stream, iree_io_stream_t);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the custom types used by the I/O stream module.
IREE_API_EXPORT iree_status_t
iree_io_stream_module_register_types(iree_vm_instance_t* instance);

typedef struct {
#if IREE_FILE_IO_ENABLE
  FILE* stdin_handle;
  FILE* stdout_handle;
  FILE* stderr_handle;
#else
  int unavailable;
#endif  // IREE_FILE_IO_ENABLE
} iree_io_stream_console_files_t;

// Creates a module for user stream I/O using the provided stdin/stdout/stderr.
// Any may be NULL to indicate the stream is not available.
IREE_API_EXPORT iree_status_t iree_io_stream_module_create(
    iree_vm_instance_t* instance, iree_io_stream_console_files_t console_files,
    iree_allocator_t host_allocator,
    iree_vm_module_t** IREE_RESTRICT out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULES_IO_STREAM_MODULE_H_
