// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_PARAMETER_INDEX_PROVIDER_H_
#define IREE_IO_PARAMETER_INDEX_PROVIDER_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reasonable default for the `max_concurrent_operations` parameter.
#define IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS 16

// Creates a parameter provider serving from the provided |index|.
// As parameters are operated on their files will be registered with the devices
// they are used on and cached for future requests.
//
// |max_concurrent_operations| can be used to limit how many file operations as
// part of a gather or scatter are allowed to be in-flight at a time. A lower
// number can reduce system resource requirements during the operation (less
// transient memory required, etc) while increasing latency (lower I/O
// utilization).
IREE_API_EXPORT iree_status_t iree_io_parameter_index_provider_create(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_host_size_t max_concurrent_operations, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_PARAMETER_INDEX_PROVIDER_H_
