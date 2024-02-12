// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_FUNCTION_IO_H_
#define IREE_TOOLING_FUNCTION_IO_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/stream.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

// Parses zero or more variants from the provided |specs| list.
// |device_allocator| is used for any HAL buffers required and on devices that
// require it an optional |device| will be used for transfer operations.
//
// Supported input string specifiers (and examples):
//  - Special values:
//    `(null)` (a null vm ref)
//  - Primitive value types:
//    `123` (i32)
//    `3.14` (f32)
//  - Shaped tensor types (using HAL buffer view parsing):
//    `f32=1.2` (tensor<f32>)
//    `2x2xf32=1,2,3,4` (tensor<2x2xf32>)
//    `2x2xi32=[[1 2][3 4]]` (tensor<2x2xi32>)
//  - Numpy files:
//    `@file.npy` (first array from the file)
//    `+file.npy` (next array from the file)
//    `*file.npy` (all following arrays from the file)
//  - Binary files:
//    `2x2xf32=@file.ext` (dense tensor<2x2xf32> at the start of the file)
//    `4xf32=+file.ext` (dense tensor<4xf32> following the prior input)
//  - Storage buffers for output arguments (shape/type used for sizing):
//    `&4xf32` (tensor<4xf32> as a HAL buffer for output operands)
//    `&4xf32=1,2,3,4` (tensor<4xf32> storage with an initial value)
iree_status_t iree_tooling_parse_variants(
    iree_string_view_t cconv, iree_string_view_list_t specs,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_vm_list_t** out_list);

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

// Formats a variant list to |builder| as strings with newlines between them.
// |list_name| will be printed alongside each element ordinal. When printing
// |max_element_count| is used to limit the number of buffer view elements.
// The textual format of this is subject to change.
//
// Prints scalars in the format:
//   value
// Prints buffers in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in
// https://github.com/openxla/iree/tree/main/runtime/src/iree/hal/api.h
iree_status_t iree_tooling_format_variants(iree_string_view_t list_name,
                                           iree_vm_list_t* list,
                                           iree_host_size_t max_element_count,
                                           iree_string_builder_t* builder);

// Prints a variant list to |stream|.
// |list_name| will be printed alongside each element ordinal. When printing
// |max_element_count| is used to limit the number of buffer view elements.
// The textual format of this is subject to change.
//
// Prints scalars in the format:
//   value
// Prints buffers in the IREE standard shaped buffer format:
//   [shape]xtype=[value]
// described in
// https://github.com/openxla/iree/tree/main/runtime/src/iree/hal/api.h
iree_status_t iree_tooling_print_variants(iree_string_view_t list_name,
                                          iree_vm_list_t* list,
                                          iree_host_size_t max_element_count,
                                          iree_io_stream_t* stream,
                                          iree_allocator_t host_allocator);

//===----------------------------------------------------------------------===//
// Writing
//===----------------------------------------------------------------------===//

// Outputs a variant list to |stream| or the targets defined by |specs|.
// If provided values will be printed to |default_stream| ala
// iree_tooling_print_variants if their spec is `-`. When printing
// |max_element_count| is used to limit the number of buffer
// view elements. The textual format of this is subject to change.
//
// Supported string output specifiers (and examples):
//  - Ignore a list element (don't output):
//    ``
//  - Print to the specified |default_stream| ala iree_tooling_print_variants:
//    `-`
//  - Numpy files:
//    `@file.npy` (write array from the specified file, discarding)
//    `+file.npy` (append array to the specified file)
//  - Binary files:
//    `@file.ext` (write buffer contents to the specified file, discarding)
//    `+file.ext` (append buffer contents to the specified file)
iree_status_t iree_tooling_write_variants(iree_vm_list_t* list,
                                          iree_string_view_list_t specs,
                                          iree_host_size_t max_element_count,
                                          iree_io_stream_t* default_stream,
                                          iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOOLING_FUNCTION_IO_H_
