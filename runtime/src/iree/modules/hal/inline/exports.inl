// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
//         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
//         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
//         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
//          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
//
//===----------------------------------------------------------------------===//
//
// This file will be auto generated from hal_inline.imports.mlir in the future;
// for now it's modified by hand but with strict alphabetical sorting required.
// The order of these functions must be sorted ascending by name in a way
// compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, target_fn, arg_type, ret_type)

// clang-format off

EXPORT_FN("buffer.allocate", iree_hal_inline_module_buffer_allocate, iI, rr)
EXPORT_FN("buffer.allocate.initialized", iree_hal_inline_module_buffer_allocate_initialized, irII, rr)
EXPORT_FN("buffer.length", iree_hal_inline_module_buffer_length, r, I)
EXPORT_FN("buffer.storage", iree_hal_inline_module_buffer_storage, r, r)
EXPORT_FN("buffer.subspan", iree_hal_inline_module_buffer_subspan, rII, r)
EXPORT_FN("buffer.wrap", iree_hal_inline_module_buffer_wrap, rII, r)

EXPORT_FN("buffer_view.assert", iree_hal_inline_module_buffer_view_assert, rriiCID, v)
EXPORT_FN("buffer_view.buffer", iree_hal_inline_module_buffer_view_buffer, r, r)
EXPORT_FN("buffer_view.create", iree_hal_inline_module_buffer_view_create, rIIiiCID, r)
EXPORT_FN("buffer_view.dim", iree_hal_inline_module_buffer_view_dim, ri, I)
EXPORT_FN("buffer_view.element_type", iree_hal_inline_module_buffer_view_element_type, r, i)
EXPORT_FN("buffer_view.encoding_type", iree_hal_inline_module_buffer_view_encoding_type, r, i)
EXPORT_FN("buffer_view.rank", iree_hal_inline_module_buffer_view_rank, r, i)
EXPORT_FN("buffer_view.trace", iree_hal_inline_module_buffer_view_trace, rCrD, v)

EXPORT_FN("device.query.i64", iree_hal_inline_module_device_query_i64, rr, iI)

// clang-format on
