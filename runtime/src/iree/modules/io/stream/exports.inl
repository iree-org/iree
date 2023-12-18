// Copyright 2023 The IREE Authors
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
// This file will be auto generated from io_stream.imports.mlir in the
// future; for now it's modified by hand but with strict alphabetical sorting
// required. The order of these functions must be sorted ascending by name in a
// way compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, target_fn, arg_type, ret_type)

// clang-format off

EXPORT_FN("console.stderr", iree_io_stream_module_console_stderr, v, r)
EXPORT_FN("console.stdin", iree_io_stream_module_console_stdin, v, r)
EXPORT_FN("console.stdout", iree_io_stream_module_console_stdout, v, r)

EXPORT_FN("length", iree_io_stream_module_length, r, I)
EXPORT_FN("offset", iree_io_stream_module_offset, r, I)

EXPORT_FN("read.byte", iree_io_stream_module_read_byte, r, i)
EXPORT_FN("read.bytes", iree_io_stream_module_read_bytes, rrII, I)
EXPORT_FN("read.delimiter", iree_io_stream_module_read_delimiter, ri, r)

EXPORT_FN("write.byte", iree_io_stream_module_write_byte, ri, v)
EXPORT_FN("write.bytes", iree_io_stream_module_write_bytes, rrII, v)

// clang-format on
