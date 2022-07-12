// Copyright 2021 The IREE Authors
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
// This file matches the vmvx.imports.mlir in the compiler. It'd be nice to
// autogenerate this as the order of these functions must be sorted ascending by
// name in a way compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, target_fn, arg_struct, arg_type, ret_type)

// clang-format off

EXPORT_FN("add.2d.f32", iree_vmvx_add2d_f32, binary2d, rIIIrIIIrIIIII, v)
EXPORT_FN("copy.2d.x16", iree_vmvx_copy2d_x16, unary2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x32", iree_vmvx_copy2d_x32, unary2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x64", iree_vmvx_copy2d_x64, unary2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x8", iree_vmvx_copy2d_x8, unary2d, rIIIrIIIII, v)
EXPORT_FN("fill.2d.x32", iree_vmvx_fill2d_x32, fill2d_x32, irIIII, v)
EXPORT_FN("matmul.f32f32f32", iree_vmvx_matmul_f32f32f32, matmul_f32, rIIrIIrIIIIIffi, v)

// clang-format on
