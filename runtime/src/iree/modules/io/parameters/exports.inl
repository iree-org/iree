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
// This file will be auto generated from io_parameters.imports.mlir in the
// future; for now it's modified by hand but with strict alphabetical sorting
// required. The order of these functions must be sorted ascending by name in a
// way compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, target_fn, arg_type, ret_type)

// clang-format off

EXPORT_FN("gather", iree_io_parameters_module_gather, rIrrrrrrr, v)
EXPORT_FN("load", iree_io_parameters_module_load, rIrrrrIIiiI, r)
EXPORT_FN("read", iree_io_parameters_module_read, rIrrrrIrII, v)
EXPORT_FN("scatter", iree_io_parameters_module_scatter, rIrrrrrrr, v)
EXPORT_FN("write", iree_io_parameters_module_write, rIrrrrIrII, v)

// clang-format on
