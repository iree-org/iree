// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// IREE module linker tool.
//
// Usage:
//  iree-link <input-file>.mlir --link-module=lib1.mlir --link-module=lib2.mlir
//  -o output.mlir iree-link <input-file>.mlir --library-path=/path/to/libs -o
//  output.mlir iree-link --list-symbols module.mlir
//
// This tool links MLIR modules by resolving external function declarations.
// For each external function in the input module, it searches for the
// definition in explicitly provided source modules or automatically discovers
// modules based on symbol prefixes (e.g., @prefix.name → prefix.mlir).

#include "iree/compiler/tool_entry_points_api.h"

int main(int argc, char** argv) { return ireeLinkRunMain(argc, argv); }
