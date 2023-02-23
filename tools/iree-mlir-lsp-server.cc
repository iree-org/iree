// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for the IREE variant of mlir-lsp-server.
//
// See https://mlir.llvm.org/docs/Tools/MLIRLSP/

#include "iree/compiler/tool_entry_points_api.h"

int main(int argc, char **argv) { return ireeMlirLspServerRunMain(argc, argv); }
