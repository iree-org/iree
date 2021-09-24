// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_COMPILER_API_TOOLS_H
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_COMPILER_API_TOOLS_H

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Runs the IREE compiler main function. This is used to build ireec-like
/// binaries that link against a common shared library.
MLIR_CAPI_EXPORTED int ireeCompilerRunMain(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif  // IREE_LLVM_EXTERNAL_PROJECTS_IREE_COMPILER_API_TOOLS_H
