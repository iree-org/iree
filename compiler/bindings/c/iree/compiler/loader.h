// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LOADER_H
#define IREE_COMPILER_LOADER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the compiler API stub by loading a libIREECompiler.so
// implementation library.
// Returns true on success. On failure, may log to stderr.
bool ireeCompilerLoadLibrary(const char *libraryPath);

#ifdef __cplusplus
}
#endif

#endif  // IREE_COMPILER_LOADER_H
