// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_PARTITIONER_LOADER_H
#define OPENXLA_PARTITIONER_LOADER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the compiler API stub by loading a libOpenXLAPartitioner.so
// implementation library.
// Returns true on success. On failure, may log to stderr.
bool openxlaPartitionerLoadLibrary(const char *libraryPath);

#ifdef __cplusplus
}
#endif

#endif  // OPENXLA_PARTITIONER_LOADER_H
