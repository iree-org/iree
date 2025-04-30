// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A test case reduction tool which reduces the size of a give test case,
// producing the same error as the original test case.
//
// Usage:
//  iree-reduce <interesting-script> <test-case>.mlir
//
// The interesting-script must be an executable, which takes an input file as
// the first arguement. The script should return 0 if the input file produces
// the required error as the original file and 1 otherwise.

#include "iree/compiler/tool_entry_points_api.h"

int main(int argc, char **argv) { return ireeReduceRunMain(argc, argv); }
