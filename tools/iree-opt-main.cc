// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for iree-opt and derived binaries.
//
// Based on mlir-opt but registers the passes and dialects we care about.

#include "iree/compiler/API2/ToolEntryPoints.h"
#include "llvm/Support/PrettyStackTrace.h"

int main(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/iree-org/iree/issues and "
      "include the crash backtrace.\n");
  return ireeOptRunMain(argc, argv);
}
