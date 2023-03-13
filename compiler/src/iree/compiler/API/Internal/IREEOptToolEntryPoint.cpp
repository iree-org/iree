// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for iree-opt and derived binaries.
//
// Based on mlir-opt but registers the passes and dialects we care about.

#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Tools/init_targets.h"
#include "iree/compiler/tool_entry_points_api.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int ireeOptRunMain(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();

  // Register the pass to drop embedded transform dialect IR.
  // TODO: this should be upstreamed.
  mlir::linalg::transform::registerDropSchedulePass();

  if (failed(MlirOptMain(argc, argv, "IREE modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
