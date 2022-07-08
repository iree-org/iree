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
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerAllPasses();
  mlir::iree_compiler::registerHALTargetBackends();

  // Also register the transform interpreter pass so that iree-opt can run
  // transform dialect IR without resorting to a separate file.
  // Resorting to separate files is a convenience for iree-compile to be able to
  // use the transform dialect without requiring special plumbing.
  // Still the preferred mode of execution should be to transport the relevant
  // piece of transform IR in the right location, for each piece of code we
  // want to transform for.
  mlir::linalg::transform::registerTransformDialectInterpreterPass();
  mlir::linalg::transform::registerDropSchedulePass();

  if (failed(MlirOptMain(argc, argv, "IREE modular optimizer driver\n",
                         registry,
                         /*preloadDialectsInContext=*/false))) {
    return 1;
  }
  return 0;
}
