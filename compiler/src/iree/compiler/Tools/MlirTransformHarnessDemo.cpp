// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demo of iree/compiler/Tools/MlirTransformHarness.h usage.
// Can be used with iree-bazel-try for one-shot MLIR transforms.

#include "iree/compiler/Tools/MlirTransformHarness.h"

// Demo 1: Count operations
void countOperations(ModuleOp module) {
  int count = 0;
  module.walk([&](Operation *op) { count++; });
  llvm::outs() << "Total operations: " << count << "\n";
}

// Demo 2: Print operation names
void printOperationNames(ModuleOp module) {
  module.walk([](Operation *op) { llvm::outs() << op->getName() << "\n"; });
}

// Demo 3: Add module attribute
void addModuleAttribute(ModuleOp module) {
  module->setAttr("demo.processed", mlir::UnitAttr::get(module.getContext()));
}

// Use the first demo by default
MLIR_TRANSFORM_MAIN(countOperations)
