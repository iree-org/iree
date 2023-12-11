// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_BINDINGS_TFLITE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_BINDINGS_TFLITE_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::TFLite {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that setup a module for use
// with the IREE TFLite runtime bindings.
void buildTransformPassPipeline(OpPassManager &passManager);

void registerTransformPassPipeline();

//===----------------------------------------------------------------------===//
// TFLite bindings support
//===----------------------------------------------------------------------===//

// Wraps all model entry points in a function that is compatible with the
// expected invocation semantics of the IREE TFLite bindings.
std::unique_ptr<OperationPass<ModuleOp>> createWrapEntryPointsPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerPasses() { createWrapEntryPointsPass(); }

} // namespace mlir::iree_compiler::IREE::TFLite

#endif // IREE_COMPILER_BINDINGS_TFLITE_TRANSFORMS_PASSES_H_
