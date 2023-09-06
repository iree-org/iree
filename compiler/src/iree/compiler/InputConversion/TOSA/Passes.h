// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_TOSA_PASSES_H_
#define IREE_COMPILER_INPUTCONVERSION_TOSA_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs input legalization for specific combination of input dialects.
void buildTOSAInputConversionPassPipeline(OpPassManager &passManager);

void registerTOSAConversionPassPipeline();

//------------------------------------------------------------------------------
// Conversions into Linalg
//------------------------------------------------------------------------------

// Verifies a module being input to the core compiler pipeline only contains
// IR structures that are supported at that level.
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyCompilerTOSAInputLegality();

// Set of patterns for materializing TOSA operations to linalg_ext.
void populateTosaToLinalgExtPatterns(RewritePatternSet *patterns);

// Creates a pass that converts TOSA operations to linalg_ext.
std::unique_ptr<OperationPass<func::FuncOp>> createTosaToLinalgExt();

// Creates a pass that converts i48 to i64.
std::unique_ptr<OperationPass<func::FuncOp>> createConverti48Toi64();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerTOSAConversionPasses();

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_INPUTCONVERSION_TOSA_PASSES_H_
