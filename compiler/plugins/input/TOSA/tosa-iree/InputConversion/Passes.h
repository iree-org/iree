// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOSA_IREE_INPUTCONVERSION_PASSES_H_
#define TOSA_IREE_INPUTCONVERSION_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs input legalization for specific combination of input dialects.
void buildTOSAInputConversionPassPipeline(OpPassManager &passManager);

void registerTOSAConversionPassPipeline();

//------------------------------------------------------------------------------
// Conversions from TOSA into Linalg and other core IREE dialects
//------------------------------------------------------------------------------

// Set of patterns for materializing TOSA operations to linalg_ext.
void populateTosaToLinalgExtPatterns(RewritePatternSet *patterns);

// Converts i48 to i64.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConverti48Toi64();

// Strips the signed/unsigned portion off of tensors.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createStripSignednessPass();

// Converts TOSA operations to linalg_ext.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTosaToLinalgExt();

// Verifies that a module only contains IR structures that are supported by the
// core compiler.
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyCompilerTOSAInputLegality();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerTOSAConversionPasses();

} // namespace mlir::iree_compiler

#endif // TOSA_IREE_INPUTCONVERSION_PASSES_H_
