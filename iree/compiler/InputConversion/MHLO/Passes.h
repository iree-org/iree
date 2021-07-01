// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES_H_
#define IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs input legalization for specific combination of input dialects.
void buildMHLOInputConversionPassPipeline(OpPassManager &passManager);

void registerMHLOConversionPassPipelines();

//------------------------------------------------------------------------------
// Conversions into Linalg
//------------------------------------------------------------------------------

// Legalizes the input types to those supported by the flow dialect.
// This will fail if types that cannot be supported at all are present, however
// conditionally supported types (based on availability, etc) may still be
// allowed to pass through successfully.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeInputTypesPass();

/// Creates XLA-HLO to Linalg on tensors transformation pass.
std::unique_ptr<OperationPass<FuncOp>> createMHLOToLinalgOnTensorsPass();

/// Creates XLA-HLO to LinalgExt and Flow transformation pass.
std::unique_ptr<OperationPass<FuncOp>>
createConvertAndDistributeMHLOToLinalgExtPass();

/// Creates XLA-HLO preprocessing transformation pass. In this pass we should
/// have all mhlo -> mhlo transformations that are shared between all
/// backends.
std::unique_ptr<OperationPass<FuncOp>> createMHLOToMHLOPreprocessingPass();

// Verifies a module being input to the core compiler pipeline only contains
// IR structures that are supported at that level.
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyCompilerMHLOInputLegality();

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<FuncOp>> createTestMHLOConvertComplexToRealPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerMHLOConversionPasses();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES_H_
