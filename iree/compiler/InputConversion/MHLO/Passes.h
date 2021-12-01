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
namespace MHLO {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs input legalization for specific combination of input dialects.
void buildMHLOInputConversionPassPipeline(OpPassManager &passManager);

// Performs some cleanup activities on programs that may have originated from
// an XLA import (or made to interop with it). This involves:
//   - Convert XLA control flow to SCF
//   - Convert SCF control flow to CFG
//   - Flatten tuples in CFG
//   - Canonicalize
// It is unfortunate to lose SCF so early in the process but CFG provides a
// large simplification to tuple heavy programs, and this compromise is taken
// in the name of compatibility.
void buildXLACleanupPassPipeline(OpPassManager &passManager);

void registerMHLOConversionPassPipelines();

//------------------------------------------------------------------------------
// Cleanup passes
//------------------------------------------------------------------------------

// Flattens tuples in functions and CFG control flow. This is a common
// form of MHLO as produced by XLA based systems.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenTuplesInCFGPass();

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

/// Creates XLA-HLO to LinalgExt pass.
std::unique_ptr<OperationPass<FuncOp>> createConvertMHLOToLinalgExtPass();

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

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES_H_
