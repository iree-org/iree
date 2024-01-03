// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES_H_
#define IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES_H_

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Forward declare from iree/compiler/PluginAPI/Client.h.
class PipelineExtensions;

namespace mlir::iree_compiler {

#define GEN_PASS_DECL
#include "iree/compiler/InputConversion/Common/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs common input legalization after specific input dialect conversions
// have taken place.
void buildCommonInputConversionPassPipeline(OpPassManager &passManager);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass();
std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass(PipelineExtensions *pipelineExtensions);
std::unique_ptr<OperationPass<ModuleOp>> createIREECheckIllegalDialectsPass();
std::unique_ptr<OperationPass<ModuleOp>> createIREEImportPublicPass();
std::unique_ptr<OperationPass<ModuleOp>> createImportMLProgramPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgQuantizedConvToConvPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgQuantizedMatmulToMatmulPass();
std::unique_ptr<OperationPass<ModuleOp>> createSanitizeModuleNamesPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerCommonInputConversionPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES_H_
