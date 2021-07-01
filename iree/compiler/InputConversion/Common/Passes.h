// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES_H_
#define IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Performs input legalization for specific combination of input dialects.
void buildCommonInputConversionPassPipeline(OpPassManager &passManager);

void registerCommonConversionPassPipelines();

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createTopLevelSCFToCFGPass();
std::unique_ptr<OperationPass<FuncOp>> createConvertUpstreamToIREE();

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

void populateConvertUpstreamToIREEPatterns(MLIRContext *context,
                                           TypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerCommonInputConversionPasses();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES_H_
