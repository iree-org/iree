// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_CONVERTTENSORTOFLOW_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_CONVERTTENSORTOFLOW_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

void buildTensorToFlowPassPipeline(OpPassManager &passManager);

void registerTensorToFlowPassPipelines();

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createConvertTensorToFlow();

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

void populateTensorToFlowPatterns(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  OwningRewritePatternList &patterns);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_CONVERTTENSORTOFLOW_H_
