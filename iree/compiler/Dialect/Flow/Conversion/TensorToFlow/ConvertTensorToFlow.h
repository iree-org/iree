// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_CONVERTTENSORTOFLOW_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_CONVERTTENSORTOFLOW_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Adds patterns for Tensor->Flow, for running before dispatch region formation.
void populateTensorToFlowPatternsBeforeDispatchFormation(
    MLIRContext *context, OwningRewritePatternList &patterns);

// Adds patterns for Tensor->Flow, for running after dispatch region formation.
void populateTensorToFlowPatternsAfterDispatchFormation(
    MLIRContext *context, OwningRewritePatternList &patterns);

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_CONVERTTENSORTOFLOW_H_
