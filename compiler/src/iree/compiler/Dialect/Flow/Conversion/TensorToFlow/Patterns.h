// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_PATTERNS_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_PATTERNS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir::iree_compiler::IREE::Flow {

// Adds patterns for Tensor->Flow, for running before dispatch region formation.
void populateTensorToFlowConversionPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_PATTERNS_H_
