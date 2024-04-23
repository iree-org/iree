// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PREPROCESSING_COMMON_PDLUTILITIES_H_
#define IREE_COMPILER_PREPROCESSING_COMMON_PDLUTILITIES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {
LogicalResult rewriteAsFlowDispatch(PatternRewriter &rewriter,
                                    Operation *rootOp, Attribute externalFnName,
                                    ValueRange inputOperands,
                                    ValueRange replacedValues,
                                    ValueRange replacedValuesShapes,
                                    ValueRange otherOperands);

LogicalResult checkTensorElementType(PatternRewriter &rewriter,
                                     Type operandType, Type elementType);
} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_PREPROCESSING_COMMON_PDLUTILITIES_H_
