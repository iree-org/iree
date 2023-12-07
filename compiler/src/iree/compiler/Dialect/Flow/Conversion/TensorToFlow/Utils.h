// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_UTILS_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_UTILS_H_

#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Location;
class LogicalResult;
class OpBuilder;
class RewriterBase;
class Value;
namespace tensor {
class ExtractSliceOp;
class InsertSliceOp;
} // namespace tensor
} // namespace mlir

namespace mlir::iree_compiler::IREE::Flow {

/// Rewrite the given InsertSliceOp into a Flow::TensorUpdateOp.
LogicalResult
convertInsertSliceOpToFlowUpdateOp(RewriterBase &rewriter,
                                   tensor::InsertSliceOp insertOp);

/// Rewrite the given ExtractSliceOp into a Flow::TensorSliceOp.
LogicalResult
convertExtractSliceOpToFlowSliceOp(RewriterBase &rewriter,
                                   tensor::ExtractSliceOp sliceOp);

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_UTILS_H_
