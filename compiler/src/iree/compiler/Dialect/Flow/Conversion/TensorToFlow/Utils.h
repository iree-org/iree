// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_UTILS_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Flow {

/// Indicates whether the given offsets/sizes/strides representing a slice from
/// baseShape is a contiguous slice, and this is mappable to Flow ops.
bool isOffsetSizeAndStrideMappableToFlow(llvm::ArrayRef<OpFoldResult> offsets,
                                         llvm::ArrayRef<OpFoldResult> sizes,
                                         llvm::ArrayRef<OpFoldResult> strides,
                                         llvm::ArrayRef<int64_t> baseShape);

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
