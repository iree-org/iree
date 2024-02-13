// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_CONVERSION_PATTERN_UTILS_H_
#define IREE_COMPILER_DIALECT_STREAM_CONVERSION_PATTERN_UTILS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void expandResourceOperand(Location loc, Value operand,
                           SmallVectorImpl<Value> &newOperands,
                           OpBuilder &builder);

SmallVector<Value> expandResourceOperands(Location loc, ValueRange operands,
                                          ConversionPatternRewriter &rewriter);

// https://reviews.llvm.org/D111620 broke 1->N type expansion during dialect
// conversion. It inserts unrealized_conversion_casts but then passes the
// illegal source dialect types for pattern operands, meaning that even though
// we say tensors are illegal the patterns get the new remapped values as
// tensors. This, naturally, breaks everything. To work around this we have this
// helper that tries to peek through the unrealized_conversion_casts and get out
// the actual values we expected to see from the conversion (and did before that
// change).
struct ConvertedTensor {
  Value resource;
  Value resourceSize;
};
ConvertedTensor consumeTensorOperand(Location loc, Value operand,
                                     OpBuilder &builder);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_STREAM_CONVERSION_PATTERN_UTILS_H_
