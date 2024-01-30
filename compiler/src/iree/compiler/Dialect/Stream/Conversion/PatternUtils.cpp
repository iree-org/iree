// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"

namespace mlir::iree_compiler {

ConvertedTensor consumeTensorOperand(Location loc, Value operand,
                                     OpBuilder &builder) {
  auto operandType = operand.getType();
  if (llvm::isa<IREE::Stream::ResourceType>(operandType)) {
    // Prior to https://reviews.llvm.org/D111620 this is the path we'd take;
    // the tensor operands would be remapped into their new resource types.
    // This is still possible during rewriting if we ourselves produce a new
    // resource type, but the automatic materialization will go down the
    // unrealized_conversion_cast path below.
    return {
        operand,
        builder.createOrFold<IREE::Stream::ResourceSizeOp>(
            loc, builder.getIndexType(), operand),
    };
  } else if (auto castOp =
                 operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    // We only have a single tensor type conversion and it expands to (resource,
    // size) so that's all we look for here.
    assert(castOp.getNumOperands() == 2 && "expected (resource, size)");
    return {
        castOp.getOperand(0),
        castOp.getOperand(1),
    };
  }
  assert(false &&
         "unexpected operand; expected either a IREE::Stream::ResourceType or "
         "the result of a mlir::UnrealizedConversionCastOp");
  return ConvertedTensor();
}

} // namespace mlir::iree_compiler
