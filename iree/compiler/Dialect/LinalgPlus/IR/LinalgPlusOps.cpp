// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusOps.h"

#include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_plus {

void SortOp::build(OpBuilder &builder, OperationState &state,
                   ValueRange operands, int64_t dimension) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));

  for (Value operand : operands) state.addTypes(operand.getType());

  state.addRegion();
}

static LogicalResult verifySortOp(SortOp op) {
  Operation::operand_range operands = op.operands();
  if (operands.empty()) return op.emitOpError("requires at least one input");

  if (llvm::all_of(operands, [](Value operand) {
        return operand.getType().cast<ShapedType>().hasRank();
      })) {
    ArrayRef<int64_t> inputShape =
        (*operands.begin()).getType().cast<ShapedType>().getShape();

    if (llvm::any_of(llvm::drop_begin(operands, 1), [&](Value operand) {
          return operand.getType().cast<ShapedType>().getShape() != inputShape;
        }))
      return op.emitOpError("requires all inputs to have the same dimensions");

    int64_t rank = inputShape.size();
    int64_t cmpDim = op.dimension();
    if (cmpDim < 0 || cmpDim >= rank)
      return op.emitOpError("dimension attribute value must be in range [0, ")
             << rank << "), but found " << cmpDim;
  }

  Block &block = op.comparator().front();
  size_t numOperands = op.getOperation()->getNumOperands();
  if (block.getNumArguments() != 2 * numOperands)
    return op.emitOpError("comparator block should have ")
           << 2 * numOperands << " arguments";

  for (auto indexedOperand : llvm::enumerate(operands)) {
    int index = indexedOperand.index();
    Type elemType =
        indexedOperand.value().getType().cast<ShapedType>().getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType)
        return op.emitOpError("comparator block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
    }
  }
  return success();
}

static LogicalResult verifyYieldOp(linalg_plus::YieldOp op) {
  auto *parentOp = op->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty()) {
    return op.emitOpError("expected single non-empty parent region");
  }
  if (parentOp->getDialect() !=
      parentOp->getContext()->getLoadedDialect<LinalgPlusDialect>()) {
    return op.emitOpError("expected parent op to be linalg_plus op");
  }

  return success();
}

}  // namespace linalg_plus
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgPlus/IR/LinalgPlusOps.cpp.inc"
