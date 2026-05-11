// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "llvm/ADT/Repeated.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler {

Value mkIntConst(OpBuilder &builder, Location loc, int64_t v) {
  MLIRContext *ctx = builder.getContext();
  return smt::IntConstantOp::create(
      builder, loc, smt::IntType::get(ctx),
      builder.getIntegerAttr(IntegerType::get(ctx, 64), v));
}

Value mkKnob(OpBuilder &builder, Location loc, StringRef name) {
  MLIRContext *ctx = builder.getContext();
  return IREE::Codegen::KnobOp::create(builder, loc, smt::IntType::get(ctx),
                                       StringAttr::get(ctx, name));
}

/// For a given loop dimension index, find an operand and its axis that maps
/// to that dimension using per-operand indexing maps.
/// Returns {operandIdx, axisIdx} or {-1, -1} if not found.
static std::pair<int, int>
findOperandAndAxisForDim(ArrayRef<AffineMap> indexingMaps, unsigned dimIdx) {
  for (auto [operandIdx, map] : llvm::enumerate(indexingMaps)) {
    for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
      if (dimExpr && dimExpr.getPosition() == dimIdx) {
        return {static_cast<int>(operandIdx), static_cast<int>(i)};
      }
    }
  }
  return {-1, -1};
}

/// Emit common SMT constraints for known-constant dim values.
// TODO(#23535): Also emit constraints from assume.int umin/umax/udiv.
static void emitCommonConstraints(OpBuilder &builder, Location loc,
                                  Block *block, ArrayRef<Value> dimValues) {
  for (auto [d, dimVal] : llvm::enumerate(dimValues)) {
    IntegerAttr constAttr;
    if (!matchPattern(dimVal, m_Constant(&constAttr))) {
      continue;
    }

    Value dimArg = block->getArgument(d);
    int64_t staticSize = constAttr.getInt();
    Value constVal = mkIntConst(builder, loc, staticSize);
    Value eq = smt::EqOp::create(builder, loc, dimArg, constVal);
    IREE::Codegen::AssertOp::create(
        builder, loc, eq,
        ("dim_" + Twine(d) + " ({}) == " + Twine(staticSize)).str(),
        ValueRange{dimArg});
  }
}

ConstraintsOpShell createConstraintsOpShell(
    OpBuilder &builder, Operation *rootOp, IREE::Codegen::RootOpAttr rootOpAttr,
    IREE::Codegen::PipelineAttrInterface pipelineAttr, DictionaryAttr knobs,
    unsigned numLoops, ArrayRef<AffineMap> indexingMaps) {
  MLIRContext *ctx = rootOp->getContext();
  Location loc = rootOp->getLoc();
  builder.setInsertionPointAfter(rootOp);

  // Extract dim values for each loop dimension.
  SmallVector<Value> dimValues;
  for (unsigned d = 0; d < numLoops; ++d) {
    auto [operandIdx, axisIdx] = findOperandAndAxisForDim(indexingMaps, d);
    assert(operandIdx >= 0 && "every loop dim must map to some operand");
    Value operand = rootOp->getOperand(operandIdx);
    Value dimVal = linalg::createOrFoldDimOp(builder, loc, operand, axisIdx);
    dimValues.push_back(dimVal);
  }

  // Create the constraints op.
  smt::IntType smtIntTy = smt::IntType::get(ctx);
  llvm::Repeated<Type> blockArgTypes(numLoops, smtIntTy);

  auto constraintsOp = IREE::Codegen::ConstraintsOp::create(
      builder, loc, rootOpAttr, pipelineAttr, knobs, dimValues);
  Region &body = constraintsOp.getBody();
  Block *block = builder.createBlock(&body, body.end(), blockArgTypes,
                                     SmallVector<Location>(numLoops, loc));
  builder.setInsertionPointToStart(block);

  // Emit common constraints (static dims).
  emitCommonConstraints(builder, loc, block, dimValues);

  // Build result.
  ConstraintsOpShell shell;
  shell.op = constraintsOp;
  for (unsigned d = 0; d < numLoops; ++d) {
    shell.smtDimArgs.push_back(block->getArgument(d));
  }
  return shell;
}

} // namespace mlir::iree_compiler
