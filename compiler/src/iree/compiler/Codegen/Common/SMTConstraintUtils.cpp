// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/SMTConstraintUtils.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

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

/// Try to resolve a tensor.dim op to the underlying index value by using
/// ReifyRankedShapedTypeOpInterface on the source tensor's defining op.
static Value resolveTensorDim(tensor::DimOp dimOp) {
  std::optional<int64_t> constIndex = dimOp.getConstantIndex();
  if (!constIndex) {
    return nullptr;
  }

  Value source = dimOp.getSource();
  Operation *defOp = source.getDefiningOp();
  if (!defOp) {
    return nullptr;
  }

  auto reifiable = dyn_cast<ReifyRankedShapedTypeOpInterface>(defOp);
  if (!reifiable) {
    return nullptr;
  }

  OpBuilder builder(dimOp);
  ReifiedRankedShapedTypeDims reifiedShapes;
  if (failed(reifiable.reifyResultShapes(builder, reifiedShapes))) {
    return nullptr;
  }

  unsigned resultIdx = cast<OpResult>(source).getResultNumber();
  if (resultIdx >= reifiedShapes.size()) {
    return nullptr;
  }

  SmallVector<OpFoldResult> &dims = reifiedShapes[resultIdx];
  unsigned dimIdx = *constIndex;
  if (dimIdx >= dims.size()) {
    return nullptr;
  }

  if (auto val = dyn_cast<Value>(dims[dimIdx])) {
    return val;
  }

  return nullptr;
}

/// Walk the SSA chain from a value to find a util.assume.int op.
static std::pair<IREE::Util::AssumeIntOp, unsigned> findAssumeIntOp(Value val) {
  while (val) {
    if (auto assumeOp = val.getDefiningOp<IREE::Util::AssumeIntOp>()) {
      unsigned idx = cast<OpResult>(val).getResultNumber();
      return {assumeOp, idx};
    }
    Operation *defOp = val.getDefiningOp();
    if (!defOp) {
      break;
    }
    if (auto dimOp = dyn_cast<tensor::DimOp>(defOp)) {
      Value resolved = resolveTensorDim(dimOp);
      if (resolved) {
        val = resolved;
        continue;
      }
      break;
    }
    if (defOp->getNumOperands() == 1 && defOp->getNumResults() == 1) {
      val = defOp->getOperand(0);
      continue;
    }
    break;
  }
  return {nullptr, 0};
}

/// Emit common SMT constraints (static dim values, assume.int bounds).
static void emitCommonConstraints(OpBuilder &builder, Location loc,
                                  Block *block, unsigned numLoops,
                                  ArrayRef<int64_t> staticLoopRanges,
                                  ArrayRef<Value> dimValues) {
  Value zero = mkIntConst(builder, loc, 0);

  for (unsigned d = 0; d < numLoops; ++d) {
    Value dimArg = block->getArgument(d);
    int64_t staticSize = staticLoopRanges[d];
    std::string dimName = llvm::formatv("dim_{}", d);

    if (!ShapedType::isDynamic(staticSize)) {
      Value constVal = mkIntConst(builder, loc, staticSize);
      Value eq = smt::EqOp::create(builder, loc, dimArg, constVal);
      IREE::Codegen::AssertOp::create(
          builder, loc, eq, dimName + " ({}) == " + std::to_string(staticSize),
          ValueRange{dimArg});
      continue;
    }

    Value dimVal = dimValues[d];
    auto [assumeOp, resultIdx] = findAssumeIntOp(dimVal);
    if (!assumeOp) {
      continue;
    }

    SmallVector<IREE::Util::IntAssumptionAttr> assumptions =
        assumeOp.getOperandAssumptions(resultIdx);
    for (IREE::Util::IntAssumptionAttr assumption : assumptions) {
      if (std::optional<int64_t> umin = assumption.getUmin()) {
        Value uminVal = mkIntConst(builder, loc, *umin);
        Value cmp = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge,
                                          dimArg, uminVal);
        IREE::Codegen::AssertOp::create(
            builder, loc, cmp, dimName + " ({}) >= " + std::to_string(*umin),
            ValueRange{dimArg});
      }
      if (std::optional<int64_t> umax = assumption.getUmax()) {
        Value umaxVal = mkIntConst(builder, loc, *umax);
        Value cmp = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le,
                                          dimArg, umaxVal);
        IREE::Codegen::AssertOp::create(
            builder, loc, cmp, dimName + " ({}) <= " + std::to_string(*umax),
            ValueRange{dimArg});
      }
      if (std::optional<int64_t> udiv = assumption.getUdiv()) {
        Value udivVal = mkIntConst(builder, loc, *udiv);
        Value rem = smt::IntModOp::create(builder, loc, dimArg, udivVal);
        Value eq = smt::EqOp::create(builder, loc, rem, zero);
        IREE::Codegen::AssertOp::create(builder, loc, eq,
                                        dimName + " ({}) divisible by " +
                                            std::to_string(*udiv),
                                        ValueRange{dimArg});
      }
    }
  }
}

ConstraintsOpShell createConstraintsOpShell(
    OpBuilder &builder, Operation *rootOp, IREE::Codegen::RootOpAttr rootOpAttr,
    Attribute pipelineAttr, DictionaryAttr knobs, unsigned numLoops,
    ArrayRef<int64_t> staticLoopRanges, ArrayRef<AffineMap> indexingMaps) {
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
  SmallVector<Type> blockArgTypes(numLoops, smtIntTy);

  auto constraintsOp = IREE::Codegen::ConstraintsOp::create(
      builder, loc, rootOpAttr, pipelineAttr, knobs, dimValues);
  Region &body = constraintsOp.getBody();
  Block *block = builder.createBlock(&body, body.end(), blockArgTypes,
                                     SmallVector<Location>(numLoops, loc));
  builder.setInsertionPointToStart(block);

  // Emit common constraints (static dims, assume.int).
  emitCommonConstraints(builder, loc, block, numLoops, staticLoopRanges,
                        dimValues);

  // Build result.
  ConstraintsOpShell shell;
  shell.op = constraintsOp;
  for (unsigned d = 0; d < numLoops; ++d) {
    shell.smtDimArgs.push_back(block->getArgument(d));
  }
  return shell;
}

} // namespace mlir::iree_compiler
