// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::GlobalOptimization {

std::optional<CastOpInterface> getDefiningNonI1ExtendingCastOp(Value input) {
  // Returns true if the source of cast op has i1 element type.
  auto isI1Src = [](CastOpInterface castOp) -> bool {
    auto elemType = getElementTypeOrSelf(castOp->getOperandTypes()[0]);
    return elemType.getIntOrFloatBitWidth() == 1;
  };

  auto isExtending = [](CastOpInterface castOp) -> bool {
    auto inElemType = getElementTypeOrSelf(castOp->getOperandTypes()[0]);
    auto outElemType = getElementTypeOrSelf(castOp->getResultTypes()[0]);
    return inElemType.getIntOrFloatBitWidth() <
           outElemType.getIntOrFloatBitWidth();
  };

  std::optional<CastOpInterface> result = std::nullopt;
  auto castOp = input.getDefiningOp<CastOpInterface>();
  if (castOp) {
    if (!isI1Src(castOp) && isExtending(castOp)) {
      result = castOp;
    }
    return result;
  }
  auto genericOp = input.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getBody()->getOperations().size() != 2 ||
      !isElementwise(genericOp)) {
    return std::nullopt;
  }
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return std::nullopt;
  }
  Value castIn = castOp->getOperand(0);
  if (isa<BlockArgument>(castIn) &&
      cast<BlockArgument>(castIn).getArgNumber() != 0) {
    return std::nullopt;
  }
  if (!isI1Src(castOp) && isExtending(castOp)) {
    result = castOp;
  }
  return result;
}

std::optional<Type> getCastElemType(Value input) {
  std::optional<CastOpInterface> castOp =
      getDefiningNonI1ExtendingCastOp(input);
  if (!castOp) {
    return std::nullopt;
  }
  Type castSrcElemType = getElementTypeOrSelf(castOp.value()->getOperand(0));
  if (isa<arith::ExtUIOp>(castOp.value())) {
    int64_t bitWidth = castSrcElemType.getIntOrFloatBitWidth();
    return IntegerType::get(castOp->getContext(), bitWidth,
                            IntegerType::SignednessSemantics::Unsigned);
  }
  return castSrcElemType;
}

Value createGenericElementwiseCastOp(
    OpBuilder &builder, Location loc, Value input, CastOpInterface castOp,
    ArrayRef<NamedAttribute> attrs,
    std::optional<IREE::Encoding::EncodingAttr> encoding) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<AffineMap> maps(
      2, AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                           builder.getContext()));
  SmallVector<utils::IteratorType> iteratorTypes(inputType.getRank(),
                                                 utils::IteratorType::parallel);
  auto elementType = getElementTypeOrSelf(castOp->getResultTypes()[0]);
  auto castedType = inputType.clone(elementType);
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(builder, loc, input);
  Value init =
      encoding
          ? builder.create<tensor::EmptyOp>(loc, inputMixedSizes, elementType,
                                            *encoding)
          : builder.create<tensor::EmptyOp>(loc, inputMixedSizes, elementType);
  return builder
      .create<linalg::GenericOp>(
          loc, castedType, input, init, maps, iteratorTypes,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value castRes =
                b.create(nestedLoc, castOp->getName().getIdentifier(), args[0],
                         elementType)
                    ->getResult(0);
            b.create<linalg::YieldOp>(nestedLoc, castRes);
          },
          attrs)
      .getResult(0);
}

FailureOr<IREE::Flow::DispatchRegionOp>
wrapConsecutiveOpsInDispatchRegion(RewriterBase &rewriter,
                                   SmallVector<Operation *> ops) {
  FailureOr<IREE::Flow::DispatchRegionOp> maybeRegionOp =
      IREE::Flow::wrapOpInDispatchRegion(rewriter, ops.back());
  if (failed(maybeRegionOp)) {
    return failure();
  }
  IREE::Flow::DispatchRegionOp regionOp = maybeRegionOp.value();

  SmallVector<Operation *> precedingOps(ops.begin(), ops.end() - 1);
  return IREE::Flow::movePrecedingOpsIntoDispatchRegion(rewriter, precedingOps,
                                                        regionOp);
}

Value sumReduceDimensionSubset(ImplicitLocOpBuilder &rewriter, Value val,
                               Type accETy, ArrayRef<bool> is_reduction) {
  auto context = val.getContext();
  RankedTensorType ty = llvm::cast<RankedTensorType>(val.getType());

  llvm::SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynSizes;
  for (int i = 0, s = is_reduction.size(); i < s; i++) {
    if (is_reduction[i])
      continue;

    staticSizes.push_back(ty.getDimSize(i));
    if (ty.isDynamicDim(i)) {
      dynSizes.push_back(rewriter.create<tensor::DimOp>(val, i));
    }
  }

  // Create a zero-filled accumulator.
  Value initAcc =
      rewriter.create<tensor::EmptyOp>(staticSizes, accETy, dynSizes);
  Value zeroInt = rewriter.create<arith::ConstantIntOp>(0, accETy).getResult();
  Value zeroAcc =
      rewriter.create<linalg::FillOp>(zeroInt, initAcc).getResult(0);

  SmallVector<AffineExpr> filterExprs(ty.getRank());
  SmallVector<AffineExpr> outputExprs;
  SmallVector<utils::IteratorType> iterators;

  for (int i = 0, s = ty.getRank(); i < s; i++) {
    if (!is_reduction[i]) {
      auto expr = rewriter.getAffineDimExpr(iterators.size());
      iterators.push_back(utils::IteratorType::parallel);

      outputExprs.push_back(expr);
      filterExprs[i] = expr;
    }
  }

  for (int i = 0, s = filterExprs.size(); i < s; i++) {
    if (!filterExprs[i]) {
      auto expr = mlir::getAffineDimExpr(iterators.size(), context);
      iterators.push_back(utils::IteratorType::reduction);
      filterExprs[i] = expr;
    }
  }

  SmallVector<AffineMap> affineMaps{
      AffineMap::get(ty.getRank(), 0, filterExprs, context),
      AffineMap::get(ty.getRank(), 0, outputExprs, context)};

  return rewriter
      .create<linalg::GenericOp>(
          zeroAcc.getType(), ValueRange{val}, ValueRange{zeroAcc}, affineMaps,
          iterators,
          [=](OpBuilder &b, Location loc, ValueRange args) {
            Value ext = b.create<arith::ExtSIOp>(loc, accETy, args[0]);
            Value sum = b.create<arith::AddIOp>(loc, ext, args[1]);
            b.create<linalg::YieldOp>(loc, sum);
          })
      .getResult(0);
}

} // namespace mlir::iree_compiler::GlobalOptimization
