// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

std::optional<CastOpInterface> getDefiningNonI1CastOp(Value input) {
  // Returns true if the source of cast op has i1 element type.
  auto isI1Src = [](CastOpInterface castOp) -> bool {
    auto elemType = getElementTypeOrSelf(castOp->getOperandTypes()[0]);
    return elemType.getIntOrFloatBitWidth() == 1;
  };

  std::optional<CastOpInterface> result = std::nullopt;
  auto castOp = input.getDefiningOp<CastOpInterface>();
  if (castOp) {
    if (!isI1Src(castOp)) {
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
  if (castIn.isa<BlockArgument>() &&
      castIn.cast<BlockArgument>().getArgNumber() != 0) {
    return std::nullopt;
  }
  if (!isI1Src(castOp)) {
    result = castOp;
  }
  return result;
}

std::optional<Type> getCastElemType(Value input) {
  std::optional<CastOpInterface> castOp = getDefiningNonI1CastOp(input);
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
    std::optional<IREE::LinalgExt::EncodingAttr> encoding) {
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

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
