// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

std::optional<CastOpInterface> getDefiningCastOp(Value input) {
  auto castOp = input.getDefiningOp<CastOpInterface>();
  if (castOp) {
    return castOp;
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
  return castOp;
}

std::optional<Type> getCastElemType(Value input) {
  std::optional<CastOpInterface> castOp = getDefiningCastOp(input);
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

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
