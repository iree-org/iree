// Copyright 2022 The IREE Authors
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

// If the producer is a CastOpInterface, or a linalg::GenericOp that performs
// only a CastOpInterface on its input, return the CastOpInterface op.
//
// **Note: If the CastOpInterface has been generalized, the return Operation
//         is the body CastOpInterface op, not the linalg::GenericOp.
Operation *getDefiningCastOp(Value input) {
  Operation *op = input.getDefiningOp();
  if (!op) {
    return nullptr;
  }
  auto castOp = dyn_cast<CastOpInterface>(op);
  if (castOp) {
    return op;
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getBody()->getOperations().size() != 2 ||
      !isElementwise(genericOp)) {
    return nullptr;
  }
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return nullptr;
  }
  Value castIn = castOp->getOperand(0);
  if (castIn.isa<BlockArgument>() &&
      castIn.cast<BlockArgument>().getArgNumber() != 0) {
    return nullptr;
  }
  return castOp;
}

// Returns an IntegerType with the specified bitwidth and signedness.
IntegerType getIntegerTypeWithSignedness(MLIRContext *ctx, int bitWidth,
                                         bool isSigned) {
  return IntegerType::get(ctx, bitWidth,
                          isSigned
                              ? IntegerType::SignednessSemantics::Signed
                              : IntegerType::SignednessSemantics::Unsigned);
}

// Returns the source element type of the defining CastOpInterface of `input`,
// if there is one.
Type getCastElemType(Value input) {
  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(input.getType())) {
    Operation *castOp = getDefiningCastOp(input);
    if (!castOp) {
      return {};
    }
    Type castSrcElemType = castOp->getOperand(0).getType();
    if (auto castTensorType = dyn_cast<RankedTensorType>(castSrcElemType)) {
      castSrcElemType = castTensorType.getElementType();
    }
    if (isa<arith::ExtUIOp>(castOp)) {
      int64_t bitWidth = castSrcElemType.getIntOrFloatBitWidth();
      castSrcElemType =
          getIntegerTypeWithSignedness(castOp->getContext(), bitWidth, false);
    }
    return castSrcElemType;
  }
  return {};
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
