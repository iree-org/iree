// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

FailureOr<FusionResult> LinalgExtFusionPattern::returningMatchAndRewrite(
    TilingInterface consumerOp, PatternRewriter &rewriter) const {
  // Try to fuse the producers of all operands to fuse.
  SmallVector<TilingInterface> fusedOps;
  for (int64_t operandToFuse : operandsToFuse) {
    // Check the operand exists.
    if (operandToFuse >= consumerOp->getNumOperands())
      return failure();

    // Check the operand is a slice of a producer result.
    auto sliceOp = consumerOp->getOperand(operandToFuse)
                       .getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      return failure();
    auto producerOp = sliceOp.getSource().getDefiningOp<TilingInterface>();
    if (!producerOp || producerOp->getNumResults() != 1)
      return failure();

    // Tile the producer.
    FailureOr<TilingResult> tileAndFuseResult =
        producerOp.generateResultTileValue(rewriter, /*resultNumber=*/0,
                                           sliceOp.getMixedOffsets(),
                                           sliceOp.getMixedSizes());
    if (failed(tileAndFuseResult))
      return failure();
    for (auto tileAndFusedOp : tileAndFuseResult->tiledOps) {
      auto interfaceOp = dyn_cast<TilingInterface>(tileAndFusedOp);
      if (!interfaceOp)
        continue;
      fusedOps.push_back(interfaceOp);
    }
  }

  // Update the consumer in-place using the tiled producer results.
  SmallVector<Value> newOperands = consumerOp->getOperands();
  for (auto it : llvm::zip(operandsToFuse, fusedOps)) {
    int64_t operandToFuse = std::get<0>(it);
    TilingInterface fusedOp = std::get<1>(it);
    newOperands[operandToFuse] = fusedOp->getResult(0);
  }
  rewriter.updateRootInPlace(consumerOp,
                             [&]() { consumerOp->setOperands(newOperands); });

  return FusionResult{consumerOp, fusedOps};
}
