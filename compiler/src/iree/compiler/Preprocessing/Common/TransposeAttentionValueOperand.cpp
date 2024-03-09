
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::Preprocessing {

using IREE::LinalgExt::AttentionOp;

namespace {

struct TransposeAttentionValueOperandPass
    : public TransposeAttentionValueOperandBase<
          TransposeAttentionValueOperandPass> {
  void runOnOperation() override;
};
} // namespace

void TransposeAttentionValueOperandPass::runOnOperation() {
  // Find all attention ops.
  Operation *op = getOperation();
  SmallVector<AttentionOp> attentionOps;

  op->walk([&](AttentionOp attnOp) {
    // Skip if already transposed.
    if (attnOp.getTransposeV()) {
      return;
    }

    attentionOps.push_back(attnOp);
  });

  IRRewriter rewriter(op->getContext());
  for (AttentionOp attnOp : attentionOps) {
    Location loc = attnOp.getLoc();
    Value valueOp = attnOp.getValue();
    rewriter.setInsertionPoint(attnOp);

    auto valueTensor = dyn_cast<RankedTensorType>(valueOp.getType());
    if (!valueTensor) {
      continue;
    }

    // TODO: Add support for dynamic dimensions.
    if (valueTensor.hasStaticShape()) {
      continue;
    }

    SmallVector<int64_t> transposedShape(valueTensor.getShape());
    int64_t rank = valueTensor.getRank();
    std::swap(transposedShape[rank - 1], transposedShape[rank - 2]);

    // Transpose the N_CTX and the HEAD_DIM dimension.
    // TODO: Ideally, this should be better captured in the op. For now, it
    // is the last two dims.
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, transposedShape, valueTensor.getElementType());

    SmallVector<int64_t> permutation = {rank - 1, rank - 2};
    Value transposedV =
        rewriter
            .create<linalg::TransposeOp>(loc, valueOp, emptyTensor, permutation)
            ->getResult(0);

    rewriter.replaceOpWithNewOp<AttentionOp>(
        attnOp, attnOp.getResultTypes(),
        ValueRange({attnOp.getQuery(), attnOp.getKey(), transposedV,
                    attnOp.getScale()}),
        attnOp.getResults(), /*transpose_v=*/true);
  }
}

std::unique_ptr<Pass> createTransposeAttentionValueOperand() {
  return std::make_unique<TransposeAttentionValueOperandPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
