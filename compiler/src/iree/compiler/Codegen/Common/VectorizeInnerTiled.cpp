// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

namespace {

// Lifts a tensor-semantics `iree_codegen.inner_tiled` op to vector semantics by
// reading each operand with `vector.transfer_read`, replacing the op with a
// vector-semantics `inner_tiled`, and writing each result back with
// `vector.transfer_write`. The downstream `DropInnerTiledUnitDimsPattern` and
// `LowerInnerTiledPattern` then handle the rest of the lowering.
//
// Used by the LLVM-CPU pipeline: GPU folds the read/write into its lane
// distribution (`DistributeInnerTiledToLanes`), but CPU has no lane
// distribution so the vectorize step is standalone.
struct VectorizeInnerTiledPattern final
    : OpRewritePattern<IREE::Codegen::InnerTiledOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Codegen::InnerTiledOp tiledOp,
                                PatternRewriter &rewriter) const override {
    if (!tiledOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(tiledOp, "expected tensor semantics");
    }
    // Statically-shaped operands only — `vector.transfer_read` of a dynamic
    // tensor would need explicit bounds and we don't have them here. For the
    // CPU codegen path, the tiling pipeline drives the iteration domain to
    // unit bounds well before this pattern runs, so all operand shapes are
    // static.
    for (Value operand : tiledOp->getOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType || !tensorType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(tiledOp, "non-static operand shape");
      }
    }

    Location loc = tiledOp.getLoc();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);

    // Read each operand into a vector matching the operand's tensor shape.
    SmallVector<Value> vectorOperands;
    vectorOperands.reserve(tiledOp->getNumOperands());
    for (Value operand : tiledOp->getOperands()) {
      auto tensorType = cast<RankedTensorType>(operand.getType());
      auto vectorType =
          VectorType::get(tensorType.getShape(), tensorType.getElementType());
      Value padding = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(tensorType.getElementType()));
      SmallVector<Value> indices(tensorType.getRank(), zeroIdx);
      Value read = vector::TransferReadOp::create(rewriter, loc, vectorType,
                                                  operand, indices, padding);
      vectorOperands.push_back(read);
    }

    int64_t numInputs = tiledOp.getNumInputs();
    auto newTiledOp = IREE::Codegen::InnerTiledOp::create(
        rewriter, loc,
        ValueRange(ArrayRef(vectorOperands).take_front(numInputs)),
        ValueRange(ArrayRef(vectorOperands).drop_front(numInputs)),
        tiledOp.getIndexingMapsAttr(), tiledOp.getIteratorTypesAttr(),
        tiledOp.getKind(), tiledOp.getSemantics(),
        tiledOp.getPermutationsAttr());

    // Write each result back into the original init tensor.
    SmallVector<Value> newResults;
    newResults.reserve(tiledOp.getNumResults());
    for (auto [vectorResult, tensorInit] :
         llvm::zip_equal(newTiledOp.getResults(), tiledOp.getOutputs())) {
      auto tensorType = cast<RankedTensorType>(tensorInit.getType());
      SmallVector<Value> indices(tensorType.getRank(), zeroIdx);
      auto writeOp = vector::TransferWriteOp::create(
          rewriter, loc, vectorResult, tensorInit, indices);
      newResults.push_back(writeOp.getResult());
    }
    rewriter.replaceOp(tiledOp, newResults);
    return success();
  }
};

} // namespace

void populateVectorizeInnerTiledPatterns(RewritePatternSet &patterns) {
  patterns.add<VectorizeInnerTiledPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler
