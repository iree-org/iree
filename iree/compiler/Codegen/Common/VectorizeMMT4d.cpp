// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

Value promoteVector(Location loc, Value inputVector, Type promotedElementType,
                    PatternRewriter &rewriter) {
  VectorType inputVectorType = inputVector.getType().cast<VectorType>();
  if (inputVectorType.getElementType() == promotedElementType) {
    return inputVector;
  } else {
    auto promotedVectorType = inputVectorType.clone(promotedElementType);
    if (promotedElementType.isIntOrIndex()) {
      return rewriter.create<arith::ExtSIOp>(loc, inputVector,
                                             promotedVectorType);
    } else {
      return rewriter.create<arith::ExtFOp>(loc, inputVector,
                                            promotedVectorType);
    }
  }
}

/// Converts linalg.mmt4d into vector.contract.
/// This converts linalg.mmt4d with operands <1x1xM0xK0>, <1x1xN0xK0>
/// to vector.contract where K0 is the contraction dimension.
struct VectorizeMMT4DOp : public OpRewritePattern<linalg::Mmt4DOp> {
  using OpRewritePattern<linalg::Mmt4DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Mmt4DOp mmt4DOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = mmt4DOp.inputs()[0];
    Value rhs = mmt4DOp.inputs()[1];
    Value dst = mmt4DOp.outputs()[0];

    ShapedType lhsType = lhs.getType().dyn_cast<ShapedType>();
    ShapedType rhsType = rhs.getType().dyn_cast<ShapedType>();
    ShapedType dstType = dst.getType().dyn_cast<ShapedType>();

    // This pattern expects tensors of static shapes.
    // In practice, dynamic shapes are meant to be handled by other passes,
    // ahead of this point. Dynamic outer dimensions (?x?xM0xK0) should be
    // handled by a tiling pass typically running just ahead of the present
    // pass. Dynamic inner dimensions (M1xK1x?x?) mean that the IR is not yet
    // specialized to a specific SIMD ISA, and should be handled by dispatching
    // to specialized code paths where these inner dimensions become static
    // (M1xK1x?x? --> M1xK1xM0xK0)
    if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape())
      return failure();

    // We expect the incoming mmt4d to already have been maximally tiled, so
    // that the outer dimensions are equal to 1.
    {
      int M1 = lhsType.getShape()[0];
      int K1 = lhsType.getShape()[1];
      int N1 = rhsType.getShape()[0];
      if (M1 != 1 || K1 != 1 || N1 != 1) return failure();
    }

    // Read the inner dimensions.
    int M0 = lhsType.getShape()[2];
    int N0 = rhsType.getShape()[2];
    int K0 = lhsType.getShape()[3];

    Location loc = mmt4DOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    Type lhsElementType = lhsType.getElementType();
    Type rhsElementType = rhsType.getElementType();
    Type dstElementType = dstType.getElementType();

    auto lhsVecType = VectorType::get({1, 1, M0, K0}, lhsElementType);
    auto rhsVecType = VectorType::get({1, 1, N0, K0}, rhsElementType);
    auto dstVecType = VectorType::get({1, 1, M0, N0}, dstElementType);

    auto lhsVecType2D = VectorType::get({M0, K0}, lhsElementType);
    auto rhsVecType2D = VectorType::get({N0, K0}, rhsElementType);
    auto dstVecType2D = VectorType::get({M0, N0}, dstElementType);

    auto identityMap = rewriter.getMultiDimIdentityMap(4);

    // Read the input tensors into vectors.
    auto lhsVec = rewriter.create<vector::TransferReadOp>(
        loc, lhsVecType, lhs, ValueRange{c0, c0, c0, c0}, identityMap);
    auto rhsVec = rewriter.create<vector::TransferReadOp>(
        loc, rhsVecType, rhs, ValueRange{c0, c0, c0, c0}, identityMap);
    auto dstVec = rewriter.create<vector::TransferReadOp>(
        loc, dstVecType, dst, ValueRange{c0, c0, c0, c0}, identityMap);

    // Convert the input vectors from 4D shapes (1x1xM0xK0) to 2D shapes (M0xK0)
    Value lhsVec2D =
        rewriter.create<vector::ShapeCastOp>(loc, lhsVecType2D, lhsVec);
    Value rhsVec2D =
        rewriter.create<vector::ShapeCastOp>(loc, rhsVecType2D, rhsVec);
    Value dstVec2D =
        rewriter.create<vector::ShapeCastOp>(loc, dstVecType2D, dstVec);

    // Promote, if needed, the element type in the lhs and rhs vectors to
    // match the dst vector, so that the vector.contract below will involve
    // only one element type. This is in line with planned design, see
    // the closing comment on https://reviews.llvm.org/D112508 where the
    // alternative of using mixed types was considered.
    Value promLhsVec2d = promoteVector(loc, lhsVec2D, dstElementType, rewriter);
    Value promRhsVec2d = promoteVector(loc, rhsVec2D, dstElementType, rewriter);

    // Generate the vector.contract on 2D vectors replacing the mmt4d op.
    auto m = rewriter.getAffineDimExpr(0);
    auto n = rewriter.getAffineDimExpr(1);
    auto k = rewriter.getAffineDimExpr(2);
    auto map0 = AffineMap::get(3, 0, {m, k}, rewriter.getContext());
    auto map1 = AffineMap::get(3, 0, {n, k}, rewriter.getContext());
    auto map2 = AffineMap::get(3, 0, {m, n}, rewriter.getContext());
    ArrayAttr indexingMaps = rewriter.getAffineMapArrayAttr({map0, map1, map2});
    ArrayAttr iterators = rewriter.getStrArrayAttr(
        {getParallelIteratorTypeName(), getParallelIteratorTypeName(),
         getReductionIteratorTypeName()});
    Value contractResult = rewriter.create<vector::ContractionOp>(
        loc, promLhsVec2d, promRhsVec2d, dstVec2D, indexingMaps, iterators);

    // Convert the output vector from 2D shape (M0xN0) to 4D shape (1x1xM0xN0)
    Value contractResult4D =
        rewriter.create<vector::ShapeCastOp>(loc, dstVecType, contractResult);

    // Replace the mmt4d op by the equivalent graph.
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        mmt4DOp, contractResult4D, dst, ValueRange{c0, c0, c0, c0},
        identityMap);
    return success();
  }
};

struct LinalgToVectorVectorizeMMT4dPass
    : public LinalgToVectorVectorizeMMT4dBase<
          LinalgToVectorVectorizeMMT4dPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<VectorizeMMT4DOp>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

void populateLinalgToVectorVectorizeMMT4dPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<VectorizeMMT4DOp>(context);
}

std::unique_ptr<OperationPass<FuncOp>>
createLinalgToVectorVectorizeMMT4dPass() {
  return std::make_unique<LinalgToVectorVectorizeMMT4dPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
