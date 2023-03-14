// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vectorize-pad"

// Note: A pull request is open to upstream this pattern:
//   https://reviews.llvm.org/D117021
// Once it lands, this pattern can be replaced.

namespace mlir {
namespace iree_compiler {

/// Gets the given `attrOrValue` as an index value by creating constant ops
/// for attributes.
static Value getAsIndexValue(OpFoldResult attrOrValue, OpBuilder &builder,
                             Location loc) {
  IntegerAttr attr;
  if (Value val = attrOrValue.dyn_cast<Value>()) {
    if (val.getType().isIndex()) return val;
    matchPattern(val, m_Constant(&attr));
  } else {
    attr = attrOrValue.get<Attribute>().cast<IntegerAttr>();
  }
  return builder.createOrFold<arith::ConstantIndexOp>(loc, attr.getInt());
}

/// Drops leading one dimensions from the given `shape`.
static ArrayRef<int64_t> dropLeadingOne(ArrayRef<int64_t> shape) {
  auto newShape = shape.drop_while([](int64_t dim) { return dim == 1; });
  return newShape.empty() ? shape.back() : newShape;
}

namespace {

/// Vectorizes tensor.pad ops by generating scf.if guards around
/// vector.transfer_read ops, e.g., converting the following IR:
///
/// ```
/// %pad = tensor.pad %s ... : tensor<1x?x?x3xf32> -> tensor<1x2x2x3xf32>
/// ```
///
/// into
///
/// ```
/// %full = <cst-vector> : vector<2x2x3xf32>
/// %slice00 = scf.if <[..][0][0][..]-in-bound> {
///   %r = vector.transfer_read %s[0, <0-lowpad1>, <0-lowpad2>, 0]
///        -> vector<3xf32>
///   scf.yield %r
/// } else {
///   scf.yield <cst-vector>
/// }
/// %insert00 = vector.insert_strided_slice %slice00, %full
/// %insert01 = <similarly-for-[..][0][1][..]>
/// %insert10 = <similarly-for-[..][1][0][..]>
/// %insert11 = <similarly-for-[..][1][1][..]>
/// %init = tensor.empty() : tensor<1x2x2x3xf32>
/// %pad = vector.transfer_write %insert11, %init
/// ```
struct VectorizePadWithConditions final
    : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // Static result shape is needed to reading padded dimensions in an
    // unrolled manner.
    if (!padOp.getType().hasStaticShape()) return failure();

    // Only support constant padding value cases.
    Value paddingValue = padOp.getConstantPaddingValue();
    if (!paddingValue) return failure();
    Attribute paddingAttr;
    if (!matchPattern(paddingValue, m_Constant(&paddingAttr))) {
      return failure();
    }

    SmallVector<OpFoldResult> lowPads = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPads = padOp.getMixedHighPad();

    /// Return true if the given `attrOrValue` is a constant zero.
    auto isConstantZero = [](OpFoldResult attrOrValue) {
      if (attrOrValue.is<Attribute>()) {
        auto attr = attrOrValue.get<Attribute>().dyn_cast<IntegerAttr>();
        return attr && attr.getValue().getZExtValue() == 0;
      }
      IntegerAttr attr;
      return matchPattern(attrOrValue.get<Value>(), m_Constant(&attr)) &&
             attr.getValue().getZExtValue() == 0;
    };

    int64_t tensorRank = padOp.getType().getRank();
    ArrayRef<int64_t> paddedTensorShape = padOp.getType().getShape();

    MLIRContext *context = padOp.getContext();
    Location loc = padOp.getLoc();

    AffineExpr sym0, sym1;
    bindSymbols(context, sym0, sym1);
    auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);
    auto subMap = AffineMap::get(0, 2, {sym0 - sym1}, context);

    /// Collects dimension indices that have non-zero low or high padding and
    /// compute the lower bounds and upper bounds for in-bound indices.
    SmallVector<int> paddedDimIndices;
    SmallVector<Value> paddedDimLBs(tensorRank);
    SmallVector<Value> paddedDimUBs(tensorRank);
    for (int i = 0; i < tensorRank; ++i) {
      if (isConstantZero(lowPads[i]) && isConstantZero(highPads[i])) continue;

      paddedDimIndices.push_back(i);
      auto srcDimSize =
          rewriter.createOrFold<tensor::DimOp>(loc, padOp.getSource(), i);
      auto lb = getAsIndexValue(lowPads[i], rewriter, loc);
      auto ub = rewriter.create<AffineApplyOp>(loc, addMap,
                                               ValueRange{lb, srcDimSize});
      paddedDimLBs[i] = lb;
      paddedDimUBs[i] = ub;
    }

    Type elementType = padOp.getType().getElementType();
    auto fullVectorType =
        VectorType::get(dropLeadingOne(paddedTensorShape), elementType);
    Value fullVector = rewriter.createOrFold<arith::ConstantOp>(
        loc, SplatElementsAttr::get(fullVectorType, {paddingAttr}));

    auto sliceVectorShape = llvm::to_vector<4>(paddedTensorShape);
    for (int dim : paddedDimIndices) sliceVectorShape[dim] = 1;
    auto sliceVectorType =
        VectorType::get(dropLeadingOne(sliceVectorShape), elementType);
    Value cstSliceVector = rewriter.createOrFold<arith::ConstantOp>(
        loc, SplatElementsAttr::get(sliceVectorType, {paddingAttr}));

    // Calculate the total count of all padded dimensions. We need to generate
    // vector read ops with scf.if guards for each of them.
    int totalCount = 1;
    for (int dim : paddedDimIndices) totalCount *= paddedTensorShape[dim];

    auto zeroIndex = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
    auto trueAttr = rewriter.getBoolAttr(true);

    SmallVector<int64_t> staticIndices(tensorRank, 0);
    SmallVector<Value> valueIndices(tensorRank, zeroIndex);
    SmallVector<Value> readIndices(tensorRank, zeroIndex);

    // All reads are inbounds given we will use scf.if to guard.
    SmallVector<bool> inBounds(sliceVectorType.getRank(), true);
    SmallVector<int64_t> staticStrides(sliceVectorType.getRank(), 1);

    for (int i = 0; i < totalCount; ++i) {
      // Delinearize the 1-D index into n-D indices needed to access the padded
      // dimensions of original tensor.
      int linearIndex = i;
      for (int dim : llvm::reverse(paddedDimIndices)) {
        staticIndices[dim] = linearIndex % paddedTensorShape[dim];
        valueIndices[dim] = rewriter.createOrFold<arith::ConstantIndexOp>(
            loc, staticIndices[dim]);
        linearIndex /= paddedTensorShape[dim];
      }

      // Build the condition: we read only if all indices are in bounds.
      Value condition = rewriter.createOrFold<arith::ConstantOp>(loc, trueAttr);
      for (int dim : paddedDimIndices) {
        Value lt = rewriter.createOrFold<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, valueIndices[dim],
            paddedDimLBs[dim]);
        Value ge = rewriter.createOrFold<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, valueIndices[dim],
            paddedDimUBs[dim]);
        Value logicalAnd = rewriter.createOrFold<arith::AndIOp>(loc, lt, ge);
        condition =
            rewriter.createOrFold<arith::AndIOp>(loc, condition, logicalAnd);
      }

      // Need to subtract the low padding to get the index into the source.
      for (int dim : paddedDimIndices) {
        readIndices[dim] = rewriter.create<AffineApplyOp>(
            loc, subMap, ValueRange{valueIndices[dim], paddedDimLBs[dim]});
      }

      auto ifOp = rewriter.create<scf::IfOp>(
          loc, condition,
          [&](OpBuilder builder, Location Loc) {
            Value read = builder.create<vector::TransferReadOp>(
                loc, sliceVectorType, padOp.getSource(), readIndices,
                paddingValue, llvm::ArrayRef(inBounds));
            builder.create<scf::YieldOp>(loc, read);
          },
          [&](OpBuilder builder, Location Loc) {
            builder.create<scf::YieldOp>(loc, cstSliceVector);
          });

      // Insert this slice back to the full vector.
      fullVector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, ifOp.getResult(0), fullVector,
          llvm::ArrayRef(staticIndices).take_back(fullVectorType.getRank()),
          staticStrides);
    }

    Value fullTensor = rewriter.create<tensor::EmptyOp>(
        loc, paddedTensorShape, elementType, ValueRange());
    valueIndices.assign(tensorRank, zeroIndex);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        padOp, fullVector, fullTensor, valueIndices);

    return success();
  }
};

struct TensorToVectorVectorizePadPass
    : public TensorToVectorVectorizePadBase<TensorToVectorVectorizePadPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, arith::ArithDialect, linalg::LinalgDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateVectorizePadPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateVectorizePadPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit) {
  patterns.add<VectorizePadWithConditions>(patterns.getContext(), baseBenefit);
}

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizePadPass() {
  return std::make_unique<TensorToVectorVectorizePadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
