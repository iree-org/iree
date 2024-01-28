// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

namespace {

/// Given a LayoutAttr, find the shape of the given layout dimension. It is
/// expected that the layout has at most one instance of the requested
/// dimension. Example:
///   LayoutAttr: <<BATCHX: 4>, <BATCHY: 4, LANEX: 4>>
///   dim: BATCHX
///   output: 4
static std::optional<int64_t> findDimShape(LayoutAttr layout,
                                           LayoutDimension dim) {
  for (PerDimLayoutAttr dimLayout : layout.getLayouts()) {
    if (std::optional<int64_t> shape = dimLayout.getShape(dim)) {
      return shape;
    }
  }
  return std::nullopt;
}

/// Given the state of the iterator, compute the indices of the original vector
/// that the current iterator state is iterating over. These indices are
/// parameterized by the thread grid.
static SmallVector<Value> computeSIMDIndex(const LayoutIterator::State &state,
                                           LayoutAttr layout,
                                           ArrayRef<Value> threadGrid,
                                           RewriterBase &rewriter) {
  MLIRContext *ctx = layout.getContext();
  AffineExpr threadX, threadY, threadZ;
  bindSymbols(ctx, threadX, threadY, threadZ);

  SmallVector<Value> simdIndex;
  // Calculate the index for each dim separately.
  for (PerDimLayoutAttr dimLayout : layout.getLayouts()) {
    AffineExpr offset = getAffineConstantExpr(0, ctx);
    AffineExpr stride = getAffineConstantExpr(1, ctx);
    for (auto [label, shape] : llvm::reverse(
             llvm::zip(dimLayout.getLabels(), dimLayout.getShapes()))) {
      int64_t position = state.lookup(label.getValue()).getPosition();

      switch (label.getValue()) {
      case LayoutDimension::LANEX:
        offset = offset + stride * threadX;
        break;
      case LayoutDimension::LANEY:
        offset = offset + stride * threadY;
        break;
      case LayoutDimension::LANEZ:
        offset = offset + stride * threadZ;
        break;
      default:
        offset = offset + stride * getAffineConstantExpr(position, ctx);
        break;
      }
      stride = stride * getAffineConstantExpr(shape, ctx);
    }

    // Compute the index for the dim.
    AffineMap indexMap = AffineMap::get(0, 3, offset);
    Value index = rewriter.create<affine::AffineApplyOp>(
        rewriter.getUnknownLoc(), indexMap, threadGrid);
    simdIndex.push_back(index);
  }

  return simdIndex;
}

struct DistributeConstants final : OpDistributionPattern<arith::ConstantOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    auto constant = dyn_cast<VectorValue>(constantOp.getResult());
    if (!constant)
      return failure();

    // Only handle splat values for now.
    auto attr = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!attr)
      return failure();

    VectorLayoutInterface layout = signature[constant];

    // Replace the original op with the distributed op.
    Type elementType = constant.getType().getElementType();
    auto vectorType =
        VectorType::get(layout.getDistributedShape(), elementType);
    Operation *distirbutedOp = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), vectorType,
        SplatElementsAttr::get(vectorType, attr.getSplatValue<Attribute>()));
    replaceOpWithDistributedValues(rewriter, constantOp,
                                   distirbutedOp->getResult(0));
    return success();
  }
};

template <typename OpTy>
struct DistributeElementwise final : OpDistributionPattern<OpTy> {
  using OpDistributionPattern<OpTy>::OpDistributionPattern;

  LogicalResult matchAndRewrite(OpTy op, DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      signature[vectorOperand]);
      }
      operands.push_back(operand);
    }

    // Get the new distributed vector types for the operation.
    SmallVector<Type> resultTypes;
    for (Value result : op->getResults()) {
      Type resultType = result.getType();

      // Distribute vector result types.
      if (auto vectorResult = dyn_cast<VectorValue>(result)) {
        VectorLayoutInterface resLayout = signature[vectorResult];
        resultType = VectorType::get(resLayout.getDistributedShape(),
                                     vectorResult.getType().getElementType());
      }
      resultTypes.push_back(resultType);
    }

    // Replace the original op with the distributed op.
    Operation *distributedOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), operands, resultTypes);

    // Propagate known attributes.
    StringRef fastmathAttrName = arith::FastMathFlagsAttr::getMnemonic();
    if (Attribute attr = op->getAttr(fastmathAttrName)) {
      distributedOp->setAttr(fastmathAttrName, attr);
    }

    DistributionPattern::replaceOpWithDistributedValues(
        rewriter, op, distributedOp->getResults());
    return success();
  }
};

/// Given a projected permutation, get a reduced permutation, i.e. without
/// the projected dimensions.
static SmallVector<int64_t> getReducedPermutation(AffineMap permutationMap) {
  assert(permutationMap.isProjectedPermutation() &&
         "permutation map should be a projected permutation.");
  // TODO: The permutation map may also have broadcasting. Currently, we do not
  // handle it. This can be fixed by adding a "BROADCAST" dimension in the
  // layout.

  SmallVector<int64_t> permutation;
  permutation.reserve(permutationMap.getNumResults());

  unsigned leadingUnitDims =
      permutationMap.getNumDims() - permutationMap.getNumResults();
  for (AffineExpr dim : permutationMap.getResults()) {
    // Get this dim's position in the permutation map.
    auto dimExpr = dyn_cast<AffineDimExpr>(dim);
    if (!dimExpr) {
      llvm::report_fatal_error("permutation map is not a projected "
                               "permutation.");
    }

    unsigned pos = dimExpr.getPosition();
    assert(pos >= leadingUnitDims && "invalid permutation map");
    pos -= leadingUnitDims;
    permutation.push_back(pos);
  }
  return permutation;
}

template <typename OpTy>
struct DistributeXferLayoutAttr : OpDistributionPattern<OpTy> {
  static_assert(std::is_same<OpTy, vector::TransferReadOp>::value ||
                    std::is_same<OpTy, vector::TransferWriteOp>::value,
                "expected vector::TransferReadOp or vector::TransferWriteOp");

  DistributeXferLayoutAttr(MLIRContext *context, ArrayRef<Value> threadGrid,
                           PatternBenefit benefit = 1)
      : OpDistributionPattern<OpTy>(context, benefit), threadGrid(threadGrid) {}

  VectorValue accessMemory(OpTy xferOp, VectorValue accumulator,
                           LayoutAttr vectorLayout,
                           PatternRewriter &rewriter) const {
    // We need to take special consideration of the permutation map when
    // lowering. When accessing memory, we use the memoryLayout, because that
    // is how the data is accessed in memory. The data is stored in the vector
    // according to vectorLayout.
    SmallVector<int64_t> permutation =
        getReducedPermutation(xferOp.getPermutationMap());
    LayoutAttr memoryLayout =
        cast<LayoutAttr>(vectorLayout.permute(permutation));

    int loadWidth = getLoadStoreWidth(memoryLayout);
    DenseMap<LayoutDimension, int64_t> steps;
    steps[LayoutDimension::VECTORX] = loadWidth;
    LayoutIterator iterator(vectorLayout, steps);

    iterator.apply([&](const LayoutIterator::State &state) {
      SmallVector<Value> memoryIndices =
          getMemoryIndices(state, memoryLayout, xferOp.getIndices(), rewriter);
      SmallVector<int64_t> accIndices = state.computeSIMTIndex();
      accumulator = accessUnit(xferOp, memoryIndices, accIndices, accumulator,
                               vectorLayout, memoryLayout, rewriter);
    });

    return accumulator;
  }

  SmallVector<Value> getMemoryIndices(const LayoutIterator::State &state,
                                      LayoutAttr memoryLayout,
                                      SmallVector<Value> indices,
                                      RewriterBase &rewriter) const {
    SmallVector<Value> simdIndices =
        computeSIMDIndex(state, memoryLayout, threadGrid, rewriter);
    SmallVector<Value> memoryIndices(indices);

    // The memory layout has some projected leading dims that indices doesn't.
    int leadingProjectedDims = memoryIndices.size() - simdIndices.size();
    for (int i = leadingProjectedDims, e = memoryIndices.size(); i < e; ++i) {
      memoryIndices[i] = rewriter.create<arith::AddIOp>(
          rewriter.getUnknownLoc(), memoryIndices[i],
          simdIndices[i - leadingProjectedDims]);
    }

    return memoryIndices;
  }

  virtual VectorValue accessUnit(OpTy xferOp, SmallVector<Value> &memoryIndices,
                                 SmallVector<int64_t> &accIndices,
                                 VectorValue accumulator,
                                 LayoutAttr vectorLayout,
                                 LayoutAttr memoryLayout,
                                 PatternRewriter &rewriter) const = 0;

  int getLoadStoreWidth(LayoutAttr layout) const {
    PerDimLayoutAttr fastestChanging = layout.getLayouts().back();
    if (std::optional<int64_t> width =
            fastestChanging.getShape(LayoutDimension::VECTORX)) {
      return *width;
    }
    return 1;
  }

  SmallVector<Value> threadGrid;
};

struct DistributeTransferReadLayoutAttr final
    : DistributeXferLayoutAttr<vector::TransferReadOp> {
  using DistributeXferLayoutAttr::DistributeXferLayoutAttr;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    LayoutAttr vectorLayout =
        dyn_cast<LayoutAttr>(signature[readOp.getResult()]);
    if (!vectorLayout) {
      return failure();
    }

    // TODO: Return failure if we need masking.

    Type elementType = readOp.getSource().getType().getElementType();
    auto vectorType =
        VectorType::get(vectorLayout.getDistributedShape(), elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        readOp.getLoc(), vectorType, rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    VectorValue readVec = accessMemory(readOp, acc, vectorLayout, rewriter);

    replaceOpWithDistributedValues(rewriter, readOp, readVec);
    return success();
  }

  VectorValue accessUnit(vector::TransferReadOp readOp,
                         SmallVector<Value> &memoryIndices,
                         SmallVector<int64_t> &accIndices,
                         VectorValue accumulator, LayoutAttr vectorLayout,
                         LayoutAttr memoryLayout,
                         PatternRewriter &rewriter) const override {
    auto unitType = VectorType::get({getLoadStoreWidth(memoryLayout)},
                                    accumulator.getType().getElementType());
    VectorValue load = rewriter.create<vector::LoadOp>(
        readOp.getLoc(), unitType, readOp.getSource(), memoryIndices);
    return rewriter.create<vector::InsertStridedSliceOp>(
        readOp.getLoc(), load, accumulator, accIndices,
        SmallVector<int64_t>{1});
  }
};

struct DistributeTransferWriteLayoutAttr final
    : DistributeXferLayoutAttr<vector::TransferWriteOp> {
  using DistributeXferLayoutAttr::DistributeXferLayoutAttr;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    LayoutAttr vectorLayout =
        dyn_cast<LayoutAttr>(signature[writeOp.getVector()]);
    if (!vectorLayout) {
      return failure();
    }

    // TODO: Return failure if we need masking.

    accessMemory(writeOp, writeOp.getVector(), vectorLayout, rewriter);

    rewriter.eraseOp(writeOp);
    return success();
  }

  VectorValue accessUnit(vector::TransferWriteOp writeOp,
                         SmallVector<Value> &memoryIndices,
                         SmallVector<int64_t> &accIndices,
                         VectorValue accumulator, LayoutAttr vectorLayout,
                         LayoutAttr memoryLayout,
                         PatternRewriter &rewriter) const override {
    int width = getLoadStoreWidth(memoryLayout);

    SmallVector<int64_t> strides(accIndices.size(), 1);
    SmallVector<int64_t> shapes(accIndices.size(), 1);
    shapes[shapes.size() - 1] = width;
    Value result = rewriter.create<vector::ExtractStridedSliceOp>(
        writeOp.getLoc(), getDistributed(rewriter, accumulator, vectorLayout),
        accIndices, shapes, strides);
    result = rewriter.create<vector::ExtractOp>(
        writeOp.getLoc(), result,
        SmallVector<int64_t>(accIndices.size() - 1, 0));
    rewriter.create<vector::StoreOp>(writeOp.getLoc(), result,
                                     writeOp.getSource(), memoryIndices);

    return accumulator;
  }
};

struct DistributeScfYield final : OpDistributionPattern<scf::YieldOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // Get the distributed operands.
    SmallVector<Value> operands;
    for (Value operand : yieldOp->getOperands()) {
      if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
                                                      signature[vectorOperand]);
      }
      operands.push_back(operand);
    }

    // Since this operation has no results, we can directly replace it.
    auto distributedYieldOp =
        rewriter.create<scf::YieldOp>(yieldOp.getLoc(), operands);
    rewriter.replaceOp(yieldOp, distributedYieldOp);

    return success();
  }
};

struct DistributeScfFor final : OpDistributionPattern<scf::ForOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Block *oldLoopBody = forOp.getBody();

    // The new vector init_args of the loop.
    SmallVector<Value> newInitArgs;
    for (Value initArg : forOp.getInitArgs()) {
      if (auto vectorInitArg = dyn_cast<VectorValue>(initArg)) {
        initArg =
            getDistributed(rewriter, vectorInitArg, signature[vectorInitArg]);
      }
      newInitArgs.push_back(initArg);
    }

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block *loopBody = newForOp.getBody();

    // Set up new iter_args. The loop body uses SIMD, so wrap the SIMD iter_args
    // of the new loop op into ToSIMDOps.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> iterArgs = getBbArgsReplacements(
        rewriter, newForOp.getRegionIterArgs(), forOp.getInitArgs());
    iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

    // Move loop body to new loop.
    rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

    // Repleace loop results.
    replaceOpWithDistributedValues(rewriter, forOp, newForOp.getResults());
    return success();
  }

  /// Helper function for loop distribution. Given a list of bbArgs of the new
  /// (distributed) loop op, wrap the distributed vector args (now distributed)
  /// into ToSIMDOps, so that the block body can be moved over to the new op.
  SmallVector<Value> getBbArgsReplacements(RewriterBase &rewriter,
                                           Block::BlockArgListType bbArgs,
                                           ValueRange oldInits) const {
    SmallVector<Value> replacements;
    for (auto [bbArg, oldInit] : llvm::zip_equal(bbArgs, oldInits)) {
      Value val = bbArg;
      if (auto oldVectorInit = dyn_cast<VectorValue>(oldInit)) {
        val = rewriter.create<IREE::VectorExt::ToSIMDOp>(
            oldVectorInit.getLoc(), oldVectorInit.getType(), val);
      }
      replacements.push_back(val);
    }
    return replacements;
  }
};

} // namespace

void populateGPUDistributionPatterns(RewritePatternSet &patterns) {
  patterns.add<DistributeConstants, DistributeScfYield, DistributeScfFor>(
      patterns.getContext());
  // Elementwise patterns.
  patterns.add<DistributeElementwise<arith::MulIOp>,
               DistributeElementwise<arith::MulFOp>,
               DistributeElementwise<arith::AddIOp>,
               DistributeElementwise<arith::AddFOp>>(patterns.getContext());
}

void populateGPUDistributionLayoutAttrPatterns(ArrayRef<Value> threadGrid,
                                               RewritePatternSet &patterns) {
  patterns
      .add<DistributeTransferReadLayoutAttr, DistributeTransferWriteLayoutAttr>(
          patterns.getContext(), threadGrid);
}

}; // namespace mlir::iree_compiler
