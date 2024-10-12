// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GENERALIZETENSORPACKPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

// Indicates whether the given permutation vector is a minor identity
// for a permutation of the given |rank|.
static bool isIdentityIndices(llvm::ArrayRef<int64_t> indices) {
  if (indices.empty()) {
    return true;
  }
  int64_t base = indices[0];
  return llvm::all_of(llvm::enumerate(indices),
                      [base](auto e) { return e.value() - base == e.index(); });
}

namespace {

// This generalizes pack ops that are consumed by flow.dispatch.tensor.store
// into linalg::transpose. E.g   %pack = tensor.pack %2 outer_dims_perm = [0, 1]
// inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<8x768xf32> ->
// tensor<1x768x8x1xf32> is converted into %transposed = linalg.transpose ins(%2
// : tensor<8x768xf32>) outs(%5 : tensor<768x8xf32>) permutation = [1, 0].

LogicalResult ConvertPackToTranspose(tensor::PackOp packOp,
                                     IRRewriter &rewriter) {
  // Bail if the user is not a `flow.dispatch.tensor.store`.
  for (auto user : packOp->getUsers()) {
    if (!isa<IREE::Flow::DispatchTensorStoreOp>(user))
      return success();
  }
  // Padding is not supported by the pattern.
  if (packOp.getPaddingValue())
    return success();
  llvm::ArrayRef<int64_t> permutation = packOp.getOuterDimsPerm();

  // Outer dim permutations are not supported by the pattern.
  if (!isIdentityIndices(permutation)) {
    return success();
  }
  Location loc = packOp.getLoc();
  llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
  llvm::ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();

  // Currently we only handle pack op with static inner tile sizes.
  if (llvm::any_of(innerTiles,
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return success();
  }
  SmallVector<OpFoldResult> newInnerTiles;
  SmallVector<int64_t> newInnerDimsPos;

  for (size_t i = 0; i < innerTiles.size(); ++i) {
    if (innerTiles[i] != 1) {
      newInnerTiles.push_back(rewriter.getIndexAttr(innerTiles[i]));
      // newPermutation.push_back(permutation[i]);
      newInnerDimsPos.push_back(innerDimsPos[i]);
    }
  }

  rewriter.setInsertionPoint(packOp);

  // construct a packOp in which unit inner tiles are removed.
  Value empty = tensor::PackOp::createDestinationTensor(
      rewriter, loc, packOp.getSource(), newInnerTiles, newInnerDimsPos,
      SmallVector<int64_t>{});

  auto newPackOp = rewriter.create<tensor::PackOp>(
      loc, packOp.getSource(), empty, newInnerDimsPos, newInnerTiles,
      /*padding=*/std::nullopt, SmallVector<int64_t>{});

  RankedTensorType destType =
      cast<RankedTensorType>(newPackOp.getDest().getType());
  ArrayRef<int64_t> destShape = destType.getShape();
  innerDimsPos = newPackOp.getInnerDimsPos();

  // If we have outer dims that are not unit then this is not a tranpose so we
  // bail.
  if (llvm::any_of(innerDimsPos, [destShape](int64_t index) {
        return destShape[index] != 1;
      })) {
    return success();
  }

  // Collect the set of transposed dimensions.
  llvm::DenseSet<int64_t> innerDims;
  for (auto innerDim : innerDimsPos) {
    innerDims.insert(innerDim);
  }

  // Construct the permutation for the transpose. It is constructed as
  // [untiled_outer_dims, inner_dims_pos].
  int64_t srcRank = newPackOp.getSourceRank();
  SmallVector<int64_t> perm;
  for (int i = 0, e = srcRank; i < e; i++) {
    if (!innerDims.count(i)) {
      perm.push_back(i);
    }
  }
  perm.append(innerDimsPos.begin(), innerDimsPos.end());

  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, newPackOp.getSource());
  applyPermutationToVector(mixedSizes, perm);
  empty = rewriter.create<tensor::EmptyOp>(loc, mixedSizes,
                                           destType.getElementType());
  Value transposed =
      rewriter
          .create<linalg::TransposeOp>(loc, newPackOp.getSource(), empty, perm)
          .getResult()[0];

  /*Value replacement = rewriter.create<IREE::Flow::TensorReshapeOp>(
      loc, packOp.getType(), transposed, SmallVector<Value>{},
      SmallVector<Value>{});*/

  rewriter.replaceAllUsesWith(packOp, transposed);
  packOp->erase();
  newPackOp->erase();

  return success();
}

struct GeneralizeTensorPackPass
    : public impl::GeneralizeTensorPackPassBase<GeneralizeTensorPackPass> {
  /*void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }*/

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    auto walkResult =
        getOperation()->walk([&rewriter](tensor::PackOp op) -> WalkResult {
          if (failed(ConvertPackToTranspose(op, rewriter))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::iree_compiler
