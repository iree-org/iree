// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <sstream>

#define DEBUG_TYPE "iree-global-opt-lift-generic-to-tranpose-batch-matmul"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

bool isCastOfBlockArgument(Operation *op) {
  return isa<CastOpInterface>(op) && op->getNumOperands() == 1 &&
         isa<BlockArgument>(op->getOperand(0));
}

bool isCastOrInputBlockArgument(Value input, int64_t numInputs) {
  if (!input.isa<BlockArgument>()) {
    Operation *castOp0 = input.getDefiningOp();
    if (!castOp0 || !isCastOfBlockArgument(castOp0)) {
      return false;
    }
    return castOp0->getOperand(0).cast<BlockArgument>().getArgNumber() !=
           numInputs;
  } else {
    return input.cast<BlockArgument>().getArgNumber() != numInputs;
  }
}

static bool isBlockArgumentAtIndex(Value input, int64_t index) {
  return input.isa<BlockArgument>() &&
         input.cast<BlockArgument>().getArgNumber() == index;
}

// Returns true if the linalg::GenericOp has a body like a matmul. This
// does not check the indexing maps
//
// This function looks for a body like:
// ```mlir
// ^bb0(%in: !lhs, %in_0: !rhs, %out: !out):
//   %3 = arith.extui %in : !lhs to !out
//   %4 = arith.extsi %in_0 : !rhs to !out
//   %5 = arith.muli %3, %4 : !out
//   %6 = arith.addi %5, %out : !out
//   linalg.yield %6 : !out
// ```
// Ensuring the following conditions:
//    1) linalg.yield result comes from an arith.add op, accumulating on %out
//    2) The other arith.add operand comes from arith.mul
//    3) Both arith.mul operands are either block input arguments, or produced
//       by a `CastOpInterface` of a block input argument
static LogicalResult hasMatmulBody(RewriterBase &rewriter,
                                   linalg::GenericOp genericOp) {
  int numInputs = genericOp.getNumDpsInputs();
  if (numInputs != 2) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "op does not have exactly 2 inputs\n");
  }
  int numOutputs = genericOp.getNumDpsInits();
  if (numOutputs != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "op does not have exactly 1 output\n");
  }

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value yieldedValue = yieldOp->getOperand(0);

  // Check that yielded value is an arith.add op, and is accumulating
  Operation *addOp = yieldedValue.getDefiningOp();
  if (!addOp) {
    return rewriter.notifyMatchFailure(
        genericOp, "linalg.yield operand has no defining op\n");
  }
  if (!isa<arith::AddFOp, arith::AddIOp>(*addOp)) {
    return rewriter.notifyMatchFailure(genericOp, "no arith.add body op\n");
  }
  Value add0 = addOp->getOperand(0);
  Value add1 = addOp->getOperand(1);
  if (!isBlockArgumentAtIndex(add0, numInputs) &&
      !isBlockArgumentAtIndex(add1, numInputs)) {
    LLVM_DEBUG(llvm::dbgs()
               << "arith.add body op not accumulating on output\n");
  }

  // Check that the producer of the add is an arith.mul op
  Operation *mulOp =
      add0.isa<BlockArgument>() ? add1.getDefiningOp() : add0.getDefiningOp();
  if (!mulOp) {
    return rewriter.notifyMatchFailure(
        genericOp, "arith.add operand has no defining op\n");
  }
  if (!isa<arith::MulFOp, arith::MulIOp>(*mulOp)) {
    return rewriter.notifyMatchFailure(genericOp, "no arith.mul body op\n");
  }

  // Check that non block args come from arith.ext ops
  if (!isCastOrInputBlockArgument(mulOp->getOperand(0), numInputs) ||
      !isCastOrInputBlockArgument(mulOp->getOperand(1), numInputs)) {
    return rewriter.notifyMatchFailure(
        genericOp,
        "arith.mul operands are not CastOpInterface or BlockArgument\n");
  }
  return success();
}

static Value transposeTensor(Location loc, PatternRewriter &rewriter,
                             Value input, SmallVector<int64_t> perm) {
  if (!perm.size()) {
    return input;
  }
  if (llvm::all_of(llvm::enumerate(perm),
                   [](auto idx) { return idx.index() == idx.value(); })) {
    return input;
  }
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<OpFoldResult> inputMixedSizes =
      tensor::getMixedSizes(rewriter, loc, input);
  SmallVector<OpFoldResult> newInputMixedSizes =
      applyPermutation(inputMixedSizes, perm);
  Value init = rewriter.create<tensor::EmptyOp>(loc, newInputMixedSizes,
                                                inputType.getElementType());
  return rewriter.create<linalg::TransposeOp>(loc, input, init, perm)
      .getResults()[0];
}

static FailureOr<Value> castTensor(Location loc, PatternRewriter &rewriter,
                                   linalg::GenericOp genericOp,
                                   int64_t inputIdx, Value input) {
  Value output = genericOp.getResults()[0];
  auto inputType = cast<RankedTensorType>(input.getType());
  auto outputType = cast<RankedTensorType>(output.getType());
  if (inputType.getElementType() == outputType.getElementType()) {
    return input;
  }
  for (auto bodyOp : genericOp.getBody()->getOps<CastOpInterface>()) {
    Value castInput = bodyOp->getOperand(0);
    if (isBlockArgumentAtIndex(castInput, inputIdx)) {
      return createGenericElementwiseCastOp(
          rewriter, loc, input, bodyOp,
          linalg::getPrunedAttributeList(genericOp));
    }
  }
  return failure();
}

template <typename OpTy>
static LogicalResult
liftGenericOp(PatternRewriter &rewriter, linalg::GenericOp genericOp,
              SmallVector<int64_t> lhsPerm, SmallVector<int64_t> rhsPerm,
              SmallVector<int64_t> outPerm) {
  static_assert((std::is_same<OpTy, linalg::BatchVecmatOp>::value ||
                 std::is_same<OpTy, linalg::BatchMatvecOp>::value ||
                 std::is_same<OpTy, linalg::BatchMatmulOp>::value) &&
                "expected only BatchVecmatOp, BatchMatvecOp, or BatchMatmulOp");
  Location loc = genericOp.getLoc();
  Value transposedLhs =
      transposeTensor(loc, rewriter, genericOp.getInputs()[0], lhsPerm);
  Value transposedRhs =
      transposeTensor(loc, rewriter, genericOp.getInputs()[1], rhsPerm);
  FailureOr<Value> extendedLhs =
      castTensor(loc, rewriter, genericOp, 0, transposedLhs);
  FailureOr<Value> extendedRhs =
      castTensor(loc, rewriter, genericOp, 1, transposedRhs);
  if (failed(extendedLhs) || failed(extendedRhs)) {
    return failure();
  }
  Value genericInit = genericOp.getDpsInitOperand(0)->get();
  SmallVector<OpFoldResult> genericMixedSizes =
      tensor::getMixedSizes(rewriter, loc, genericInit);
  SmallVector<OpFoldResult> batchMixedSizes =
      applyPermutation(genericMixedSizes, invertPermutationVector(outPerm));
  Value out = genericOp.getResults()[0];
  auto outType = cast<RankedTensorType>(out.getType());
  Value batchEmpty = rewriter.create<tensor::EmptyOp>(loc, batchMixedSizes,
                                                      outType.getElementType());
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(outType.getElementType()));
  Value batchInit =
      rewriter.create<linalg::FillOp>(loc, zero, batchEmpty).getResult(0);
  OpTy batchOp = rewriter.create<OpTy>(
      loc, TypeRange{batchInit.getType()},
      ValueRange{extendedLhs.value(), extendedRhs.value()},
      ValueRange{batchInit});
  Value newOut = transposeTensor(loc, rewriter, batchOp.getResult(0), outPerm);
  rewriter.replaceOp(genericOp, newOut);
  return success();
}

static LogicalResult
liftToBatchVecmat(PatternRewriter &rewriter, linalg::GenericOp genericOp,
                  linalg::ContractionDimensions contractionDims) {
  if (contractionDims.batch.size() != 1 || contractionDims.m.size() != 0 ||
      contractionDims.n.size() != 1 || contractionDims.k.size() != 1) {
    return rewriter.notifyMatchFailure(
        genericOp, "expected batch vecmat contraction dims\n\n");
  }
  AffineMap vecMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap matMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (vecMap.getNumResults() != 2 || matMap.getNumResults() != 3 ||
      outMap.getNumResults() != 2) {
    return rewriter.notifyMatchFailure(
        genericOp, "wrong numResults for indexing maps\n\n");
  }

  auto getResultIndex = [&](AffineMap map, int64_t dimIndex) {
    return *(map.getResultPosition(rewriter.getAffineDimExpr(dimIndex)));
  };
  // Permutation from GenericOp lhs shape to BatchVecmatOp lhs shape
  SmallVector<int64_t> vecPerm;
  vecPerm.push_back(getResultIndex(vecMap, contractionDims.batch[0]));
  vecPerm.push_back(getResultIndex(vecMap, contractionDims.k[0]));
  // Permutation from GenericOp rhs shape to BatchVecmatOp rhs shape
  SmallVector<int64_t> matPerm;
  matPerm.push_back(getResultIndex(matMap, contractionDims.batch[0]));
  matPerm.push_back(getResultIndex(matMap, contractionDims.k[0]));
  matPerm.push_back(getResultIndex(matMap, contractionDims.n[0]));
  // Permutation from BatchVecmatOp result shape to GenericOp result shape
  SmallVector<int64_t> outPerm;
  outPerm.push_back(getResultIndex(outMap, contractionDims.batch[0]));
  outPerm.push_back(getResultIndex(outMap, contractionDims.n[0]));
  outPerm = invertPermutationVector(outPerm);
  return liftGenericOp<linalg::BatchVecmatOp>(rewriter, genericOp, vecPerm,
                                              matPerm, outPerm);
}

static LogicalResult
liftToBatchMatvec(PatternRewriter &rewriter, linalg::GenericOp genericOp,
                  linalg::ContractionDimensions contractionDims) {
  if (contractionDims.batch.size() != 1 || contractionDims.m.size() != 1 ||
      contractionDims.n.size() != 0 || contractionDims.k.size() != 1) {
    return rewriter.notifyMatchFailure(
        genericOp, "expected batch matvec contraction dims\n\n");
  }
  AffineMap matMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap vecMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (vecMap.getNumResults() != 2 || matMap.getNumResults() != 3 ||
      outMap.getNumResults() != 2) {
    return rewriter.notifyMatchFailure(
        genericOp, "wrong numResults for indexing maps\n\n");
  }

  auto getResultIndex = [&](AffineMap map, int64_t dimIndex) {
    return *(map.getResultPosition(rewriter.getAffineDimExpr(dimIndex)));
  };
  // Permutation from GenericOp lhs shape to BatchMatvecOp lhs shape
  SmallVector<int64_t> matPerm;
  matPerm.push_back(getResultIndex(matMap, contractionDims.batch[0]));
  matPerm.push_back(getResultIndex(matMap, contractionDims.m[0]));
  matPerm.push_back(getResultIndex(matMap, contractionDims.k[0]));
  // Permutation from GenericOp rhs shape to BatchMatvecOp rhs shape
  SmallVector<int64_t> vecPerm;
  vecPerm.push_back(getResultIndex(vecMap, contractionDims.batch[0]));
  vecPerm.push_back(getResultIndex(vecMap, contractionDims.k[0]));
  // Permutation from BatchMatvecOp result shape to GenericOp result shape
  SmallVector<int64_t> outPerm;
  outPerm.push_back(getResultIndex(outMap, contractionDims.batch[0]));
  outPerm.push_back(getResultIndex(outMap, contractionDims.m[0]));
  outPerm = invertPermutationVector(outPerm);
  return liftGenericOp<linalg::BatchMatvecOp>(rewriter, genericOp, matPerm,
                                              vecPerm, outPerm);
}

static LogicalResult
liftToBatchMatmul(PatternRewriter &rewriter, linalg::GenericOp genericOp,
                  linalg::ContractionDimensions contractionDims) {
  if (contractionDims.batch.size() != 1 || contractionDims.m.size() != 1 ||
      contractionDims.n.size() != 1 || contractionDims.k.size() != 1) {
    return rewriter.notifyMatchFailure(
        genericOp, "expected batch matmul contraction dims\n\n");
  }
  assert(contractionDims.batch.size() == 1 && contractionDims.m.size() == 1 &&
         contractionDims.n.size() == 1 && contractionDims.k.size() == 1 &&
         "expected batch matmul contraction dims");
  AffineMap lhsMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap rhsMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  if (lhsMap.getNumResults() != 3 || rhsMap.getNumResults() != 3 ||
      outMap.getNumResults() != 3) {
    return rewriter.notifyMatchFailure(
        genericOp, "wrong numResults for indexing maps\n\n");
  }

  auto getResultIndex = [&](AffineMap map, int64_t dimIndex) {
    return *(map.getResultPosition(rewriter.getAffineDimExpr(dimIndex)));
  };
  // Permutation from GenericOp lhs shape to BatchMatmulOp lhs shape
  SmallVector<int64_t> lhsPerm;
  lhsPerm.push_back(getResultIndex(lhsMap, contractionDims.batch[0]));
  lhsPerm.push_back(getResultIndex(lhsMap, contractionDims.m[0]));
  lhsPerm.push_back(getResultIndex(lhsMap, contractionDims.k[0]));
  // Permutation from GenericOp rhs shape to BatchMatmulOp rhs shape
  SmallVector<int64_t> rhsPerm;
  rhsPerm.push_back(getResultIndex(rhsMap, contractionDims.batch[0]));
  rhsPerm.push_back(getResultIndex(rhsMap, contractionDims.k[0]));
  rhsPerm.push_back(getResultIndex(rhsMap, contractionDims.n[0]));
  // Permutation from BatchMatmulOp result shape to GenericOp result shape
  SmallVector<int64_t> outPerm;
  outPerm.push_back(getResultIndex(outMap, contractionDims.batch[0]));
  outPerm.push_back(getResultIndex(outMap, contractionDims.m[0]));
  outPerm.push_back(getResultIndex(outMap, contractionDims.n[0]));
  outPerm = invertPermutationVector(outPerm);
  return liftGenericOp<linalg::BatchMatmulOp>(rewriter, genericOp, lhsPerm,
                                              rhsPerm, outPerm);
}

// Converts linalg.generic op to linalg.batch_matmul, linalg.batch_matvec,
// or linalg.batch_vecmat, plus linalg.transpose ops on the inputs
class LiftGenericToTransposeBatchMatmul
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(genericOp);
    if (failed(contractionDims)) {
      return rewriter.notifyMatchFailure(
          genericOp, "failed to infer contraction dims\n\n");
    }

    auto lhsType =
        dyn_cast<RankedTensorType>(genericOp.getOperands()[0].getType());
    auto rhsType =
        dyn_cast<RankedTensorType>(genericOp.getOperands()[1].getType());
    auto outType =
        dyn_cast<RankedTensorType>(genericOp.getResults()[0].getType());
    if (!lhsType || !rhsType || !outType) {
      return rewriter.notifyMatchFailure(
          genericOp, "Operands do not have RankedTensorType\n\n");
    }

    if (failed(hasMatmulBody(rewriter, genericOp))) {
      return rewriter.notifyMatchFailure(
          genericOp, "genericOp does not have a matmul body\n\n");
    }

    // TODO(#15373) Support non-batch cases
    if (!failed(liftToBatchVecmat(rewriter, genericOp, *contractionDims))) {
      return success();
    };
    if (!failed(liftToBatchMatvec(rewriter, genericOp, *contractionDims))) {
      return success();
    };
    if (!failed(liftToBatchMatmul(rewriter, genericOp, *contractionDims))) {
      return success();
    };
    return failure();
  }
};

struct LiftGenericToTransposeBatchMatmulPass
    : public LiftGenericToTransposeBatchMatmulBase<
          LiftGenericToTransposeBatchMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<LiftGenericToTransposeBatchMatmul>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLiftGenericToTransposeBatchMatmulPass() {
  return std::make_unique<LiftGenericToTransposeBatchMatmulPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
