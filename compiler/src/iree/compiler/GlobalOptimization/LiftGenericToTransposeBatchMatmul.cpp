// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <sstream>

#define DEBUG_TYPE "iree-global-opt-lift-generic-to-tranpose-batch-matmul"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

template <typename T, unsigned int N>
mlir::raw_ostream &operator<<(mlir::raw_ostream &s,
                              const SmallVector<T, N> &v) {
  s << "[ ";
  for (const auto &e : v) {
    s << e << " ";
  }
  s << "]";
  return s;
}

static bool hasMatmulBody(linalg::GenericOp genericOp) {
  int numInputs = genericOp.getNumDpsInputs();
  if (numInputs != 2) {
    LLVM_DEBUG(llvm::dbgs() << "op does not have exactly 2 inputs\n");
    return false;
  }
  int numOutputs = genericOp.getNumDpsInits();
  if (numOutputs != 1) {
    LLVM_DEBUG(llvm::dbgs() << "op does not have exactly 1 output\n");
    return false;
  }

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value yieldedValue = yieldOp->getOperand(0);

  // Check that yielded value is an arith.add op, and is accumulating
  Operation *addOp = yieldedValue.getDefiningOp();
  if (!addOp || addOp->getNumOperands() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "no arith.add body op, wrong numOperands\n");
    return false;
  }
  if (!matchPattern(addOp, m_Op<arith::AddFOp>()) &&
      !matchPattern(addOp, m_Op<arith::AddIOp>())) {
    LLVM_DEBUG(llvm::dbgs() << "no arith.add body op\n");
    return false;
  }
  Value add0 = addOp->getOperand(0);
  Value add1 = addOp->getOperand(1);
  if (!(add0.isa<BlockArgument>() &&
        add0.cast<BlockArgument>().getArgNumber() == numInputs) &&
      !(add1.isa<BlockArgument>() &&
        add1.cast<BlockArgument>().getArgNumber() == numInputs)) {
    LLVM_DEBUG(llvm::dbgs()
               << "arith.add body op not accumulating on output\n");
    return false;
  }

  // Check that the producer of the add is an arith.mul op
  Operation *mulOp;
  if (add0.isa<BlockArgument>()) {
    mulOp = add1.getDefiningOp();
  } else {
    mulOp = add0.getDefiningOp();
  }
  if (!mulOp || mulOp->getNumOperands() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "no arith.mul body op, wrong numOperands\n");
    return false;
  }
  if (!matchPattern(mulOp, m_Op<arith::MulFOp>()) &&
      !matchPattern(mulOp, m_Op<arith::MulIOp>())) {
    LLVM_DEBUG(llvm::dbgs() << "no arith.mul body op\n");
    return false;
  }
  Value mul0 = mulOp->getOperand(0);
  Value mul1 = mulOp->getOperand(1);
  if (mul0.isa<BlockArgument>() && mul1.isa<BlockArgument>()) {
    return true;
  }

  // Check that non block args come from arith.ext ops
  if (!mul0.isa<BlockArgument>()) {
    Operation *extOp0 = mul0.getDefiningOp();
    if (!extOp0 || extOp0->getNumOperands() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "no arith.ext body op on in0, wrong numOperands\n");
      return false;
    }
    if (!matchPattern(extOp0, m_Op<arith::ExtUIOp>()) &&
        !matchPattern(extOp0, m_Op<arith::ExtSIOp>()) &&
        !matchPattern(extOp0, m_Op<arith::ExtFOp>())) {
      LLVM_DEBUG(llvm::dbgs() << "no arith.ext body op on in0\n");
      return false;
    }
    Value ext0 = extOp0->getOperand(0);
    if (!ext0 || !ext0.isa<BlockArgument>()) {
      return false;
    }
  }
  if (!mul1.isa<BlockArgument>()) {
    Operation *extOp1 = mul1.getDefiningOp();
    if (!extOp1 || extOp1->getNumOperands() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "no arith.ext body op on in1, wrong numOperands\n");
      return false;
    }
    if (!matchPattern(extOp1, m_Op<arith::ExtUIOp>()) &&
        !matchPattern(extOp1, m_Op<arith::ExtSIOp>()) &&
        !matchPattern(extOp1, m_Op<arith::ExtFOp>())) {
      LLVM_DEBUG(llvm::dbgs() << "no arith.ext body op on in1\n");
      return false;
    }
    Value ext1 = extOp1->getOperand(0);
    if (!ext1 || !ext1.isa<BlockArgument>()) {
      return false;
    }
  }
  return true;
}

static SmallVector<int64_t> getInverseTransposePerm(SmallVector<int64_t> perm) {
  SmallVector<int64_t> inversePerm(perm);
  for (auto permIdx : llvm::enumerate(perm)) {
    inversePerm[permIdx.value()] = permIdx.index();
  }
  return inversePerm;
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
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> newInputShape(inputShape);
  for (auto permIdx : llvm::enumerate(perm)) {
    newInputShape[permIdx.index()] = inputShape[permIdx.value()];
  }
  Value init = rewriter.create<tensor::EmptyOp>(loc, newInputShape,
                                                inputType.getElementType());
  return rewriter.create<linalg::TransposeOp>(loc, input, init, perm)
      .getResults()[0];
}

static FailureOr<Value> extendTensor(Location loc, PatternRewriter &rewriter,
                                     linalg::GenericOp genericOp,
                                     int64_t inputIdx, Value input) {
  Value output = genericOp.getResults()[0];
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  auto outputType = llvm::cast<RankedTensorType>(output.getType());
  if (inputType.getElementType() == outputType.getElementType()) {
    return input;
  }
  auto extendedType =
      RankedTensorType::get(inputType.getShape(), outputType.getElementType());
  for (auto bodyOp : genericOp.getBody()->getOps<arith::ExtUIOp>()) {
    Value extInput = bodyOp.getIn();
    if (extInput.isa<BlockArgument>() &&
        extInput.cast<BlockArgument>().getArgNumber() == inputIdx) {
      return rewriter.create<arith::ExtUIOp>(loc, extendedType, input)
          .getResult();
    }
  }
  for (auto bodyOp : genericOp.getBody()->getOps<arith::ExtSIOp>()) {
    Value extInput = bodyOp.getIn();
    if (extInput.isa<BlockArgument>() &&
        extInput.cast<BlockArgument>().getArgNumber() == inputIdx) {
      return rewriter.create<arith::ExtSIOp>(loc, extendedType, input)
          .getResult();
    }
  }
  return failure();
}

template <typename OpTy>
static LogicalResult
liftGenericOp(PatternRewriter &rewriter, linalg::GenericOp genericOp,
              SmallVector<int64_t> lhsPerm, SmallVector<int64_t> rhsPerm,
              SmallVector<int64_t> outPerm) {
  assert((std::is_same<OpTy, linalg::BatchVecmatOp>::value ||
          std::is_same<OpTy, linalg::BatchMatvecOp>::value ||
          std::is_same<OpTy, linalg::BatchMatmulOp>::value) &&
         "expected only BatchVecmatOp, BatchMatvecOp, or BatchMatmulOp");
  LLVM_DEBUG(llvm::dbgs() << "lhsPerm: " << lhsPerm << "\n");
  LLVM_DEBUG(llvm::dbgs() << "rhsPerm: " << rhsPerm << "\n");
  LLVM_DEBUG(llvm::dbgs() << "outPerm: " << outPerm << "\n");
  Location loc = genericOp.getLoc();
  Value transposedLhs =
      transposeTensor(loc, rewriter, genericOp.getInputs()[0], lhsPerm);
  Value transposedRhs =
      transposeTensor(loc, rewriter, genericOp.getInputs()[1], rhsPerm);
  FailureOr<Value> extendedLhs =
      extendTensor(loc, rewriter, genericOp, 0, transposedLhs);
  FailureOr<Value> extendedRhs =
      extendTensor(loc, rewriter, genericOp, 1, transposedRhs);
  if (failed(extendedLhs) || failed(extendedRhs)) {
    return failure();
  }
  Value out = genericOp.getResults()[0];
  auto outType = llvm::cast<RankedTensorType>(out.getType());
  ArrayRef<int64_t> outputShape = outType.getShape();
  SmallVector<int64_t> batchInitShape(outputShape);
  for (auto idx : llvm::enumerate(getInverseTransposePerm(outPerm))) {
    batchInitShape[idx.index()] = outputShape[idx.value()];
  }
  Value batchEmpty = rewriter.create<tensor::EmptyOp>(loc, batchInitShape,
                                                      outType.getElementType());
  Value zero =
      outType.getElementType().isa<IntegerType>()
          ? rewriter.create<arith::ConstantOp>(
                loc, rewriter.getIntegerAttr(outType.getElementType(), 0))
          : rewriter.create<arith::ConstantOp>(
                loc, rewriter.getFloatAttr(outType.getElementType(), 0.0));
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
raiseToBatchVecmat(PatternRewriter &rewriter, linalg::GenericOp genericOp,
                   linalg::ContractionDimensions contractionDims) {
  assert(contractionDims.batch.size() == 1 && contractionDims.m.size() == 0 &&
         contractionDims.n.size() == 1 && contractionDims.k.size() == 1 &&
         "expected batch vecmat contraction dims");
  AffineMap vecMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
  AffineMap matMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(1));
  AffineMap outMap =
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0));
  assert(vecMap.getNumResults() == 2 && matMap.getNumResults() == 3 &&
         outMap.getNumResults() == 2 && "wrong numResults for indexing maps");

  // Permutation from GenericOp lhs shape to BatchVecmatOp lhs shape
  SmallVector<int64_t> vecPerm{
      *(vecMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.batch[0]))),
      *(vecMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.k[0])))};
  // Permutation from GenericOp rhs shape to BatchVecmatOp rhs shape
  SmallVector<int64_t> matPerm{
      *(matMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.batch[0]))),
      *(matMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.k[0]))),
      *(matMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.n[0])))};
  // Permutation from BatchVecmatOp result shape to GenericOp result shape
  SmallVector<int64_t> outPerm{
      *(outMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.batch[0]))),
      *(outMap.getResultPosition(
          rewriter.getAffineDimExpr(contractionDims.n[0])))};
  outPerm = getInverseTransposePerm(outPerm);
  return liftGenericOp<linalg::BatchVecmatOp>(rewriter, genericOp, vecPerm,
                                              matPerm, outPerm);
}

static LogicalResult
raiseToBatchMatvec(PatternRewriter &rewriter, linalg::GenericOp genericOp,
                   linalg::ContractionDimensions contractionDims) {
  LLVM_DEBUG(llvm::dbgs() << "Not implemented\n\n");
  return failure();
}

static LogicalResult
raiseToBatchMatmul(PatternRewriter &rewriter, linalg::GenericOp genericOp,
                   linalg::ContractionDimensions contractionDims) {
  LLVM_DEBUG(llvm::dbgs() << "Not implemented\n\n");
  return failure();
}

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
class LiftGenericToTransposeBatchMatmul
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "LiftGenericToTransposeBatchMatmul on "
                            << genericOp << "\n\n\n");

    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(genericOp);
    if (failed(contractionDims)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to infer contraction dims\n\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "batch: " << contractionDims->batch << "\n");
    LLVM_DEBUG(llvm::dbgs() << "m: " << contractionDims->m << "\n");
    LLVM_DEBUG(llvm::dbgs() << "n: " << contractionDims->n << "\n");
    LLVM_DEBUG(llvm::dbgs() << "k: " << contractionDims->k << "\n");

    auto lhsType =
        llvm::dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    auto rhsType =
        llvm::dyn_cast<RankedTensorType>(genericOp.getInputs()[1].getType());
    auto outType =
        llvm::dyn_cast<RankedTensorType>(genericOp.getResults()[0].getType());
    if (!lhsType || !rhsType || !outType) {
      LLVM_DEBUG(llvm::dbgs() << "Inputs do not have RankedTensorType\n\n");
      return failure();
    }

    if (!hasMatmulBody(genericOp)) {
      LLVM_DEBUG(llvm::dbgs() << "genericOp does not have a matmul body\n\n");
      return failure();
    }

    if (contractionDims->batch.size() == 1 && contractionDims->m.size() == 0 &&
        contractionDims->n.size() == 1 && contractionDims->k.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Lifting to linalg.batch_vecmat\n");
      return raiseToBatchVecmat(rewriter, genericOp, *contractionDims);
    } else if (contractionDims->batch.size() == 1 &&
               contractionDims->m.size() == 1 &&
               contractionDims->n.size() == 0 &&
               contractionDims->k.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Lifting to linalg.batch_matvec\n");
      return raiseToBatchMatvec(rewriter, genericOp, *contractionDims);
    } else if (contractionDims->batch.size() == 1 &&
               contractionDims->m.size() == 1 &&
               contractionDims->n.size() == 1 &&
               contractionDims->k.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Lifting to linalg.batch_matmul\n");
      return raiseToBatchMatmul(rewriter, genericOp, *contractionDims);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Did not match any batch case\n\n");
      return failure();
    }
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

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
