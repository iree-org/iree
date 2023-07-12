// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace {

/// Returns the op that contains lowering config. Checks whether the provided op
/// contains the lowering config and returns it. Otherwise, tries to find the
/// lowering config across the function. If there are multiple ops with the same
/// lowering configs, returns the first one found. Returns failure if there are
/// multiple op with different lowering config.
static FailureOr<Operation *> getRootOp(Operation *op) {
  // Check for self first.
  if (iree_compiler::getLoweringConfig(op)) {
    return op;
  }

  // Get the function op.
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    funcOp = op->getParentOfType<func::FuncOp>();
  }

  assert(funcOp && "Missing funcOp");

  Operation *rootOp = nullptr;
  mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr rootLoweringConfig;
  auto result = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig = iree_compiler::getLoweringConfig(op);
    if (!loweringConfig) {
      return WalkResult::advance();
    }
    if (rootLoweringConfig) {
      if (rootLoweringConfig != loweringConfig) {
        return WalkResult::interrupt();
      }
    } else {
      rootOp = op;
      rootLoweringConfig = loweringConfig;
    }
    return WalkResult::advance();
  });

  if (!rootOp || result.wasInterrupted()) {
    return failure();
  }
  return rootOp;
}

/// Computes the canonical shape used to vectorize this dispatch. Retrieves
/// the vectorization tile sizes (parallel and reduction levels) out of the
/// lowering config and adjusts them to the format expected by the Linalg
/// vectorizer.
static SmallVector<int64_t> getCanonicalVectorShape(func::FuncOp funcOp) {
  FailureOr<Operation *> rootOp = getRootOp(funcOp);
  if (failed(rootOp)) {
    return {};
  }

  unsigned numTileLevels =
      mlir::iree_compiler::getNumTileLevels(rootOp.value());
  if (numTileLevels < 3) {
    return {};
  }

  // Retrieve the tile sizes from the last two tiling levels (parallel and
  // reduction) used for vectorization.
  SmallVector<int64_t> canonicalVectorShape =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 2);
  SmallVector<int64_t> reductionTileSizes =
      mlir::iree_compiler::getTileSizes(rootOp.value(), numTileLevels - 1);

  if (!reductionTileSizes.empty()) {
    assert(canonicalVectorShape.size() == reductionTileSizes.size() &&
           "Unexpected tile sizes");

    // Combine the reduction tile sizes with the parallel tile sizes already in
    // the canonical vector shape.
    for (int i = 0, end = canonicalVectorShape.size(); i < end; ++i) {
      if (reductionTileSizes[i] > 0)
        canonicalVectorShape[i] = reductionTileSizes[i];
    }
  }

  // Replace zeros in canonical vector shape to turn it into a valid shape.
  std::replace(canonicalVectorShape.begin(), canonicalVectorShape.end(), 0, 1);
  return canonicalVectorShape;
}

/// Tries to infer the vector sizes from an IR using ValueBounds analysis.
/// Returns failure if vector sizes can't be inferred.
static FailureOr<SmallVector<int64_t>>
inferVectorSizesFromIR(linalg::LinalgOp linalgOp) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring vector sizes for:\n" << linalgOp << "\n");

  SmallVector<int64_t> vectorSizes;
  unsigned numDims = linalgOp.getNumLoops();

  for (int dim = 0; dim < numDims; ++dim) {
    // Map dimension `dim` to an operand dimension that we will use to
    // traverse the U-D chain to get `dim` vector size information.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return failure();
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    if (!ShapedType::isDynamic(dimSize)) {
      vectorSizes.push_back(dimSize);
      LLVM_DEBUG(VEC_DBGS() << "Inferred vector size '" << dimSize
                            << "' for dimension '" << dim << "'\n");
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<int64_t> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand, operandDim,
          /*stopCondition=*/nullptr, /*closedUB=*/true);

      if (succeeded(maybeDimBound)) {
        break;
      }
    }

    if (failed(maybeDimBound)) {
      return failure();
    }

    dimSize = maybeDimBound.value();
    vectorSizes.push_back(dimSize);
    LLVM_DEBUG(VEC_DBGS() << "Inferred vector size '" << dimSize
                          << "' for dimension '" << dim << "'\n");
  }

  return vectorSizes;
}

// Give the canonical vector shape of a dispatch, returns the vector sizes for a
// particular linalg op within that dispatch.
static SmallVector<int64_t>
getVectorSizes(linalg::LinalgOp linalgOp,
               ArrayRef<int64_t> canonicalVectorShape) {
  // Try to infer the vector sizes from the IR. If it fails, try to get them
  // from the lowering config.
  auto inferredVectorSizes = inferVectorSizesFromIR(linalgOp);
  if (succeeded(inferredVectorSizes)) {
    return *inferredVectorSizes;
  }

  FailureOr<Operation *> rootOp = getRootOp(linalgOp);
  if (failed(rootOp)) {
    return {};
  }

  // TODO: Infer the tiles sizes for an op that is not the root op.
  if (*rootOp != linalgOp.getOperation()) {
    return {};
  }

  if (canonicalVectorShape.empty()) {
    return {};
  }

  assert(canonicalVectorShape.size() >= linalgOp.getNumLoops() &&
         "Unexpected canonical vector shape or number of loops");

  // Return the valid canonical vector shape subset based on the number of loops
  // of the linalg op.
  SmallVector<int64_t> vecSize(
      canonicalVectorShape.take_front(linalgOp.getNumLoops()));
  for (auto [idx, val] : llvm::enumerate(linalgOp.getStaticLoopRanges())) {
    if (ShapedType::isDynamic(val))
      continue;
    vecSize[idx] = std::max(vecSize[idx], val);
  }

  return vecSize;
}

/// Returns true if all rank reduced in the given `extractOp` happen in leading
/// dimensions earlier than last `trailingRank` dimensions.
static bool areAllRankReducedLeadingDim(tensor::ExtractSliceOp extractOp,
                                        unsigned trailingRank) {
  // If no ranks are reduced at all, it's a degenerated case; always true.
  if (extractOp.getSourceType().getRank() == extractOp.getType().getRank())
    return true;

  RankedTensorType inferredType = extractOp.inferResultType(
      extractOp.getSourceType(), extractOp.getMixedOffsets(),
      extractOp.getMixedSizes(), extractOp.getMixedStrides());
  return extractOp.getType().getShape().take_back(trailingRank) ==
         inferredType.getShape().take_back(trailingRank);
}

namespace {
/// Fold transfer_reads of a tensor.extract_slice op. E.g.:
///
/// ```
/// %0 = tensor.extract_slice %t[%a, %b] [%c, %d] [1, 1]
///     : tensor<?x?xf32> to tensor<?x?xf32>
/// %1 = vector.transfer_read %0[%e, %f], %cst {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<4x5xf32>
/// ```
/// is rewritten to:
/// ```
/// %p0 = arith.addi %a, %e : index
/// %p1 = arith.addi %b, %f : index
/// %1 = vector.transfer_read %t[%p0, %p1], %cst {in_bounds = [true, true]}
///     : tensor<?x?xf32>, vector<4x5xf32>
/// ```
// TODO: this is brittle and should be deprecated in favor of a more general
// pattern that applies on-demand.
class FoldExtractSliceIntoTransferRead final
    : public OpRewritePattern<vector::TransferReadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();
    if (xferOp.hasOutOfBoundsDim())
      return failure();
    if (!xferOp.getPermutationMap().isMinorIdentity())
      return failure();
    if (xferOp.getMask())
      return failure();
    auto extractOp = xferOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();
    if (!extractOp.hasUnitStride())
      return failure();

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = tensor.extract_slice %t[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x1x4xf32> to tensor<2x4xf32>
    //    %1 = vector.transfer_read %0[0,0], %cst :
    //      tensor<2x4xf32>, vector<2x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_read %t[0,0,0], %cst :
    //      tensor<2x1x4xf32>, vector<2x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the extract_slice
    // result tensor match the trailing dims of the inferred result tensor.
    if (!areAllRankReducedLeadingDim(extractOp, extractOp.getType().getRank()))
      return failure();

    int64_t rankReduced =
        extractOp.getSourceType().getRank() - extractOp.getType().getRank();

    SmallVector<Value> newIndices;
    // In case this is a rank-reducing ExtractSliceOp, copy rank-reduced
    // indices first.
    for (int64_t i = 0; i < rankReduced; ++i) {
      OpFoldResult offset = extractOp.getMixedOffsets()[i];
      newIndices.push_back(getValueOrCreateConstantIndexOp(
          rewriter, extractOp.getLoc(), offset));
    }
    for (const auto &it : llvm::enumerate(xferOp.getIndices())) {
      OpFoldResult offset =
          extractOp.getMixedOffsets()[it.index() + rankReduced];
      newIndices.push_back(rewriter.create<arith::AddIOp>(
          xferOp->getLoc(), it.value(),
          getValueOrCreateConstantIndexOp(rewriter, extractOp.getLoc(),
                                          offset)));
    }
    SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        xferOp, xferOp.getVectorType(), extractOp.getSource(), newIndices,
        xferOp.getPadding(), ArrayRef<bool>{inBounds});

    return success();
  }
};

/// Fold tensor.insert_slice into vector.transfer_write if the transfer_write
/// could directly write to the insert_slice's destination. E.g.:
///
/// ```
/// %0 = vector.transfer_write %v, %t1[%c0, %c0] {in_bounds = [true, true]}
///     : vector<4x5xf32>, tensor<4x5xf32>
/// %1 = tensor.insert_slice %0 into %t2[%a, %b] [4, 5] [1, 1]
///     : tensor<4x5xf32> into tensor<?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %1 = vector.transfer_write %v, %t2[%a, %b] {in_bounds = [true, true]}
///     : vector<4x5xf32>, tensor<?x?xf32>
/// ```
// TODO: this is brittle and should be deprecated in favor of a more general
// pattern that applies on-demand.
class FoldInsertSliceIntoTransferWrite final
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (!insertOp.hasUnitStride())
      return failure();

    auto xferOp = insertOp.getSource().getDefiningOp<vector::TransferWriteOp>();
    if (!xferOp)
      return failure();

    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();

    if (xferOp.hasOutOfBoundsDim())
      return failure();
    if (xferOp.getVectorType().getRank() != xferOp.getShapedType().getRank())
      return failure();
    if (xferOp.getMask())
      return failure();
    // Fold only if the TransferWriteOp completely overwrites the `source` with
    // a vector. I.e., the result of the TransferWriteOp is a new tensor whose
    // content is the data of the vector.
    if (!llvm::equal(xferOp.getVectorType().getShape(),
                     xferOp.getShapedType().getShape()))
      return failure();
    if (!xferOp.getPermutationMap().isIdentity())
      return failure();

    // Bail on illegal rank-reduction: we need to check that the rank-reduced
    // dims are exactly the leading dims. I.e. the following is illegal:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0], %cst :
    //      vector<2x4xf32>, tensor<2x4xf32>
    //    %1 = tensor.insert_slice %0 into %tt[0,0,0][2,1,4][1,1,1] :
    //      tensor<2x4xf32> into tensor<2x1x4xf32>
    // ```
    //
    // Cannot fold into:
    // ```
    //    %0 = vector.transfer_write %v, %t[0,0,0], %cst :
    //      vector<2x4xf32>, tensor<2x1x4xf32>
    // ```
    // For this, check the trailing `vectorRank` dims of the insert_slice result
    // tensor match the trailing dims of the inferred result tensor.
    int64_t rankReduced =
        insertOp.getType().getRank() - insertOp.getSourceType().getRank();
    int64_t vectorRank = xferOp.getVectorType().getRank();
    RankedTensorType inferredSourceTensorType =
        tensor::ExtractSliceOp::inferResultType(
            insertOp.getType(), insertOp.getMixedOffsets(),
            insertOp.getMixedSizes(), insertOp.getMixedStrides());
    auto actualSourceTensorShape = insertOp.getSourceType().getShape();
    if (rankReduced > 0 &&
        actualSourceTensorShape.take_back(vectorRank) !=
            inferredSourceTensorType.getShape().take_back(vectorRank))
      return failure();

    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, insertOp.getLoc(), insertOp.getMixedOffsets());
    SmallVector<bool> inBounds(xferOp.getTransferRank(), true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        insertOp, xferOp.getVector(), insertOp.getDest(), indices,
        ArrayRef<bool>{inBounds});
    return success();
  }
};

} // namespace

class GenericVectorizationPass
    : public GenericVectorizationBase<GenericVectorizationPass> {
public:
  using GenericVectorizationBase::GenericVectorizationBase;
  GenericVectorizationPass(const GenericVectorizationPassOptions &options) {
    this->enableVectorMasking.setValue(options.enableVectorMasking);
    this->vectorizePadding.setValue(options.vectorizePadding);
    this->vectorizeGatherAccesses.setValue(options.vectorizeGatherAccesses);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  SmallVector<int64_t> canonicalVectorShape;
  if (enableVectorMasking) {
    canonicalVectorShape = getCanonicalVectorShape(funcOp);
  }

  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op))
      candidates.push_back(op);
    if (vectorizePadding && enableVectorMasking && isa<tensor::PadOp>(op))
      candidates.push_back(op);
  });
  for (auto op : candidates) {
    SmallVector<int64_t> vectorSizes;
    if (enableVectorMasking) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        vectorSizes.append(getVectorSizes(linalgOp, canonicalVectorShape));
      } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
        auto ty = padOp.getResultType();
        // TODO(hanchung): Infer the vector sizes for pad op after
        // maskedVectorize method allows dynamic result shapes.
        if (!ty.hasStaticShape())
          continue;
        vectorSizes.append(ty.getShape().begin(), ty.getShape().end());
      }
    }
    SmallVector<bool> scalableVecDims(vectorSizes.size(), false);
    (void)linalg::vectorize(rewriter, op, vectorSizes, scalableVecDims,
                            vectorizeGatherAccesses);
  };

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
      vectorizationPatterns);
  vector::populateVectorTransferPermutationMapLoweringPatterns(
      vectorizationPatterns);
  vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                            linalg::LinalgCopyVTWForwardingPattern>(
      funcOp.getContext(), /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                      funcOp.getContext());
  vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                       funcOp.getContext());
  vectorizationPatterns
      .add<FoldExtractSliceIntoTransferRead, FoldInsertSliceIntoTransferWrite>(
          funcOp.getContext());
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  // Apply the pad tensor op vectorization separately to avoid running the
  // GenericPadOpVectorizationPattern too early.
  // TODO: Improve once we have better infrastructure to control pattern
  // application.
  if (vectorizePadding) {
    RewritePatternSet patterns(funcOp.getContext());
    linalg::populatePadOpVectorizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGenericVectorizationPass() {
  return std::make_unique<GenericVectorizationPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>>
createGenericVectorizationPass(const GenericVectorizationPassOptions &options) {
  return std::make_unique<GenericVectorizationPass>(options);
}
} // namespace iree_compiler
} // namespace mlir
