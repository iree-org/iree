// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler {

namespace {

using DimBoundSize = vector::ConstantOrScalableBound::BoundSize;

struct VectorizationTileSizes {
  SmallVector<int64_t> destShape;
  SmallVector<int64_t> vectorSizes;
  SmallVector<bool> vectorScalableFlags;
};

/// Computes the upper bound of `operandDim` for the ShapedType `operand`. If
/// the optional `vscaleRange` is provided then the computed bound can be a
/// scalable quantity.
static FailureOr<DimBoundSize>
computeDimUpperBound(Value operand, unsigned operandDim,
                     std::optional<VscaleRange> vscaleRange) {
  if (!vscaleRange.has_value()) {
    auto maybeDimBoundSize = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, {operand, operandDim},
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (succeeded(maybeDimBoundSize))
      return DimBoundSize{.baseSize = *maybeDimBoundSize, .scalable = false};
    return failure();
  }
  auto maybeDimBound =
      vector::ScalableValueBoundsConstraintSet::computeScalableBound(
          operand, operandDim,
          /*vscaleMin=*/vscaleRange->min,
          /*vscaleMax=*/vscaleRange->max, presburger::BoundType::UB);
  if (succeeded(maybeDimBound))
    return maybeDimBound->getSize();
  return failure();
}

/// Returns a VectorizationTileSizes which contains the inferred bounded result
/// shape and vector input sizes. This is useful to infer the sizes from a
/// chain.
static std::optional<VectorizationTileSizes> inferSizesFromIR(Value val);

/// Tries to infer the vector sizes from an IR using ValueBounds analysis. If
/// `opResult` is provided, it stores the bounded result shapes to destShape.
/// Returns std::nullopt if vector sizes can't be inferred.
static std::optional<VectorizationTileSizes>
inferSizesFromIR(linalg::LinalgOp linalgOp, std::optional<OpResult> opResult) {
  LLVM_DEBUG({
    VEC_DBGS() << "Inferring sizes for:\n" << linalgOp;
    if (opResult) {
      VEC_DBGS() << " with OpResult.resultNumber="
                 << opResult->getResultNumber();
    }
    VEC_DBGS() << '\n';
  });

  std::optional<VscaleRange> vscaleRange;
  if (!opResult) {
    // Note: Inferring scalable sizes is not supported is `opResult` is set
    // (which is used to compute sizes for tensor.pack/unpack).
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(linalgOp);
    vscaleRange = getDefaultVscaleRange(targetAttr);
  }

  VectorizationTileSizes result;
  unsigned numDims = linalgOp.getNumLoops();
  for (int dim = 0; dim < numDims; ++dim) {
    // Map dimension `dim` to an operand dimension that we will use to
    // traverse the U-D chain to get `dim` vector size information.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return std::nullopt;
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    bool dimScalable = false;
    if (!ShapedType::isDynamic(dimSize)) {
      result.vectorSizes.push_back(dimSize);
      result.vectorScalableFlags.push_back(dimScalable);
      LLVM_DEBUG(VEC_DBGS() << "Inferred iteration size '" << dimSize
                            << "' for dimension '" << dim << "'\n");
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<DimBoundSize> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = computeDimUpperBound(operand, operandDim, vscaleRange);
      if (succeeded(maybeDimBound)) {
        break;
      }
    }

    if (failed(maybeDimBound)) {
      return std::nullopt;
    }

    dimSize = maybeDimBound->baseSize;
    dimScalable = maybeDimBound->scalable;
    result.vectorSizes.push_back(dimSize);
    result.vectorScalableFlags.push_back(dimScalable);

    LLVM_DEBUG(VEC_DBGS() << "Inferred iteration size '" << dimSize
                          << (dimScalable ? " x vscale" : "")
                          << "' for dimension '" << dim << "'\n");
  }

  if (opResult) {
    assert(!llvm::is_contained(result.vectorScalableFlags, true) &&
           "inferring scalable bounds with `opResult` not supported!");
    result.destShape = linalgOp.getIndexingMapMatchingResult(opResult.value())
                           .compose(result.vectorSizes);
  }

  return result;
}

/// Returns the result sizes and vector input sizes of the tensor.pack op. The
/// inferred bounding size is returned if it is dynamic shape. Returns
/// std::nullopt if the shape inference failed.
static std::optional<VectorizationTileSizes>
inferSizesFromIR(tensor::PackOp op) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring dest sizes for:\n" << op << "\n");

  if (llvm::any_of(op.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LLVM_DEBUG(VEC_DBGS() << "skip, because inner_tiles are not all constant");
    return std::nullopt;
  }

  VectorizationTileSizes result;
  std::optional<VectorizationTileSizes> inferred =
      inferSizesFromIR(op.getSource());
  if (!inferred) {
    return std::nullopt;
  }
  result.vectorSizes = inferred.value().destShape;

  for (auto [dimPos, tileSize] :
       llvm::zip_equal(op.getInnerDimsPos(), op.getStaticInnerTiles())) {
    if (result.vectorSizes[dimPos] % tileSize != 0) {
      return std::nullopt;
    }
    result.vectorSizes[dimPos] /= tileSize;
  }
  auto outerDimsPerm = op.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(result.vectorSizes, outerDimsPerm);
  }

  LLVM_DEBUG({
    VEC_DBGS() << "After adjustment with inner tiles and "
                  "outer_dims_perm:\n";
    for (auto [idx, val] : llvm::enumerate(result.vectorSizes)) {
      llvm::dbgs() << "Dim #" << idx << ": " << val << "\n";
    }
  });
  result.destShape = result.vectorSizes;

  return result;
}

/// Returns the result sizes and vector input sizes of the tensor.unpack op. The
/// inferred bounding size is returned if it is dynamic shape. Returns
/// std::nullopt if the shape inference failed.
static std::optional<VectorizationTileSizes>
inferSizesFromIR(tensor::UnPackOp op) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring dest sizes for:\n" << op << "\n");

  if (llvm::any_of(op.getInnerTiles(), [](OpFoldResult v) {
        return !getConstantIntValue(v).has_value();
      })) {
    LLVM_DEBUG(
        VEC_DBGS()
        << "failed on inference because inner_tiles are not all constant");
    return std::nullopt;
  }

  VectorizationTileSizes result;
  std::optional<VectorizationTileSizes> inferred =
      inferSizesFromIR(op.getSource());
  if (!inferred) {
    return std::nullopt;
  }
  result.vectorSizes = inferred.value().destShape;

  result.vectorSizes.resize(op.getDestType().getRank());
  auto outerDimsPerm = op.getOuterDimsPerm();
  if (!outerDimsPerm.empty()) {
    applyPermutationToVector(result.vectorSizes,
                             invertPermutationVector(outerDimsPerm));
  }
  for (auto [dimPos, tileSize] :
       llvm::zip_equal(op.getInnerDimsPos(), op.getStaticInnerTiles())) {
    result.vectorSizes[dimPos] *= tileSize;
  }

  LLVM_DEBUG({
    VEC_DBGS() << "After adjustment with inner tiles and "
                  "outer_dims_perm:\n";
    for (auto [idx, val] : llvm::enumerate(result.vectorSizes)) {
      llvm::dbgs() << "Dim #" << idx << ": " << val << "\n";
    }
  });
  result.destShape = result.vectorSizes;

  return result;
}

/// See the documentation in the above function declaration.
static std::optional<VectorizationTileSizes> inferSizesFromIR(Value val) {
  std::optional<VectorizationTileSizes> result;
  TypeSwitch<Operation *, void>(val.getDefiningOp())
      .Case<linalg::LinalgOp>(
          [&](auto op) { result = inferSizesFromIR(op, cast<OpResult>(val)); })
      .Case<tensor::PackOp>([&](auto op) { result = inferSizesFromIR(op); })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp op) {
        // tensor::ExtractSliceOp is not vectorizable, so only `destShape` has
        // the values.
        result = VectorizationTileSizes();
        LLVM_DEBUG(VEC_DBGS() << "Inferring sizes for:\n" << op << "\n");
        int64_t destRank = op.getResult().getType().getRank();
        for (int dim = 0; dim < destRank; ++dim) {
          LLVM_DEBUG(VEC_DBGS() << "Dim #" << dim << ": ");
          FailureOr<int64_t> maybeDimBound =
              ValueBoundsConstraintSet::computeConstantBound(
                  presburger::BoundType::UB, {op, dim},
                  /*stopCondition=*/nullptr, /*closedUB=*/true);
          if (failed(maybeDimBound)) {
            LLVM_DEBUG(llvm::dbgs() << "failed\n");
            result = std::nullopt;
            return;
          }
          LLVM_DEBUG(llvm::dbgs() << maybeDimBound.value() << "\n");
          result->destShape.push_back(maybeDimBound.value());
        }
      })
      .Default([&](Operation *) {});
  return result;
}

// Returns the vector sizes from the local lowering config or try to infer them
// from the tensor shapes and tiled loops in the IR.
static std::optional<SizesAndScalableFlags>
getVectorSizes(Operation *op, bool useConfiguredVectorSizes) {
  // Get vector sizes from the lowering config, if available in the op itself.
  auto loweringConfig =
      getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(op);
  if (useConfiguredVectorSizes && loweringConfig) {
    TilingConfig tilingConfig(loweringConfig);
    auto [vectorSizes, scalableFlags] = tilingConfig.getVectorTileSizes();
    // Replace zeros in canonical vector shape to turn it into a valid shape.
    std::replace(vectorSizes.begin(), vectorSizes.end(), 0, 1);
    return std::make_pair(vectorSizes, scalableFlags);
  }

  // Try to infer the vector sizes from the IR.
  std::optional<SmallVector<int64_t>> vectorSizes;
  SmallVector<bool> scalableFlags;
  TypeSwitch<Operation *, void>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        std::optional<VectorizationTileSizes> result =
            inferSizesFromIR(linalgOp, /*opResult=*/std::nullopt);
        if (result) {
          vectorSizes = result->vectorSizes;
          scalableFlags = result->vectorScalableFlags;
        }
      })
      .Case<tensor::PackOp, tensor::UnPackOp>([&](auto op) {
        std::optional<VectorizationTileSizes> result = inferSizesFromIR(op);
        if (result) {
          vectorSizes = result->vectorSizes;
        }
      })
      .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
        auto ty = padOp.getResultType();
        // TODO(hanchung): Infer the vector sizes for pad op after
        // maskedVectorize method allows dynamic result shapes.
        if (!ty.hasStaticShape())
          return;
        vectorSizes = SmallVector<int64_t>(ty.getShape());
      })
      .Default([&](Operation *) {});

  if (vectorSizes) {
    scalableFlags.resize(vectorSizes->size(), false);
    return std::make_pair(vectorSizes.value(), scalableFlags);
  }
  return std::nullopt;
}

static LogicalResult isWithinVectorSizeLimit(linalg::LinalgOp linalgOp,
                                             int64_t maxVectorSize) {
  int64_t maxFlatVecSize = 1;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    auto type = llvm::dyn_cast<ShapedType>(operand.get().getType());
    if (!type)
      continue;
    if (!type.hasStaticShape())
      return failure();
    maxFlatVecSize = std::max(maxFlatVecSize, type.getNumElements());
  }
  return success(maxFlatVecSize < maxVectorSize);
}

class GenericVectorizationPass
    : public GenericVectorizationBase<GenericVectorizationPass> {
public:
  using GenericVectorizationBase::GenericVectorizationBase;
  GenericVectorizationPass(const GenericVectorizationPassOptions &options) {
    this->enableVectorMasking.setValue(options.enableVectorMasking);
    this->useConfiguredVectorSizes.setValue(options.useConfiguredVectorSizes);
    this->vectorizePadding.setValue(options.vectorizePadding);
    this->vectorizeGatherAccesses.setValue(options.vectorizeGatherAccesses);
    this->enableCleanup.setValue(options.enableCleanup);
    this->generateContract.setValue(options.generateContract);
    this->foldCastIntoContract.setValue(options.foldCastIntoContract);
    this->maxVectorSize.setValue(options.maxVectorSize);
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

  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op)) {
      candidates.push_back(op);
    } else if (vectorizePadding && enableVectorMasking &&
               isa<tensor::PadOp>(op)) {
      candidates.push_back(op);
    } else if (enableVectorMasking &&
               isa<tensor::PackOp, tensor::UnPackOp>(op)) {
      candidates.push_back(op);
    }
  });

  // The vector input sizes inference needs to use producers, so we apply
  // vectorization from bottom to top.
  std::reverse(candidates.begin(), candidates.end());
  for (Operation *op : candidates) {
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableVecDims;
    if (enableVectorMasking) {
      std::optional<SizesAndScalableFlags> vectorSizesAndScalableDims =
          getVectorSizes(op, useConfiguredVectorSizes);
      if (vectorSizesAndScalableDims) {
        auto [sizes, scalableDims] = *vectorSizesAndScalableDims;
        vectorSizes.append(sizes.begin(), sizes.end());
        scalableVecDims.append(scalableDims.begin(), scalableDims.end());
      }
    }

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not vectorize the op if the vector size is greater than or equal
      // to limit.
      if (enableVectorMasking) {
        if (std::accumulate(vectorSizes.begin(), vectorSizes.end(), 1,
                            std::multiplies<int64_t>()) >= maxVectorSize)
          continue;
      } else {
        if (failed(isWithinVectorSizeLimit(linalgOp, maxVectorSize)))
          continue;
      }
    }
    // Pad scalable dims with `false` to match the vector sizes.
    scalableVecDims.resize(vectorSizes.size());
    (void)linalg::vectorize(rewriter, op, vectorSizes, scalableVecDims,
                            vectorizeGatherAccesses);
  };

  {
    // Canonicalize mask related ops before we lower them.
    RewritePatternSet maskCanonPatterns(funcOp.getContext());
    vector::CreateMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                      funcOp.getContext());
    vector::ConstantMaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                        funcOp.getContext());
    vector::MaskOp::getCanonicalizationPatterns(maskCanonPatterns,
                                                funcOp.getContext());
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(maskCanonPatterns)))) {
      return signalPassFailure();
    }
  }

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  if (generateContract) {
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  }
  if (foldCastIntoContract) {
    vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  }
  if (enableVectorMasking) {
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
  }

  if (enableCleanup) {
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
  }
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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGenericVectorizationPass() {
  return std::make_unique<GenericVectorizationPass>();
}
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGenericVectorizationPass(const GenericVectorizationPassOptions &options) {
  return std::make_unique<GenericVectorizationPass>(options);
}

} // namespace mlir::iree_compiler
