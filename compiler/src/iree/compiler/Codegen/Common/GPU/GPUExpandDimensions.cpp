// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-expand-dimensions"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUEXPANDDIMENSIONSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct RemoveOptimizationBarrier final
    : public OpRewritePattern<IREE::Util::OptimizationBarrierOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::Util::OptimizationBarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(barrierOp, barrierOp.getOperands());
    return success();
  }
};

struct GPUExpandDimensionsPass final
    : impl::GPUExpandDimensionsPassBase<GPUExpandDimensionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Map from tensor dimension index to thread level tile size.
using DimensionExpansionInfo = llvm::SmallDenseMap<unsigned, int64_t>;

/// Parse the expand_dims and thread tiling level to compute expansion
/// information. expand_dims format: [[0], [1], [2,3]] means dim 0→0, dim 1→1,
/// dim 2→[2,3].
static DimensionExpansionInfo
getExpansionInfo(IREE::GPU::LoweringConfigAttr config) {
  // Get expand_dims structure
  SmallVector<SmallVector<int64_t>> expansionFactors =
      IREE::GPU::getDimensionExpansion(config).value();
  SmallVector<int64_t> threadSizes = config.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Thread), /*opIdx=*/0);

  DimensionExpansionInfo expansionInfo;

  for (auto [origDimIdx, newDimIndices] : llvm::enumerate(expansionFactors)) {
    if (newDimIndices.size() > 1) {
      for (unsigned newDimIdx : newDimIndices) {
        if (newDimIdx < threadSizes.size()) {
          int64_t factor = threadSizes[newDimIdx];
          if (factor > 1) {
            expansionInfo[origDimIdx] = factor;
          }
        }
      }
    }
  }

  for (auto [dim, factor] : expansionInfo) {
    LLVM_DEBUG({
      llvm::dbgs() << "Dimension " << dim << " will be expanded by factor "
                   << factor << "\n";
    });
  }

  return expansionInfo;
}

/// Helper struct to hold the expand/collapse shape ops created for dimension
/// expansion.
struct ReshapeOps {
  tensor::ExpandShapeOp expandShapeOp;
  tensor::CollapseShapeOp collapseShapeOp;
};

/// For a value `v`, expand dimensions according to `expansionInfo` by
/// inserting:
///
/// ```mlir
/// %v_expand = tensor.expand_shape %v
/// %barrier = util.optimization.barrier %v_expand
/// %v_collapse = tensor.collapse_shape %barrier
/// ```
///
/// where the generated `tensor.expand_shape` and `tensor.collapse_shape` are
/// inverses of each other. The `util.optimization.barrier` prevents these from
/// getting folded away during reshape propagation.
static std::optional<ReshapeOps>
expandIterationSpaceOfValue(RewriterBase &rewriter,
                            const DimensionExpansionInfo &expansionInfo,
                            Value v) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  SmallVector<OpFoldResult> outputShape;
  SmallVector<ReassociationIndices> reassociation;
  Location loc = v.getLoc();
  SmallVector<OpFoldResult> origShape = tensor::getMixedSizes(rewriter, loc, v);

  for (auto [index, dim] : llvm::enumerate(origShape)) {
    reassociation.emplace_back(ReassociationIndices{});

    if (!expansionInfo.contains(index)) {
      reassociation.back().push_back(outputShape.size());
      outputShape.push_back(dim);
      continue;
    }

    int64_t factor = expansionInfo.lookup(index);
    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    AffineExpr divExpr = s0.floorDiv(factor);
    OpFoldResult newOuterDim = affine::makeComposedFoldedAffineApply(
        rewriter, loc, divExpr, ArrayRef<OpFoldResult>{dim});
    OpFoldResult newInnerDim = rewriter.getIndexAttr(factor);

    reassociation.back().push_back(outputShape.size());
    reassociation.back().push_back(outputShape.size() + 1);

    outputShape.push_back(newOuterDim);
    outputShape.push_back(newInnerDim);
  }

  auto staticOutputShape =
      llvm::map_to_vector(outputShape, [](OpFoldResult ofr) {
        if (auto staticShapeAttr = dyn_cast<Attribute>(ofr)) {
          return cast<IntegerAttr>(staticShapeAttr).getInt();
        }
        return ShapedType::kDynamic;
      });
  auto outputType = RankedTensorType::get(
      staticOutputShape, tensorType.getElementType(), tensorType.getEncoding());

  auto expandShapeOp = tensor::ExpandShapeOp::create(
      rewriter, loc, outputType, v, reassociation, outputShape);
  Value barrier = IREE::Util::OptimizationBarrierOp::create(
                      rewriter, loc, expandShapeOp.getResult())
                      .getResult(0);
  auto collapseShapeOp = tensor::CollapseShapeOp::create(
      rewriter, loc, tensorType, barrier, reassociation);
  return ReshapeOps{expandShapeOp, collapseShapeOp};
}

static LogicalResult expandIterationSpace(RewriterBase &rewriter,
                                          linalg::GenericOp genericOp) {
  auto loweringConfig = getLoweringConfig(genericOp);
  if (!loweringConfig)
    return success();

  auto gpuConfig = dyn_cast<IREE::GPU::LoweringConfigAttr>(loweringConfig);
  if (!gpuConfig)
    return success();

  if (failed(IREE::GPU::getDimensionExpansion(gpuConfig)))
    return success();

  DimensionExpansionInfo expansionInfo = getExpansionInfo(gpuConfig);
  if (expansionInfo.empty())
    return success();

  LLVM_DEBUG({
    llvm::dbgs() << "Expanding dimensions for op:\n";
    genericOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  // Expand operands
  // expansionInfo maps iteration space dims → factor
  // We need to translate to tensor dims for each operand based on indexing map
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();

  for (OpOperand &operand : genericOp->getOpOperands()) {
    if (operand.get().getDefiningOp<tensor::CollapseShapeOp>())
      continue;
    if (!isa<RankedTensorType>(operand.get().getType()))
      continue;

    // Translate iteration space expansion to tensor space expansion
    AffineMap indexingMap = indexingMaps[operand.getOperandNumber()];
    DimensionExpansionInfo tensorExpansionInfo;

    for (auto [iterDim, factor] : expansionInfo) {
      AffineExpr iterExpr = getAffineDimExpr(iterDim, genericOp.getContext());
      if (std::optional<unsigned> tensorDim =
              indexingMap.getResultPosition(iterExpr)) {
        tensorExpansionInfo[tensorDim.value()] = factor;
      }
    }

    if (tensorExpansionInfo.empty())
      continue;

    std::optional<ReshapeOps> reshapes = expandIterationSpaceOfValue(
        rewriter, tensorExpansionInfo, operand.get());
    if (reshapes) {
      rewriter.modifyOpInPlace(
          genericOp, [&]() { operand.set(reshapes->collapseShapeOp); });
    }
  }

  // Expand results.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(genericOp);

  for (OpResult result : genericOp->getResults()) {
    if (!isa<RankedTensorType>(result.getType()))
      continue;

    unsigned resultMapIndex =
        genericOp.getNumDpsInputs() + result.getResultNumber();
    AffineMap indexingMap = indexingMaps[resultMapIndex];
    DimensionExpansionInfo tensorExpansionInfo;

    for (auto [iterDim, factor] : expansionInfo) {
      AffineExpr iterExpr = getAffineDimExpr(iterDim, genericOp.getContext());
      if (std::optional<unsigned> tensorDim =
              indexingMap.getResultPosition(iterExpr)) {
        tensorExpansionInfo[tensorDim.value()] = factor;
      }
    }

    if (tensorExpansionInfo.empty())
      continue;

    std::optional<ReshapeOps> reshapes =
        expandIterationSpaceOfValue(rewriter, tensorExpansionInfo, result);
    if (reshapes) {
      // Replace uses of the result with the collapse_shape, but exclude:
      // - The expand_shape operation itself (it uses the result as input).
      // - tensor.dim operations (they query the original dimensions).
      auto replaceIf = [&](OpOperand &use) {
        Operation *user = use.getOwner();
        if (user == reshapes->expandShapeOp) {
          return false;
        }
        if (isa<tensor::DimOp>(user)) {
          return false;
        }
        return true;
      };
      rewriter.replaceUsesWithIf(result, reshapes->collapseShapeOp.getResult(),
                                 replaceIf);
    }
  }

  return success();
}

static LogicalResult expandIterationSpace(RewriterBase &rewriter,
                                          Operation *operation) {
  if (auto genericOp = dyn_cast<linalg::GenericOp>(operation))
    return expandIterationSpace(rewriter, genericOp);
  return success();
}

void GPUExpandDimensionsPass::runOnOperation() {
  Operation *operation = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  auto walkResult = operation->walk([&](Operation *op) -> WalkResult {
    rewriter.setInsertionPoint(op);
    return expandIterationSpace(rewriter, op);
  });

  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After expanding dimensions:\n";
    operation->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  ConfigTrackingListener listener;
  GreedyRewriteConfig config;
  config.setListener(&listener);

  {
    RewritePatternSet bubbleExpandShapePatterns(context);
    linalg::ControlFusionFn controlFn = [](OpOperand *opOperand) {
      return !isa_and_nonnull<linalg::FillOp, tensor::EmptyOp>(
          opOperand->get().getDefiningOp());
    };
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
    populateCombineRelayoutOpPatterns(bubbleExpandShapePatterns);
    populateFoldTensorReshapeIntoBufferPatterns(bubbleExpandShapePatterns);
    tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
    tensor::populateBubbleUpExpandShapePatterns(bubbleExpandShapePatterns);
    linalg::FillOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                                context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        bubbleExpandShapePatterns);
    if (failed(applyPatternsGreedily(
            operation, std::move(bubbleExpandShapePatterns), config))) {
      operation->emitOpError(
          "failed in application of bubble up expand shape patterns");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After reshape propagation:\n";
    operation->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  {
    RewritePatternSet removeBarrierOpsPatterns(context);
    removeBarrierOpsPatterns.insert<RemoveOptimizationBarrier>(context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(removeBarrierOpsPatterns,
                                                       context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        removeBarrierOpsPatterns, context);
    populateReshapeToInterfaceTensorPatterns(removeBarrierOpsPatterns);
    populateCombineRelayoutOpPatterns(removeBarrierOpsPatterns);
    populateFoldTensorReshapeIntoBufferPatterns(removeBarrierOpsPatterns);
    tensor::populateFoldTensorEmptyPatterns(removeBarrierOpsPatterns);
    linalg::FillOp::getCanonicalizationPatterns(removeBarrierOpsPatterns,
                                                context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        removeBarrierOpsPatterns);
    if (failed(applyPatternsGreedily(operation,
                                     std::move(removeBarrierOpsPatterns)))) {
      operation->emitOpError("failed in cleanup patterns");
      return signalPassFailure();
    }
    moveUpMemrefReshapeOps(rewriter, operation);
  }

  return;
}

} // namespace mlir::iree_compiler
