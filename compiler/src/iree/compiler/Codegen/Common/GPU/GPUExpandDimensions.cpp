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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
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
getExpansionInfo(IREE::GPU::LoweringConfigAttr config, linalg::LinalgOp op) {
  // Get expand_dims structure
  SmallVector<SmallVector<int64_t>> expansionFactors =
      IREE::GPU::getDimensionExpansion(config).value();
  SmallVector<int64_t> threadSizes = config.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Thread), op);

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

  LLVM_DEBUG(for (auto [dim, factor]
                  : expansionInfo) {
    llvm::dbgs() << "Dimension " << dim << " will be expanded by factor "
                 << factor << "\n";
  });

  return expansionInfo;
}

static LogicalResult expandIterationSpace(RewriterBase &rewriter,
                                          linalg::LinalgOp op) {
  auto loweringConfig = getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
  if (!loweringConfig) {
    return success();
  }

  if (failed(IREE::GPU::getDimensionExpansion(loweringConfig))) {
    return success();
  }

  DimensionExpansionInfo expansionInfo = getExpansionInfo(loweringConfig, op);
  if (expansionInfo.empty()) {
    return success();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Expanding dimensions for op:\n";
    op->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();

  for (OpOperand &operand : op->getOpOperands()) {
    if (operand.get().getDefiningOp<tensor::CollapseShapeOp>())
      continue;
    if (!isa<RankedTensorType>(operand.get().getType()))
      continue;

    // Translate iteration space expansion to tensor space expansion
    AffineMap indexingMap = indexingMaps[operand.getOperandNumber()];
    DimensionExpansionInfo tensorExpansionInfo;

    for (auto [iterDim, factor] : expansionInfo) {
      AffineExpr iterExpr = getAffineDimExpr(iterDim, op.getContext());
      if (std::optional<unsigned> tensorDim =
              indexingMap.getResultPosition(iterExpr)) {
        tensorExpansionInfo[tensorDim.value()] = factor;
      }
    }

    if (tensorExpansionInfo.empty()) {
      continue;
    }

    std::optional<ReshapeOps> reshapes = createDimensionExpansionOps(
        rewriter, tensorExpansionInfo, operand.get());
    if (reshapes) {
      rewriter.modifyOpInPlace(
          op, [&]() { operand.set(reshapes->collapseShapeOp); });
    }
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);

  for (OpResult result : op->getResults()) {
    if (!isa<RankedTensorType>(result.getType())) {
      continue;
    }

    unsigned resultMapIndex = op.getNumDpsInputs() + result.getResultNumber();
    AffineMap indexingMap = indexingMaps[resultMapIndex];
    DimensionExpansionInfo tensorExpansionInfo;

    for (auto [iterDim, factor] : expansionInfo) {
      AffineExpr iterExpr = getAffineDimExpr(iterDim, op.getContext());
      if (std::optional<unsigned> tensorDim =
              indexingMap.getResultPosition(iterExpr)) {
        tensorExpansionInfo[tensorDim.value()] = factor;
      }
    }

    if (tensorExpansionInfo.empty()) {
      continue;
    }

    std::optional<ReshapeOps> reshapes =
        createDimensionExpansionOps(rewriter, tensorExpansionInfo, result);
    if (reshapes) {
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
  if (auto op = dyn_cast<linalg::LinalgOp>(operation)) {
    return expandIterationSpace(rewriter, op);
  }
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
    populateReshapePropagationPatterns(bubbleExpandShapePatterns, controlFn);
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
    populateRemoveOptimizationBarrierPatterns(removeBarrierOpsPatterns);
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
