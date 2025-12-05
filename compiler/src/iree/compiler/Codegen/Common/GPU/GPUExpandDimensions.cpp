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
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/DebugLog.h"
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

struct GPUExpandDimensionsPass final
    : impl::GPUExpandDimensionsPassBase<GPUExpandDimensionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

using DimensionExpansionInfo = llvm::SmallDenseMap<unsigned, int64_t>;

static DimensionExpansionInfo
getExpansionInfo(IREE::GPU::DimensionExpansionAttr config) {
  DimensionExpansionInfo expansionInfo;

  auto reassociationIndices = config.getReassociationIndices();
  auto outputShape = config.getOutputShape();

  auto computeExpansion = [&](ArrayRef<int64_t> indices) {
    return llvm::product_of(llvm::make_filter_range(
        llvm::map_range(indices, [&](int64_t i) { return outputShape[i]; }),
        [](int64_t size) { return !ShapedType::isDynamic(size); }));
  };

  for (ReassociationIndices indices : reassociationIndices) {
    if (indices.size() <= 1)
      continue;

    expansionInfo[indices.front()] = computeExpansion(indices);
  }

  for (auto [dim, factor] : expansionInfo) {
    LDBG() << "Dimension " << dim << " will be expanded by factor " << factor;
  }

  return expansionInfo;
}

static LogicalResult expandIterationSpace(RewriterBase &rewriter,
                                          linalg::LinalgOp op) {
  auto dimensionExpansionConfig =
      IREE::GPU::getDimensionExpansion<IREE::GPU::DimensionExpansionAttr>(op);
  if (!dimensionExpansionConfig) {
    return success();
  }

  DimensionExpansionInfo expansionInfo =
      getExpansionInfo(dimensionExpansionConfig);
  if (expansionInfo.empty()) {
    return success();
  }

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  for (auto [iterDim, factor] : expansionInfo) {
    if (factor < 1) {
      return op.emitError("invalid expansion factor ") << factor;
    }

    if (iterDim >= loopRanges.size()) {
      return op.emitOpError("expand_dims dimension ")
             << iterDim << " out of bounds";
    }

    // TODO: Support expansion of dynamic/unaligned shapes.
    int64_t dimSize = loopRanges[iterDim];
    if (ShapedType::isDynamic(dimSize)) {
      return op.emitOpError("dimension ")
             << iterDim
             << " is dynamic, but expand_dims requires static dimensions";
    }

    if (dimSize % factor != 0) {
      return op.emitOpError("dimension ")
             << iterDim << " (size=" << dimSize
             << ") not divisible by expansion factor " << factor;
    }
  }

  LDBG() << "Expanding dimensions for op: " << *op;

  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();

  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<RankedTensorType>(operand.get().getType())) {
      continue;
    }

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

  SmallVector<Operation *> worklist;
  operation->walk([&](Operation *op) {
    if (IREE::GPU::getDimensionExpansion<IREE::GPU::DimensionExpansionAttr>(
            op)) {
      worklist.push_back(op);
    }
  });

  for (Operation *op : worklist) {
    rewriter.setInsertionPoint(op);
    if (failed(expandIterationSpace(rewriter, op))) {
      return signalPassFailure();
    }
  }

  LDBG() << "After expanding dimensions: " << *operation;

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

  LDBG() << "After reshape propagation: " << *operation;

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
