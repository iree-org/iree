// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
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

static FailureOr<SmallVector<OpFoldResult>>
computeExpandedGroupShape(RewriterBase &rewriter, Location loc,
                          OpFoldResult origDimSize,
                          ArrayRef<int64_t> groupTargetShape,
                          unsigned iteratorDim, linalg::LinalgOp op) {
  if (groupTargetShape.size() == 1) {
    return SmallVector<OpFoldResult>{origDimSize};
  }

  std::optional<int64_t> staticOrigDim = getConstantIntValue(origDimSize);
  if (!staticOrigDim) {
    return op.emitOpError("dimension ")
           << iteratorDim
           << " is dynamic, but expand_dims requires static dimensions";
  }

  int64_t staticFactor = llvm::product_of(llvm::make_filter_range(
      groupTargetShape, [](int64_t s) { return !ShapedType::isDynamic(s); }));

  if (staticFactor < 1) {
    return op.emitOpError("invalid expansion factor ")
           << staticFactor << " for iterator dimension " << iteratorDim;
  }

  if (staticOrigDim.value() % staticFactor != 0) {
    return op.emitOpError("dimension ")
           << iteratorDim << " (size=" << staticOrigDim.value()
           << ") not divisible by expansion factor " << staticFactor;
  }

  return llvm::map_to_vector(
      groupTargetShape, [&](int64_t size) -> OpFoldResult {
        if (!ShapedType::isDynamic(size)) {
          return rewriter.getIndexAttr(size);
        }
        AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
        return affine::makeComposedFoldedAffineApply(
            rewriter, loc, s0.floorDiv(staticFactor), {origDimSize});
      });
}

static FailureOr<ReshapeOps>
createDimensionExpansionOps(RewriterBase &rewriter,
                            IREE::GPU::DimensionExpansionAttr config, Value v,
                            AffineMap indexingMap, linalg::LinalgOp op) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return failure();
  }

  Location loc = v.getLoc();
  MLIRContext *ctx = op.getContext();
  ArrayRef<int64_t> preprocessedOutputShape =
      config.getOutputShape().asArrayRef();
  SmallVector<OpFoldResult> origShape = tensor::getMixedSizes(rewriter, loc, v);

  SmallVector<OpFoldResult> expandedTensorShape;
  SmallVector<ReassociationIndices> tensorReassociation;
  int64_t expandedDimIdx = 0;

  for (auto [origDimIdx, expandedReassocIdx] :
       llvm::enumerate(config.getReassociationIndices())) {
    AffineExpr dimExpr = getAffineDimExpr(origDimIdx, ctx);
    std::optional<unsigned> tensorDim = indexingMap.getResultPosition(dimExpr);
    if (!tensorDim) {
      continue;
    }

    int64_t start = expandedDimIdx;
    int64_t count = static_cast<int64_t>(expandedReassocIdx.size());
    auto tensorGroup =
        llvm::to_vector(llvm::seq<int64_t>(start, start + count));
    tensorReassociation.push_back(std::move(tensorGroup));
    expandedDimIdx += count;

    auto preprocessedGroupOutputShape =
        llvm::map_to_vector(expandedReassocIdx, [&](int64_t i) {
          return preprocessedOutputShape[i];
        });

    FailureOr<SmallVector<OpFoldResult>> groupShape =
        computeExpandedGroupShape(rewriter, loc, origShape[tensorDim.value()],
                                  preprocessedGroupOutputShape, origDimIdx, op);
    if (failed(groupShape)) {
      return failure();
    }
    llvm::append_range(expandedTensorShape, groupShape.value());
  }

  auto staticExpandedShape =
      llvm::map_to_vector(expandedTensorShape, [](OpFoldResult ofr) {
        return getConstantIntValue(ofr).value_or(ShapedType::kDynamic);
      });

  auto expandedType =
      RankedTensorType::get(staticExpandedShape, tensorType.getElementType(),
                            tensorType.getEncoding());

  auto expandShapeOp = tensor::ExpandShapeOp::create(
      rewriter, loc, expandedType, v, tensorReassociation, expandedTensorShape);

  Value barrier = IREE::Util::OptimizationBarrierOp::create(
                      rewriter, loc, expandShapeOp.getResult())
                      .getResult(0);

  auto collapseShapeOp = tensor::CollapseShapeOp::create(
      rewriter, loc, tensorType, barrier, tensorReassociation);

  return ReshapeOps{expandShapeOp, collapseShapeOp};
}

static LogicalResult expandIterationSpace(RewriterBase &rewriter,
                                          linalg::LinalgOp op) {
  auto config =
      IREE::GPU::getDimensionExpansion<IREE::GPU::DimensionExpansionAttr>(op);
  if (!config) {
    return success();
  }

  LDBG() << "Expanding dimensions for op: " << *op;

  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();

  for (OpOperand &operand : op->getOpOperands()) {
    AffineMap indexingMap = indexingMaps[operand.getOperandNumber()];
    FailureOr<std::optional<ReshapeOps>> reshapes = createDimensionExpansionOps(
        rewriter, config, operand.get(), indexingMap, op);
    if (failed(reshapes)) {
      return failure();
    }
    if (reshapes->has_value()) {
      rewriter.modifyOpInPlace(
          op, [&]() { operand.set(reshapes->value().collapseShapeOp); });
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
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
    populateCombineRelayoutOpPatterns(bubbleExpandShapePatterns);
    populateFoldTensorReshapeIntoBufferPatterns(bubbleExpandShapePatterns);
    tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
    tensor::populateBubbleUpExpandShapePatterns(bubbleExpandShapePatterns);
    linalg::FillOp::getCanonicalizationPatterns(
        bubbleExpandShapePatterns, bubbleExpandShapePatterns.getContext());
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        bubbleExpandShapePatterns);
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
