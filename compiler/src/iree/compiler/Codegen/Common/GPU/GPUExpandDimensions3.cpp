// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
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

/// This pass expands dimensions of tensor operands based on the lowering
/// configuration's ExpandDims tiling level. Unlike BlockDynamicDimensions,
/// this works on both static and dynamic dimensions and is driven by the
/// lowering config, not by divisibility analysis.
///
/// For example, if a config specifies:
///   expand_dims = [[0], [1,2]]
///   thread = [0, 1, 8]  (was previously [0, 8])
///
/// Then dimension 1 will be expanded by thread[2] = 8:
///   tensor<4096x4096xf16> -> tensor<4096x512x8xf16>
///
/// The expand_dims uses the same reassociation format as tensor.expand_shape:
/// - [0] means dimension 0 stays as dimension 0
/// - [1,2] means dimension 1 expands into dimensions 1 and 2
/// The expansion factors come from the thread array at the target positions.
///
/// The expansion is done using the same expand/barrier/collapse pattern
/// as BlockDynamicDimensions, and uses the same propagation patterns.
struct GPUExpandDimensionsPass final
    : impl::GPUExpandDimensionsPassBase<GPUExpandDimensionsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Represents information about how to expand a single dimension.
/// Maps dimension index -> (expansion factors, thread positions)
/// e.g., for expand_dims = [[0], [1,2]] with thread=[0, 1, 8]:
///   source dimension 1 expands into positions [1,2]
///   {sourceDim=1, factors=[1, 8], targetDims=[1,2]}
struct DimensionExpansionInfo {
  int64_t sourceDim;           // Original dimension to expand
  SmallVector<int64_t> factors; // Expansion factors (e.g., [4, 8])
  SmallVector<int64_t> targetDims; // Target dimensions after expansion
};
using TensorExpansionInfo = SmallVector<DimensionExpansionInfo>;

/// Extract expansion information from the ExpandDims tiling level in the
/// lowering config. Returns expansion info for each dimension that needs
/// to be expanded.
static std::optional<TensorExpansionInfo>
getTensorExpansionInfo(IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
                       Value v) {
  if (!loweringConfig) {
    return std::nullopt;
  }

  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    return std::nullopt;
  }

  // Check if config has ExpandDims level
  unsigned expandDimsLevel = llvm::to_underlying(IREE::GPU::TilingLevel::ExpandDims);
  if (!loweringConfig.hasTilingLevel(expandDimsLevel)) {
    return std::nullopt;
  }

  // Get the GPU lowering config to access the expand_dims attribute
  auto gpuConfig = dyn_cast<IREE::GPU::LoweringConfigAttr>(loweringConfig);
  if (!gpuConfig) {
    return std::nullopt;
  }

  // Get expand_dims from the config dictionary
  auto expandDimsAttr = gpuConfig.getAttributes().getAs<ArrayAttr>("expand_dims");
  if (!expandDimsAttr) {
    return std::nullopt;
  }

  // Get thread tile sizes to determine expansion factors
  SmallVector<int64_t> threadSizes = loweringConfig.getStaticTilingLevelSizes(
      llvm::to_underlying(IREE::GPU::TilingLevel::Thread), nullptr);
  if (threadSizes.empty()) {
    return std::nullopt;
  }

  TensorExpansionInfo expansionInfo;

  // Parse expand_dims as reassociation indices (same format as tensor.expand_shape)
  // Example: expand_dims = [[0], [1,2]], thread = [0, 1, 8]
  // - [0] means dimension 0 stays as dimension 0 (not split)
  // - [1,2] means dimension 1 (source) expands into positions 1 and 2 (target)
  // - Factors come from thread[1]=1 and thread[2]=8
  // - Dimension 1 (4096) expands to [512, 1, 8] → [512, 8] after filtering 1s
  for (auto [sourceDim, reassocGroup] : llvm::enumerate(expandDimsAttr)) {
    auto groupArray = dyn_cast<ArrayAttr>(reassocGroup);
    if (!groupArray) {
      continue;
    }

    // If group has only one element, this dimension doesn't split
    if (groupArray.size() <= 1) {
      continue;
    }

    // Group has multiple elements - this dimension needs to be expanded
    SmallVector<int64_t> targetDims;
    SmallVector<int64_t> factors;

    for (auto targetDimAttr : groupArray) {
      auto intAttr = dyn_cast<IntegerAttr>(targetDimAttr);
      if (!intAttr) {
        continue;
      }
      int64_t targetDim = intAttr.getInt();
      targetDims.push_back(targetDim);

      // Get the expansion factor from thread sizes at this target position
      if (targetDim < threadSizes.size()) {
        int64_t factor = threadSizes[targetDim];
        factors.push_back(factor);
      }
    }

    if (targetDims.empty() || factors.empty()) {
      continue;
    }

    expansionInfo.push_back(DimensionExpansionInfo{
        static_cast<int64_t>(sourceDim), factors, targetDims});
  }

  return expansionInfo.empty() ? std::nullopt
                                : std::make_optional(expansionInfo);
}

/// For a value `v`, insert expand/barrier/collapse pattern based on
/// expansion info from the lowering config.
///
/// ```mlir
/// %v_expand = tensor.expand_shape %v
/// %barrier = util.optimization.barrier %v_expand
/// %v_collapse = tensor.collapse_shape %barrier
/// ```
///
/// where the generated `tensor.expand_shape` and `tensor.collapse_shape` are
/// inverses of each other. The `util.optimization.barrier` avoids these from
/// getting folded away during reshape propagation. Return the result of the
/// `tensor.collapse_shape` generated.
struct ReshapeOps {
  tensor::ExpandShapeOp expandShapeOp;
  tensor::CollapseShapeOp collapseShapeOp;
};
static std::optional<ReshapeOps>
expandDimensionsOfValue(RewriterBase &rewriter,
                        const TensorExpansionInfo &expansionInfo,
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

    // Find if this dimension should be expanded
    auto expansionIt = llvm::find_if(
        expansionInfo, [index = index](const DimensionExpansionInfo &info) {
          return info.sourceDim == static_cast<int64_t>(index);
        });

    if (expansionIt == expansionInfo.end()) {
      reassociation.back().push_back(outputShape.size());
      outputShape.push_back(dim);
      continue;
    }

    // Expand this dimension according to config
    const DimensionExpansionInfo &dimInfo = *expansionIt;

    // For expansion like d1 -> [d1_outer, 8], we need:
    // - newDim = originalDim / 8 (or affine.ceildiv for safety with dynamic)
    // - staticDim = 8
    // We support multiple factors: d1 -> [d1_outer, 4, 8] means divide by 4*8=32

    // Compute total expansion factor
    int64_t totalFactor = 1;
    for (int64_t factor : dimInfo.factors) {
      totalFactor *= factor;
    }

    // First dimension: outer dynamic/static part
    reassociation.back().push_back(outputShape.size());

    if (tensorType.isDynamicDim(index)) {
      // Dynamic dimension: compute outer_dim = original_dim / total_factor
      AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
      AffineExpr divExpr = s0.ceilDiv(totalFactor); // Use ceildiv for safety
      OpFoldResult outerDim = affine::makeComposedFoldedAffineApply(
          rewriter, loc, divExpr, ArrayRef<OpFoldResult>{dim});
      outputShape.push_back(outerDim);
    } else {
      // Static dimension: compute at compile time
      int64_t staticSize = tensorType.getDimSize(index);
      assert(staticSize % totalFactor == 0 &&
             "static dimension not divisible by expansion factors");
      outputShape.push_back(rewriter.getIndexAttr(staticSize / totalFactor));
    }

    // Add the static expansion dimensions (skip factors of 1)
    for (int64_t factor : dimInfo.factors) {
      if (factor > 1) {
        reassociation.back().push_back(outputShape.size());
        outputShape.push_back(rewriter.getIndexAttr(factor));
      }
    }
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

//===---------------------------------------------------------------------===//
// Methods for blocking operands of operations
//===---------------------------------------------------------------------===//

/// For an operation, replace the operands at indices specified in
/// `limitToOperandNumbers` with the result of
/// `tensor.expand_shape`/`tensor.collapse_shape` pair to materialize the
/// information about dynamic dimensions that are known to be a multiple of a
/// compile-time static value. For example,
///
/// ```mlir
/// %1 = <some_op>(..., %0, ...) : ... , tensor<4x?x6xf32>
/// ```
///
/// If the dynamic dimension is known to be a multiple of 16, then generate
///
/// ```mlir
/// %expanded = tensor.expand_shape %0 :
///    tensor<4x?x5xf32> into tensor<4x?x16x6xf32>
/// %barrier = util.optimization.barrier %expanded
/// %collapsed = tensor.collapse_shape %barrier
///     : tensor<4x?x16x5xf32> into tensor<4x?x5xf32>
/// %1 = <some_op>(..., %collaped, ...) : ... , tensor<4x?x6xf32>
/// ```
static LogicalResult expandDimensions(
    RewriterBase &rewriter,
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
    Operation *operation, llvm::SmallDenseSet<int64_t> limitToOperandNumbers,
    llvm::SmallDenseSet<int64_t> limitToResultNumbers) {
  for (OpOperand &operand : operation->getOpOperands()) {
    if (!limitToOperandNumbers.contains(operand.getOperandNumber()))
      continue;
    std::optional<TensorExpansionInfo> operandExpansionInfo =
        getTensorExpansionInfo(loweringConfig, operand.get());
    if (!operandExpansionInfo)
      continue;
    std::optional<ReshapeOps> reshapes = expandDimensionsOfValue(
        rewriter, *operandExpansionInfo, operand.get());
    if (reshapes) {
      rewriter.modifyOpInPlace(
          operation, [&]() { operand.set(reshapes->collapseShapeOp); });
    }
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(operation);
  for (OpResult result : operation->getResults()) {
    if (!limitToResultNumbers.contains(result.getResultNumber()))
      continue;
    std::optional<TensorExpansionInfo> resultExpansionInfo =
        getTensorExpansionInfo(loweringConfig, result);
    if (!resultExpansionInfo)
      continue;
    std::optional<ReshapeOps> reshapes =
        expandDimensionsOfValue(rewriter, *resultExpansionInfo, result);
    if (reshapes) {
      llvm::SmallPtrSet<Operation *, 1> ignoreUses;
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

/// Generic method for expanding all tensor operands and results of an operation.
static LogicalResult expandDimensionsOfAllTensorOperandsAndResults(
    RewriterBase &rewriter,
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
    Operation *op) {
  llvm::SmallDenseSet<int64_t> tensorOperandsList, tensorResultsList;
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (isa<RankedTensorType>(opOperand.get().getType())) {
      tensorOperandsList.insert(opOperand.getOperandNumber());
    }
  }
  for (OpResult result : op->getResults()) {
    if (isa<RankedTensorType>(result.getType())) {
      tensorResultsList.insert(result.getResultNumber());
    }
  }
  return expandDimensions(rewriter, loweringConfig, op,
                          tensorOperandsList, tensorResultsList);
}

/// Expand dimensions in operands of `LinalgOp` based on lowering config.
static LogicalResult
expandDimensions(RewriterBase &rewriter, linalg::LinalgOp linalgOp) {
  // Only expand if the op has a lowering config with ExpandDims level
  IREE::Codegen::LoweringConfigAttrInterface loweringConfig =
      getLoweringConfig(linalgOp);
  if (!loweringConfig) {
    return success();
  }

  unsigned expandDimsLevel =
      llvm::to_underlying(IREE::GPU::TilingLevel::ExpandDims);
  if (!loweringConfig.hasTilingLevel(expandDimsLevel)) {
    return success();
  }

  // Expand all tensor operands and results
  if (isa<linalg::GenericOp>(linalgOp)) {
    return expandDimensionsOfAllTensorOperandsAndResults(
        rewriter, loweringConfig, linalgOp);
  }
  return success();
}

/// Dispatch to methods that expand dimensions of operations.
static LogicalResult
expandDimensions(RewriterBase &rewriter, Operation *operation) {
  return TypeSwitch<Operation *, LogicalResult>(operation)
      .Case<linalg::LinalgOp>([&](auto linalgOp) {
        return expandDimensions(rewriter, linalgOp);
      })
      .Default([&](Operation *op) { return success(); });
}

void GPUExpandDimensionsPass::runOnOperation() {
  Operation *operation = getOperation();
  MLIRContext *context = &getContext();

  // Phase 1: Identify operations with ExpandDims config and expand them
  IRRewriter rewriter(context);
  auto walkResult = operation->walk([&](Operation *op) -> WalkResult {
    rewriter.setInsertionPoint(op);
    return expandDimensions(rewriter, op);
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After expanding dimensions:\n";
    operation->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n";
  });

  // Phase 2: Propagate reshapes through operations
  {
    RewritePatternSet bubbleExpandShapePatterns(context);
    // Add patterns to "push down" the `tensor.collapse_shape` patterns (which
    // are the dual of the patterns to "bubble up" `tensor.expand_shape`
    // patterns)
    linalg::ControlFusionFn controlFn = [](OpOperand *opOperand) {
      // Avoid fusion with fills/empty using the propagation patterns.
      return !isa_and_nonnull<linalg::FillOp, tensor::EmptyOp>(
          opOperand->get().getDefiningOp());
    };
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                      controlFn);
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        bubbleExpandShapePatterns, controlFn);
    // Add patterns to fold the "bubbled-up" `tensor.expand_shape` operation and
    // "pushed-down" `tensor.collapse_shape` operation with their interface
    // bindings or `tensor.empty` operations.
    populateReshapeToInterfaceTensorPatterns(bubbleExpandShapePatterns);
    populateCombineRelayoutOpPatterns(bubbleExpandShapePatterns);
    populateFoldTensorReshapeIntoBufferPatterns(bubbleExpandShapePatterns);
    tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
    tensor::populateBubbleUpExpandShapePatterns(bubbleExpandShapePatterns);
    linalg::FillOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                                context);
    // Add some additional patterns that can simplify the IR and remove dead
    // operations.
    memref::populateResolveRankedShapedTypeResultDimsPatterns(
        bubbleExpandShapePatterns);
    if (failed(applyPatternsGreedily(operation,
                                     std::move(bubbleExpandShapePatterns)))) {
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

  // Phase 3: Delete the optimization barrier and run cleanup
  {
    RewritePatternSet removeBarrierOpsPatterns(context);
    removeBarrierOpsPatterns.insert<RemoveOptimizationBarrier>(context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(removeBarrierOpsPatterns,
                                                       context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        removeBarrierOpsPatterns, context);
    // Add patterns to fold the remaining reshape operation with their interface
    // bindings or `tensor.empty` operations.
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
